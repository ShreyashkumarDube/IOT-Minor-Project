"""World L's controller: the LLM makes its OWN trust estimates and decisions, and World L enforces them.

Isolation rules (see docs/ARCHITECTURE.md):
* Nothing computed by the Bayesian controller ever reaches this module. It has its own
  ledger, nonce memory, sliding window, baselines, quarantine and probation state.
* The LLM context is built ONLY from this controller's own observations (facts, not
  ready-made anomaly labels), the device's public attributes, the LLM's own previous
  assessment, topology and the policy thresholds as guidance.
* Between evaluations a device keeps the LLM's last decision. Before its first
  evaluation it is PENDING and gets the declared default (ALLOW if the ZK check passes).
* An LLM failure (timeout, unreachable, invalid JSON, invalid schema) is recorded as
  LLM_ERROR; the last LLM decision stays in force; there is never a Bayesian fallback.
* Lockstep pacing: requests are processed during the tick; at the end of the tick the
  devices that are due are evaluated concurrently and the tick waits for them.
"""
from __future__ import annotations

import json
import math
import os
import time
from collections import Counter, deque
from concurrent.futures import ThreadPoolExecutor, wait
from dataclasses import dataclass, field
from typing import Any, Callable, Deque, Dict, List, Optional, Set, Tuple

from . import live_config as C
from .events import CONTROLLER_ID, EventType, ObservedEvent, is_night
from .llm_engine import LLMUnavailable
from .zk_auth import verify

ACTIONS = ('ALLOW', 'STEP-UP', 'BLOCK')
PENDING, LLM_ERROR = 'PENDING', 'LLM_ERROR'
DECISION_BAND = {'ALLOW': 'Trusted', 'STEP-UP': 'Suspicious', 'BLOCK': 'Isolated', PENDING: 'Pending'}
PUBLIC_PROFILE_FIELDS = ('type', 'protocol', 'firmware')   # public attributes; never name, role, status

LIVE_SYSTEM_PROMPT = """You are the Trust Evaluation and Policy Engine of an IoT zero-trust controller.
You receive ONLY observed facts about ONE device over a recent window: its own request counts,
destinations, ports, payload sizes, zero-knowledge (ZK) proof results, nonce reuse, sensor alarm
counts, step-up results and its own learned baseline. Counters that are zero or empty are omitted.
No label tells you whether the device is an attacker; infer it from the facts. Compare the window with
the device's own baseline and its configured links. ZK failures and nonce reuse are strong evidence of
compromise; a valid proof alone does not make a device trustworthy. Legitimate devices occasionally
update firmware, reboot (one or two failed proofs) and perform maintenance access.
Decide one action: ALLOW (normal), STEP-UP (require a fresh proof + MFA) or BLOCK (quarantine the
device). policy_guidance describes how trust usually maps to actions, but the decision is yours.
Do not invent facts that are not in the input.
Return ONLY a JSON object with these keys:
trust_score (number 0..1, probability the device is legitimate), action ("ALLOW"|"STEP-UP"|"BLOCK"),
confidence (number 0..1), reason (ONE short sentence),
factors (at most 4 short strings naming the observed facts you relied on)."""

# JSON schema for constrained output (Ollama `format`, OpenAI-compatible `response_format: json_schema`).
LIVE_RESPONSE_SCHEMA = {
    'type': 'object',
    'properties': {
        'trust_score': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'action': {'type': 'string', 'enum': list(ACTIONS)},
        'confidence': {'type': 'number', 'minimum': 0, 'maximum': 1},
        'reason': {'type': 'string'},
        'factors': {'type': 'array', 'items': {'type': 'string'}, 'maxItems': 4},
    },
    'required': ['trust_score', 'action', 'confidence', 'reason', 'factors'],
}
if C.LLM_ASK_BEHAVIOR_ASSESSMENT:   # optional; off by default to keep replies short
    LIVE_RESPONSE_SCHEMA['properties']['behavior_assessment'] = {
        'type': 'array', 'items': {'type': 'object', 'properties': {'behavior': {'type': 'string'}, 'risk': {'type': 'number'}}}}


def compact_facts(value):
    """Drop zero counters and empty dicts / lists (recursively); keep everything else unchanged."""
    if isinstance(value, dict):
        out = {k: compact_facts(v) for k, v in value.items()}
        return {k: v for k, v in out.items() if not _empty(v)}
    if isinstance(value, list):
        return [compact_facts(v) for v in value]
    return value


def _empty(v) -> bool:
    if isinstance(v, bool):
        return False                      # a False flag is information, not an empty counter
    if isinstance(v, (int, float)):
        return v == 0
    return v is None or v == {} or v == []


def make_live_evaluator(provider: str, model: str, endpoint: str, api_key: str = '', timeout: float = C.LLM_TIMEOUT_S,
                        max_tokens: Optional[int] = C.LLM_MAX_OUTPUT_TOKENS, keep_alive: Optional[str] = C.LLM_KEEP_ALIVE,
                        num_ctx: Optional[int] = C.LLM_NUM_CTX, think: Optional[bool] = C.LLM_THINK,
                        temperature: float = C.LLM_TEMPERATURE, rpm: int = C.GEMINI_MAX_RPM):
    """The Live-mode LLM client: compact JSON, schema-constrained reply, short output, model kept loaded.
    `provider` is 'ollama', 'gemini' (Google Gemini API) or 'openai_compatible' (e.g. llama.cpp's llama-server)."""
    from .llm_engine import LLMConfig, LLMEvaluator
    if provider == 'gemini':
        from .gemini_engine import GeminiEvaluator
        return GeminiEvaluator(model=model, api_key=api_key or None, system_prompt=LIVE_SYSTEM_PROMPT,
                               response_schema=LIVE_RESPONSE_SCHEMA, max_output_tokens=max_tokens, timeout=timeout, rpm=rpm)
    cfg = LLMConfig(provider, model, endpoint, api_key, timeout, temperature, max_tokens=max_tokens,
                    keep_alive=keep_alive if provider == 'ollama' else None, num_ctx=num_ctx if provider == 'ollama' else None,
                    think=think if provider == 'ollama' else None, compact_json=True)
    return LLMEvaluator(cfg, system_prompt=LIVE_SYSTEM_PROMPT, response_schema=LIVE_RESPONSE_SCHEMA)


def resolve_llm_args(args):
    """Command-line defaults per provider (run_batch / llm_speed_check). Adds `llm_note` = provider + model ID."""
    if args.provider == 'gemini':
        from .gemini_engine import DEFAULT_MODEL, load_dotenv
        load_dotenv()
        args.model = args.model or os.getenv('GEMINI_MODEL') or DEFAULT_MODEL
        args.endpoint = 'Google Gemini API'
        args.api_key = os.getenv('GEMINI_API_KEY', '')
    else:
        args.model = args.model or os.getenv('OLLAMA_MODEL', 'llama3.2:3b')
        args.endpoint = args.endpoint or os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api/chat')
        args.api_key = os.getenv('LLM_API_KEY', '')
    args.llm_note = f'{args.model} ({args.provider}) @ {args.endpoint}'
    return args


TEST_CONTEXT = {'device': {'id': 'TEST', 'type': 'Sensor'}, 'window_facts': {'requests': 3, 'requests_per_tick': 1.0},
                'omitted_fields_mean_zero': True, 'note': 'connection test, not a real device'}


class InvalidLLMOutput(ValueError):
    """The LLM answered, but not in the required schema."""


def parse_llm_output(raw: Any) -> Tuple[dict, int]:
    """Strict validation. Returns (clean result, number of values clamped into [0, 1]).
    Missing/invalid trust_score or action -> InvalidLLMOutput (counted as LLM_ERROR, never defaulted)."""
    if not isinstance(raw, dict):
        raise InvalidLLMOutput('reply is not a JSON object')
    clamped = 0

    def unit(value, name, required):
        nonlocal clamped
        if value is None:
            if required:
                raise InvalidLLMOutput(f'missing {name}')
            return None
        try:
            x = float(value)
        except (TypeError, ValueError):
            raise InvalidLLMOutput(f'{name} is not a number: {value!r}')
        if math.isnan(x):
            raise InvalidLLMOutput(f'{name} is NaN')
        if not 0.0 <= x <= 1.0:
            clamped += 1
            x = max(0.0, min(1.0, x))
        return x

    trust = unit(raw.get('trust_score'), 'trust_score', True)
    action = str(raw.get('action', '')).strip().upper().replace('STEP_UP', 'STEP-UP').replace('STEPUP', 'STEP-UP')
    if action not in ACTIONS:
        raise InvalidLLMOutput(f'invalid action {raw.get("action")!r}')
    risk = unit(raw.get('risk_score'), 'risk_score', False)
    confidence = unit(raw.get('confidence'), 'confidence', False)
    factors = [str(f) for f in raw.get('factors', [])] if isinstance(raw.get('factors'), list) else []
    behaviors = raw.get('behavior_assessment') if isinstance(raw.get('behavior_assessment'), list) else []
    return ({'trust_score': trust, 'action': action, 'risk_score': risk if risk is not None else 1 - trust,
             'confidence': confidence, 'reason': str(raw.get('reason', '')), 'factors': factors,
             'behavior_assessment': behaviors}, clamped)


def raw_reply_text(raw: Any) -> Optional[str]:
    """The reply as the model sent it: the HTTP reply text if the evaluator kept it, else the returned object as JSON."""
    if isinstance(raw, BaseException):
        return None
    if isinstance(raw, dict) and isinstance(raw.get('_raw_text'), str):
        return raw['_raw_text']
    try:
        return json.dumps({k: v for k, v in raw.items() if not str(k).startswith('_')} if isinstance(raw, dict) else raw)
    except (TypeError, ValueError):
        return repr(raw)


def classify_error(exc: BaseException) -> str:
    """Map an exception to an error type for the metrics."""
    if isinstance(exc, InvalidLLMOutput):
        return 'invalid_schema'
    name = type(exc).__name__
    if name == 'GeminiQuotaExhausted':
        return 'quota_exhausted'
    if name == 'GeminiRateLimited':
        return 'rate_limited'
    if isinstance(exc, LLMUnavailable):
        return 'timeout' if 'timed out' in str(exc).lower() or 'timeout' in str(exc).lower() else 'unreachable'
    if isinstance(exc, TimeoutError):
        return 'timeout'
    if isinstance(exc, ValueError):
        return 'invalid_json'
    return 'error'


@dataclass
class LLMLedger:
    """Everything World L remembers about one device. Shares nothing with the Bayesian ledger."""
    device_id: str
    public_key: int
    zone: str
    sensitivity: float
    enrolled_tick: int = 0
    profile: Dict[str, str] = field(default_factory=dict)
    trust: Optional[float] = None            # None = PENDING (no LLM evaluation yet)
    decision: str = PENDING
    confidence: Optional[float] = None
    last_eval_tick: Optional[int] = None
    last_attempt_tick: Optional[int] = None    # last evaluation attempt (ok or LLM_ERROR), for the min-ticks rule
    last_reason: str = ''
    last_factors: List[str] = field(default_factory=list)
    last_error: Optional[str] = None
    seen_nonces: Set[str] = field(default_factory=set)
    window: Deque[dict] = field(default_factory=deque)
    warmup_requests: int = 0
    warmup_bytes: Dict[str, List[int]] = field(default_factory=dict)
    baseline_rate: Optional[float] = None
    baseline_bytes: Dict[str, float] = field(default_factory=dict)
    events_since_eval: int = 0
    due_reason: Optional[str] = None
    quarantined_until: Optional[int] = None
    probation: bool = False
    times_quarantined: int = 0

    @property
    def band(self) -> str:
        return DECISION_BAND[self.decision]

    @property
    def classification(self) -> str:
        if self.quarantined_until is not None:
            return 'Quarantined'
        if self.probation:
            return 'Probation'
        return self.band


class LLMController:
    """Same interface as the Bayesian ZeroTrustController, but the LLM decides."""
    engine = 'llm'

    def __init__(self, evaluator, every_n: int = C.LLM_EVERY_N, timeout_s: float = C.LLM_TIMEOUT_S,
                 max_workers: int = C.LLM_MAX_WORKERS, synchronous: bool = False,
                 min_ticks_between: int = C.LLM_MIN_TICKS_BETWEEN_EVALS, recent_requests: int = C.LLM_RECENT_REQUESTS,
                 compact_context: bool = C.LLM_COMPACT_CONTEXT):
        self.evaluator = evaluator
        self.every_n = max(1, int(every_n))
        self.min_ticks_between = max(0, int(min_ticks_between))
        self.recent_requests = max(0, int(recent_requests))
        self.compact_context = bool(compact_context)
        self.timeout_s = float(timeout_s)
        self.synchronous = synchronous   # True: call one by one in this thread (tests / reproducible batch)
        self.executor = None if synchronous else ThreadPoolExecutor(max_workers=max_workers)
        self.ledgers: Dict[str, LLMLedger] = {}
        self.edges: Set[Tuple[str, str]] = set()
        self.tick = 0
        self.allow_threshold, self.block_threshold = C.ALLOW_THRESHOLD, C.BLOCK_THRESHOLD  # guidance only
        self.evaluations: List[dict] = []
        self.errors: Counter = Counter()
        self.clamped = 0
        self.consecutive_unreachable = 0
        self.quota_exhausted = False   # Gemini per-day quota hit: offline for the rest of the run (the world keeps running)

    @property
    def model_name(self) -> str:
        """Which model answers (for display): the configured model, or the evaluator class (e.g. the FakeLLM)."""
        cfg = getattr(self.evaluator, 'config', None)
        if cfg is not None and getattr(cfg, 'model', None):
            return f'{cfg.model} ({cfg.provider})'
        return type(self.evaluator).__name__

    # ----- setup ------------------------------------------------------------------------
    def enroll(self, device_id: str, public_key: int, zone: str, target_sensitivity: float) -> None:
        self.ledgers[device_id] = LLMLedger(device_id, public_key, zone, float(target_sensitivity), enrolled_tick=self.tick)

    def register_profile(self, device_id: str, profile: Dict[str, str]) -> None:
        """Public, non-identifying attributes (type, protocol, firmware) for the LLM context."""
        self.ledgers[device_id].profile = {k: str(profile[k]) for k in PUBLIC_PROFILE_FIELDS if k in profile}

    def set_topology(self, connections) -> None:
        self.edges = {tuple(e) for e in connections}

    @property
    def offline(self) -> bool:
        return self.quota_exhausted or self.consecutive_unreachable >= C.LLM_OFFLINE_AFTER

    @property
    def offline_message(self) -> str:
        if self.quota_exhausted:
            return 'Gemini daily quota exhausted'
        return 'LLM offline' if self.offline else ''

    # ----- tick hooks ---------------------------------------------------------------------
    def begin_tick(self, tick: int) -> List[str]:
        """Release quarantines into probation and learn baselines (World L's own copy of these rules)."""
        self.tick = tick
        notes = []
        for led in self.ledgers.values():
            if led.quarantined_until is not None and tick >= led.quarantined_until:
                led.quarantined_until, led.probation = None, True
                led.due_reason = 'released from quarantine'
                notes.append(f'{led.device_id} released from quarantine → PROBATION (must pass step-up; LLM re-evaluates)')
            if led.baseline_rate is None and tick - led.enrolled_tick > C.WARMUP_TICKS:
                led.baseline_rate = led.warmup_requests / C.WARMUP_TICKS
                led.baseline_bytes = {k: round(sum(v) / len(v), 1) for k, v in led.warmup_bytes.items()}
        return notes

    def quarantine_end(self, device_id: str) -> Optional[int]:
        led = self.ledgers.get(device_id)
        return led.quarantined_until if led else None

    # ----- per request ------------------------------------------------------------------------
    def process(self, event: ObservedEvent, alarms: Dict[str, bool]) -> dict:
        """Record the observed facts and apply the device's CURRENT LLM decision to this request."""
        if not isinstance(event, ObservedEvent):
            raise TypeError('process() only accepts an ObservedEvent')
        if set(alarms) - set(C.SENSORS) or not all(isinstance(v, bool) for v in alarms.values()):
            raise TypeError('alarms must be {sensor name: bool} for the configured sensors only')
        led = self.ledgers[event.src]
        kind = EventType(event.event_type)
        port, size = event.details.get('port', 0), event.details.get('bytes', 0)
        base = {'tick': event.tick, 'src': event.src, 'dst': event.dst, 'event_type': kind.value, 'port': port,
                'bytes': size, 'trust_before': led.trust, 'trust_after': led.trust, 'band_before': led.band,
                'band_after': led.band, 'terms': [], 'challenge': None, 'step_up': None, 'quarantined_until': None,
                'nonce': event.nonce, 'llm_decision_tick': led.last_eval_tick}

        blocked_end = self.quarantine_end(event.src) or self.quarantine_end(event.dst)
        if blocked_end is not None:   # World L's own quarantine cut the link
            who = event.src if self.quarantine_end(event.src) is not None else event.dst
            return {**base, 'action': 'DROPPED', 'reason': f'{who} is quarantined (LLM world) until t={blocked_end}',
                    'kind_hint': 'dropped', 'derived': [], 'alarms': {}, 'zk': '—', 'replay': False,
                    'token_count': 0, 'proof': None}

        # Facts this controller observes itself.
        zk_ok = bool(event.proof) and verify(led.public_key, event.proof, event.src)
        replay = event.nonce is not None and event.nonce in led.seen_nonces
        if event.nonce is not None:
            led.seen_nonces.add(event.nonce)
        target = self.ledgers.get(event.dst)
        linked = event.dst == CONTROLLER_ID or (event.src, event.dst) in self.edges
        fact = {'tick': event.tick, 'dst': event.dst, 'kind': kind.value, 'port': port, 'bytes': size,
                'zk_ok': zk_ok, 'nonce_reused': replay, 'linked': linked,
                'zone_to': target.zone if target else CONTROLLER_ID,
                'zone_changed': bool(target and target.zone != led.zone),
                'target_sensitivity': target.sensitivity if target else 0.0,
                'night': is_night(event.tick), 'alarms': sorted(k for k, v in alarms.items() if v), 'step_up': None}
        led.window.append(fact)
        while led.window and led.window[0]['tick'] <= event.tick - C.WINDOW_TICKS:
            led.window.popleft()
        if led.baseline_rate is None:
            led.warmup_requests += 1
            led.warmup_bytes.setdefault(kind.value, []).append(size)

        # Scheduling: every N processed requests, or immediately (end of tick) on a hard event.
        led.events_since_eval += 1
        if not zk_ok:
            led.due_reason = 'hard event: zk_auth_fail'
        elif replay:
            led.due_reason = 'hard event: token_replay'
        elif led.due_reason is None and led.events_since_eval >= self.every_n:
            led.due_reason = f'every {self.every_n} events'

        # Policy: ZK hard gate (cryptographic, both worlds), then the LLM's own decision.
        current = f'LLM decision {led.decision} from t={led.last_eval_tick}' if led.last_eval_tick is not None else \
            f'PENDING — default {C.PENDING_DEFAULT_ACTION} with ZK check'
        if led.last_error:
            current += f' (last evaluation: {LLM_ERROR} {led.last_error}; decision kept)'
        if not zk_ok:
            action, reason = 'BLOCK', 'ZK proof rejected (hard gate)'
        elif replay:
            action, reason = 'BLOCK', 'nonce replayed (hard gate)'
        elif led.probation:
            action, reason = 'STEP-UP', 'on probation after quarantine'
        elif led.decision == PENDING:
            action, reason = C.PENDING_DEFAULT_ACTION, current
        else:
            action, reason = led.decision, current
        derived = [n for n, hit in (('zk_auth_fail', not zk_ok), ('token_replay', replay), ('no_topology_edge', not linked)) if hit]
        public_proof = {k: event.proof[k] for k in ('commitment', 'challenge', 'response')} if event.proof else None
        return {**base, 'action': action, 'reason': reason, 'derived': derived, 'alarms': dict(alarms),
                'zk': 'PASS' if zk_ok else 'FAIL', 'replay': replay, 'proof': public_proof,
                'token_count': sum(1 for f in led.window if f['kind'] == 'token_request'),
                'challenge': f'{event.src}-challenge-{event.tick}-{event.nonce}' if action == 'STEP-UP' else None}

    def complete_step_up(self, record: dict, proof: Optional[dict], mfa_passed: bool) -> dict:
        """Verify a fresh proof on the challenge nonce plus the observed MFA outcome."""
        if record.get('action') != 'STEP-UP' or not record.get('challenge'):
            raise ValueError('record is not awaiting a step-up challenge')
        led = self.ledgers[record['src']]
        zk_ok = bool(proof) and proof.get('nonce') == record['challenge'] and verify(led.public_key, proof, led.device_id)
        passed = zk_ok and bool(mfa_passed)
        if led.window:
            led.window[-1]['step_up'] = 'passed' if passed else 'failed'
        record['band_before_step_up'] = led.band
        record['step_up'] = {'zk': 'PASS' if zk_ok else 'FAIL', 'mfa': 'PASS' if mfa_passed else 'FAIL', 'term': None,
                             'trust_before': led.trust, 'trust_after': led.trust}
        record['action'] = 'ALLOW' if passed else 'BLOCK'
        record['reason'] = ('step-up passed' if passed else 'step-up failed') + ' (LLM world)'
        if passed and led.probation:
            led.probation = False
            led.due_reason = led.due_reason or 'probation step-up passed'
        if not passed:
            led.due_reason = 'hard event: step_up_fail'
        return record

    # ----- end of tick: evaluate the devices that are due ------------------------------------------
    def end_tick(self, tick: int, progress: Optional[Callable[[int], None]] = None) -> List[dict]:
        """Evaluate all due devices concurrently; the tick waits for them (lockstep pacing)."""
        due = [d for d in sorted(self.ledgers) if self.ledgers[d].due_reason and self.ledgers[d].quarantined_until is None
               and self._may_evaluate(self.ledgers[d], tick)]
        if not due:
            return []
        if progress:
            progress(len(due))
        contexts = {d: self.build_context(d) for d in due}
        outcomes: Dict[str, Tuple[Any, float]] = {}
        if self.synchronous:
            for d in due:
                outcomes[d] = self._call(contexts[d])
        else:
            futures = {d: self.executor.submit(self._call, contexts[d]) for d in due}
            done, _ = wait(list(futures.values()), timeout=self.timeout_s)   # the tick waits at most the timeout
            for d, fut in futures.items():
                if fut in done:
                    outcomes[d] = fut.result()
                else:
                    fut.cancel()
                    outcomes[d] = (TimeoutError(f'no answer within {self.timeout_s:.0f} s'), self.timeout_s * 1000)
        return [self._apply(d, contexts[d], *outcomes[d]) for d in due]

    def _may_evaluate(self, led: LLMLedger, tick: int) -> bool:
        """Hard events are evaluated at once; otherwise wait until min_ticks_between ticks have passed since the
        device's last evaluation attempt (the device stays due and is evaluated at the first allowed tick)."""
        if led.due_reason.startswith('hard event') or led.last_attempt_tick is None:
            return True
        return tick - led.last_attempt_tick >= self.min_ticks_between

    def _call(self, context: dict) -> Tuple[Any, float]:
        """One LLM request (runs in a worker thread). Returns (raw reply or exception, latency ms)."""
        t0 = time.perf_counter()
        try:
            ask = getattr(self.evaluator, 'evaluate_raw', None) or self.evaluator.evaluate
            raw = ask(context)
        except BaseException as exc:  # noqa: BLE001 — every failure becomes LLM_ERROR
            return exc, (time.perf_counter() - t0) * 1000
        latency = raw.get('_latency_ms') if isinstance(raw, dict) else None
        return raw, float(latency) if latency is not None else (time.perf_counter() - t0) * 1000

    def _apply(self, device_id: str, context: dict, raw: Any, latency_ms: float) -> dict:
        led = self.ledgers[device_id]
        trigger, led.due_reason, led.events_since_eval = led.due_reason, None, 0
        led.last_attempt_tick = self.tick
        record = {'tick': self.tick, 'kind': 'llm_eval', 'src': device_id, 'trigger': trigger,
                  'latency_ms': round(latency_ms, 1), 'trust_before': led.trust, 'decision_before': led.decision,
                  'context': context, 'quarantined_until': None,
                  'raw_reply': raw_reply_text(raw), 'llm_model': self.model_name}   # display only
        if isinstance(raw, dict) and raw.get('_usage'):
            record['llm_usage'] = raw['_usage']   # display only: prompt / output token counts reported by the server
        try:
            if isinstance(raw, BaseException):
                raise raw
            result, clamped = parse_llm_output(raw)
        except BaseException as exc:  # noqa: BLE001
            kind = classify_error(exc)
            self.errors[kind] += 1
            led.last_error = kind
            self.consecutive_unreachable = self.consecutive_unreachable + 1 if kind == 'unreachable' else 0
            if kind == 'quota_exhausted':
                self.quota_exhausted = True
            record.update({'ok': False, 'error': kind, 'message': str(exc)[:200], 'llm_trust': None,
                           'llm_action': None, 'confidence': None, 'reason': '', 'factors': [],
                           'behavior_assessment': [], 'clamped': 0, 'trust_after': led.trust,
                           'decision_after': led.decision})
            self.evaluations.append(record)
            return record
        self.consecutive_unreachable = 0
        self.clamped += clamped
        led.trust, led.decision, led.confidence = result['trust_score'], result['action'], result['confidence']
        led.last_eval_tick, led.last_reason, led.last_factors, led.last_error = self.tick, result['reason'], result['factors'], None
        if result['action'] == 'BLOCK' and led.quarantined_until is None:
            led.quarantined_until = self.tick + C.QUARANTINE_TICKS
            led.probation = False
            led.times_quarantined += 1
            record['quarantined_until'] = led.quarantined_until
        record.update({'ok': True, 'error': None, 'llm_trust': result['trust_score'], 'llm_action': result['action'],
                       'confidence': result['confidence'], 'reason': result['reason'], 'factors': result['factors'],
                       'behavior_assessment': result['behavior_assessment'], 'clamped': clamped,
                       'trust_after': led.trust, 'decision_after': led.decision})
        self.evaluations.append(record)
        return record

    # ----- the LLM's view of a device: World L's own facts only ------------------------------------------
    def build_context(self, device_id: str) -> Dict[str, Any]:
        ctx = self._full_context(device_id)
        if self.compact_context:
            ctx['window_facts'] = compact_facts(ctx['window_facts'])
            for r in ctx['recent_requests']:
                if not r['sensor_alarms']:
                    del r['sensor_alarms']
            ctx['omitted_fields_mean_zero'] = True
        return ctx

    def _full_context(self, device_id: str) -> Dict[str, Any]:
        led = self.ledgers[device_id]
        window = list(led.window)
        span = max(1, min(C.WINDOW_TICKS, self.tick - led.enrolled_tick))
        max_bytes: Dict[str, int] = {}
        for f in window:
            max_bytes[f['kind']] = max(max_bytes.get(f['kind'], 0), f['bytes'])
        alarm_counts = Counter(a for f in window for a in f['alarms'])
        prev = None if led.last_eval_tick is None else {
            'tick': led.last_eval_tick, 'trust_score': led.trust, 'action': led.decision, 'confidence': led.confidence}
        return {
            'device': {'id': device_id, **led.profile, 'zone': led.zone, 'target_sensitivity': led.sensitivity},
            'configured_outgoing_links': sorted(d for s, d in self.edges if s == device_id),
            'current_tick': self.tick,
            'evaluation_trigger': led.due_reason,
            'enforcement_state': {'on_probation': led.probation, 'times_quarantined_so_far': led.times_quarantined},
            'previous_llm_assessment': prev,
            'window_ticks': C.WINDOW_TICKS,
            'window_facts': {
                'requests': len(window),
                'requests_per_tick': round(len(window) / span, 2),
                'baseline_requests_per_tick': led.baseline_rate,
                'request_kind_counts': dict(Counter(f['kind'] for f in window)),
                'token_requests': sum(f['kind'] == 'token_request' for f in window),
                'distinct_destinations': len({f['dst'] for f in window if f['dst'] != CONTROLLER_ID}),
                'distinct_ports': len({f['port'] for f in window}),
                'requests_without_configured_link': sum(not f['linked'] for f in window),
                'zone_changing_requests': sum(f['zone_changed'] for f in window),
                'requests_to_sensitive_targets': sum(f['target_sensitivity'] >= C.SENSITIVE_TARGET for f in window),
                'max_payload_bytes_by_kind': max_bytes,
                'baseline_mean_payload_bytes_by_kind': dict(led.baseline_bytes),
                'zk_failures': sum(not f['zk_ok'] for f in window),
                'nonce_reuse_detected': sum(f['nonce_reused'] for f in window),
                'night_requests': sum(f['night'] for f in window),
                'sensor_alarm_counts': {name: alarm_counts.get(name, 0) for name in C.SENSORS},
                'step_up_challenges': {'passed': sum(f['step_up'] == 'passed' for f in window),
                                       'failed': sum(f['step_up'] == 'failed' for f in window)},
            },
            'recent_requests': [{'tick': f['tick'], 'dst': f['dst'], 'kind': f['kind'], 'port': f['port'], 'bytes': f['bytes'],
                                 'zk': 'PASS' if f['zk_ok'] else 'FAIL', 'nonce_reused': f['nonce_reused'],
                                 'configured_link': f['linked'], 'zone_to': f['zone_to'],
                                 'target_sensitivity': f['target_sensitivity'], 'sensor_alarms': f['alarms']}
                                for f in (window[-self.recent_requests:] if self.recent_requests else [])],
            'sensor_reference': {name: {'true_positive_rate': tp, 'false_positive_rate': fp} for name, (tp, fp) in C.SENSORS.items()},
            'policy_guidance': {'allow_threshold': self.allow_threshold, 'block_threshold': self.block_threshold,
                                'meaning': 'trust >= allow_threshold usually ALLOW; between the two usually STEP-UP; below block_threshold usually BLOCK'},
        }

    def snapshot(self) -> Dict[str, dict]:
        return {d: {'trust': None if l.trust is None else round(l.trust, 4), 'band': l.band, 'status': l.classification,
                    'quarantined_until': l.quarantined_until, 'decision': l.decision, 'confidence': l.confidence,
                    'last_eval_tick': l.last_eval_tick, 'error': l.last_error} for d, l in self.ledgers.items()}

    def shutdown(self) -> None:
        if self.executor:
            self.executor.shutdown(wait=False, cancel_futures=True)
