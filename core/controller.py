"""The Zero-Trust controller: live trust ledger, Bayesian updates and policy bands.

What the controller knows:
  * public enrollment data per device: id, public key, zone, target sensitivity
  * the configured topology (which directed links are authorised)
  * a stream of ObservedEvent objects and per-request sensor alarms (True/False)

What it does NOT know: who is an attacker. It infers suspicion only from the
observed stream, and it detects nonce replay, rate spikes, scans, unusual
payloads etc. itself from what it has seen.

Trust math (log-odds form of Bayes' rule), per device:
    L = logit(trust)
    once per tick:   L = λ·L + (1-λ)·logit(prior)           decay / recovery toward the prior
    per request:     L = L + Σ log(P(e|legit) / P(e|malicious))  for each observed event e
                     L = L + Σ log(P(alarm_obs|legit) / P(alarm_obs|malicious))  for each sensor
    trust = clamp(sigmoid(L), TRUST_MIN, TRUST_MAX)
Every update keeps a list of the terms it added so the UI can show the calculation.
"""
from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Dict, Iterable, List, Optional, Set, Tuple

from . import live_config as C
from .events import CONTROLLER_ID, EventType, ObservedEvent, is_night, likelihood
from .zk_auth import verify

TRUSTED, SUSPICIOUS, ISOLATED = 'Trusted', 'Suspicious', 'Isolated'
QUARANTINED, PROBATION = 'Quarantined', 'Probation'  # display classifications on top of the bands
BAND_ORDER = [ISOLATED, SUSPICIOUS, TRUSTED]  # worst -> best
BAND_ACTION = {TRUSTED: 'ALLOW', SUSPICIOUS: 'STEP-UP', ISOLATED: 'BLOCK'}


def logit(p: float) -> float:
    """Probability -> log-odds."""
    return math.log(p / (1 - p))


def sigmoid(x: float) -> float:
    """Log-odds -> probability."""
    return 1 / (1 + math.exp(-x))


def clamp_trust(p: float) -> float:
    """Keep trust away from 0 and 1 so it can always move again."""
    return max(C.TRUST_MIN, min(C.TRUST_MAX, p))


def raw_band(trust: float, allow: float, block: float) -> str:
    """Band from thresholds only (no hysteresis)."""
    if trust >= allow:
        return TRUSTED
    if trust >= block:
        return SUSPICIOUS
    return ISOLATED


def classify(trust: float, previous: str, allow: float, block: float, margin: float) -> str:
    """Band with hysteresis: a device only changes band once it is `margin` past the threshold."""
    new = raw_band(trust, allow, block)
    if new == previous:
        return new
    getting_worse = BAND_ORDER.index(new) < BAND_ORDER.index(previous)
    nudged = trust + margin if getting_worse else trust - margin
    if raw_band(nudged, allow, block) == previous:
        return previous  # not far enough past the threshold yet
    return new


@dataclass
class DeviceLedger:
    """Everything the controller remembers about one device (all derived from observations)."""
    device_id: str
    public_key: int
    zone: str
    sensitivity: float
    trust: float = C.LIVE_PRIOR
    band: str = TRUSTED
    enrolled_tick: int = 0
    seen_nonces: Set[str] = field(default_factory=set)
    window: Deque[Tuple[int, str, str, int]] = field(default_factory=deque)  # (tick, kind, dst, port)
    warmup_requests: int = 0
    warmup_bytes: Dict[str, List[int]] = field(default_factory=dict)
    baseline_rate: Optional[float] = None           # requests per tick, learned during warm-up
    baseline_bytes: Dict[str, float] = field(default_factory=dict)
    pending_decay: float = 0.0                      # log-odds moved by decay since the last logged update
    quarantined_until: Optional[int] = None         # links cut until this tick
    probation: bool = False                         # released from quarantine, must pass step-up first

    @property
    def classification(self) -> str:
        """What the status panel shows: Quarantined / Probation / Trusted / Suspicious / Isolated."""
        if self.quarantined_until is not None:
            return QUARANTINED
        if self.probation:
            return PROBATION
        return self.band


class ZeroTrustController:
    """Bayesian live controller. Public methods accept observed data only."""
    engine = 'bayesian'

    def __init__(self, allow_threshold: float = C.ALLOW_THRESHOLD, block_threshold: float = C.BLOCK_THRESHOLD,
                 decay_lambda: float = C.DECAY_LAMBDA, prior: float = C.LIVE_PRIOR):
        self.allow_threshold = allow_threshold
        self.block_threshold = block_threshold
        self.decay_lambda = decay_lambda
        self.prior = prior
        self.ledgers: Dict[str, DeviceLedger] = {}
        self.edges: Set[Tuple[str, str]] = set()
        self.tick = 0

    # ----- setup ------------------------------------------------------------
    def enroll(self, device_id: str, public_key: int, zone: str, target_sensitivity: float) -> None:
        """Register a device's PUBLIC data. No name, role or label is accepted."""
        self.ledgers[device_id] = DeviceLedger(device_id, public_key, zone, float(target_sensitivity),
                                               trust=self.prior, enrolled_tick=self.tick)

    def set_topology(self, connections: Iterable[Tuple[str, str]]) -> None:
        """The configured directed links are the authorised paths."""
        self.edges = {tuple(e) for e in connections}

    # ----- per tick: decay / recovery -------------------------------------------
    def begin_tick(self, tick: int) -> List[str]:
        """Pull every device's log-odds slightly back toward the prior. Returns band-change notes."""
        self.tick = tick
        notes = []
        for led in self.ledgers.values():
            if led.quarantined_until is not None:
                if tick < led.quarantined_until:
                    continue  # links are cut: no traffic, trust frozen
                led.quarantined_until, led.probation = None, True
                led.trust, led.band, led.pending_decay = C.PROBATION_TRUST, SUSPICIOUS, 0.0
                notes.append(f'{led.device_id} released from quarantine → PROBATION (trust reset to {C.PROBATION_TRUST:.2f}; must pass step-up)')
                continue
            before = logit(led.trust)
            after = self.decay_lambda * before + (1 - self.decay_lambda) * logit(self.prior)
            led.trust = clamp_trust(sigmoid(after))
            led.pending_decay += logit(led.trust) - before
            new_band = classify(led.trust, led.band, self.allow_threshold, self.block_threshold, C.HYSTERESIS)
            if new_band != led.band:
                notes.append(f'{led.device_id} recovered: {led.band} → {new_band} (trust {led.trust:.2f})')
                led.band = new_band
            self._learn_baseline(led, tick)
        return notes

    def _learn_baseline(self, led: DeviceLedger, tick: int) -> None:
        """Freeze the device's own rate/payload baseline once its warm-up period is over."""
        if led.baseline_rate is None and tick - led.enrolled_tick > C.WARMUP_TICKS:
            led.baseline_rate = led.warmup_requests / C.WARMUP_TICKS
            led.baseline_bytes = {k: sum(v) / len(v) for k, v in led.warmup_bytes.items()}

    # ----- per request ------------------------------------------------------------
    def process(self, event: ObservedEvent, alarms: Dict[str, bool]) -> dict:
        """Update trust from one observed request and decide ALLOW / STEP-UP / BLOCK.

        Returns a plain dict record (used for the live terminal, exports and tests).
        """
        if not isinstance(event, ObservedEvent):
            raise TypeError('process() only accepts an ObservedEvent')
        if set(alarms) - set(C.SENSORS) or not all(isinstance(v, bool) for v in alarms.values()):
            raise TypeError('alarms must be {sensor name: bool} for the configured sensors only')
        led = self.ledgers[event.src]
        trust_before, band_before = led.trust, led.band
        kind = EventType(event.event_type)
        port, size = event.details.get('port', 0), event.details.get('bytes', 0)

        # 0) Quarantine: links of an isolated device are cut, so the request is dropped (no trust update).
        blocked_end = self.quarantine_end(event.src) or self.quarantine_end(event.dst)
        if blocked_end is not None:
            who = event.src if self.quarantine_end(event.src) is not None else event.dst
            return {'tick': event.tick, 'src': event.src, 'dst': event.dst, 'event_type': kind.value,
                    'action': 'DROPPED', 'reason': f'{who} is quarantined until t={blocked_end}', 'kind_hint': 'dropped',
                    'trust_before': round(trust_before, 4), 'trust_after': round(trust_before, 4),
                    'band_before': band_before, 'band_after': led.band, 'derived': [], 'terms': [],
                    'alarms': {}, 'zk': '—', 'replay': False, 'token_count': 0, 'port': port, 'bytes': size,
                    'nonce': event.nonce, 'proof': None, 'challenge': None, 'step_up': None, 'quarantined_until': None}

        # 1) Zero-knowledge check and nonce replay, done by the controller itself.
        zk_ok = bool(event.proof) and verify(led.public_key, event.proof, event.src)
        replay = event.nonce is not None and event.nonce in led.seen_nonces
        if event.nonce is not None:
            led.seen_nonces.add(event.nonce)

        # 2) Sliding window of this device's own recent requests.
        led.window.append((event.tick, kind.value, event.dst, port))
        while led.window and led.window[0][0] <= event.tick - C.WINDOW_TICKS:
            led.window.popleft()
        if led.baseline_rate is None:
            led.warmup_requests += 1
            led.warmup_bytes.setdefault(kind.value, []).append(size)

        # 3) Evidence terms, each one labelled for the explanation.
        terms: List[Tuple[str, float]] = []
        if led.pending_decay:
            terms.append(('decay', led.pending_decay))
            led.pending_decay = 0.0
        token_count = sum(1 for w in led.window if w[1] == EventType.TOKEN_REQUEST.value)
        if kind == EventType.TOKEN_REQUEST:
            if token_count > C.TOKEN_FREE_PER_WINDOW:
                terms.append((f'token_request×{token_count}', self._event_term(kind)))
        else:
            terms.append((kind.value, self._event_term(kind)))
        derived = self._derive_anomalies(led, event, kind, zk_ok, replay)
        for anomaly in derived:
            terms.append((anomaly.value, self._event_term(anomaly)))
        for sensor, fired in alarms.items():
            terms.append((f'{sensor}{"!" if fired else "·"}', self._sensor_term(sensor, fired)))

        # 4) Bayes update in log-odds, then clamp.
        log_odds = logit(led.trust) + sum(delta for _, delta in terms if _ != 'decay')
        led.trust = clamp_trust(sigmoid(log_odds))
        led.band = classify(led.trust, led.band, self.allow_threshold, self.block_threshold, C.HYSTERESIS)
        # Display-only bookkeeping: the same numbers unrounded, so the UI can show a hand-checkable calculation.
        # Nothing below feeds back into a decision.
        calc = {'trust_before': trust_before, 'L_start': logit(trust_before),
                'terms': [(label, delta) for label, delta in terms], 'L_end': log_odds, 'sigmoid': sigmoid(log_odds),
                'trust_after': led.trust, 'raw_band': raw_band(led.trust, self.allow_threshold, self.block_threshold),
                'band_before': band_before, 'band_after': led.band, 'probation': led.probation,
                'thresholds': {'allow': self.allow_threshold, 'block': self.block_threshold, 'hysteresis': C.HYSTERESIS},
                'evidence': self._evidence_facts(led, event, kind, token_count)}

        # 5) Policy: hard gates first, then the band (probation always needs a step-up).
        if not zk_ok:
            action, reason = 'BLOCK', 'ZK proof rejected (hard gate)'
        elif replay:
            action, reason = 'BLOCK', 'nonce replayed (hard gate)'
        elif led.probation and led.band != ISOLATED:
            action, reason = 'STEP-UP', 'on probation after quarantine'
        else:
            action, reason = BAND_ACTION[led.band], f'band {led.band}'
        quarantined_until = self._maybe_quarantine(led)
        public_proof = {k: event.proof[k] for k in ('commitment', 'challenge', 'response')} if event.proof else None
        return {
            'nonce': event.nonce, 'proof': public_proof, 'quarantined_until': quarantined_until,
            'tick': event.tick, 'src': event.src, 'dst': event.dst, 'event_type': kind.value,
            'port': port, 'bytes': size, 'token_count': token_count, 'zk': 'PASS' if zk_ok else 'FAIL',
            'replay': replay, 'alarms': dict(alarms), 'derived': [a.value for a in derived],
            'terms': [(label, round(delta, 4)) for label, delta in terms],
            'trust_before': round(trust_before, 4), 'trust_after': round(led.trust, 4),
            'band_before': band_before, 'band_after': led.band, 'action': action, 'reason': reason,
            'challenge': f'{event.src}-challenge-{event.tick}-{event.nonce}' if action == 'STEP-UP' else None,
            'step_up': None, 'calc': calc,
        }

    def complete_step_up(self, record: dict, proof: Optional[dict], mfa_passed: bool) -> dict:
        """Finish a STEP-UP: verify a fresh proof on the challenge nonce plus the (observed) MFA outcome."""
        if record.get('action') != 'STEP-UP' or not record.get('challenge'):
            raise ValueError('record is not awaiting a step-up challenge')
        led = self.ledgers[record['src']]
        record['band_before_step_up'] = record.get('band_after', led.band)
        zk_ok = bool(proof) and proof.get('nonce') == record['challenge'] and verify(led.public_key, proof, led.device_id)
        passed = zk_ok and bool(mfa_passed)
        outcome = EventType.STEP_UP_PASS if passed else EventType.STEP_UP_FAIL
        delta = self._event_term(outcome)
        before = led.trust
        band_before_step = led.band
        led.trust = clamp_trust(sigmoid(logit(led.trust) + delta))
        led.band = classify(led.trust, led.band, self.allow_threshold, self.block_threshold, C.HYSTERESIS)
        if 'calc' in record:   # display-only, unrounded
            record['calc']['step_up'] = {'term': (outcome.value, delta), 'trust_before': before, 'L_start': logit(before),
                                         'L_end': logit(before) + delta, 'sigmoid': sigmoid(logit(before) + delta),
                                         'trust_after': led.trust, 'band_before': band_before_step,
                                         'raw_band': raw_band(led.trust, self.allow_threshold, self.block_threshold),
                                         'band_after': led.band}
        record['step_up'] = {'zk': 'PASS' if zk_ok else 'FAIL', 'mfa': 'PASS' if mfa_passed else 'FAIL',
                             'term': (outcome.value, round(delta, 4)),
                             'trust_before': round(before, 4), 'trust_after': round(led.trust, 4)}
        record['trust_after'], record['band_after'] = round(led.trust, 4), led.band
        record['action'] = 'ALLOW' if passed else 'BLOCK'
        record['reason'] = 'step-up passed' if passed else 'step-up failed'
        if passed:
            led.probation = False  # a successful step-up ends probation
        record['quarantined_until'] = self._maybe_quarantine(led)
        return record

    def _maybe_quarantine(self, led: DeviceLedger) -> Optional[int]:
        """Isolated band -> cut the device's links for QUARANTINE_TICKS. Returns the end tick if it just started."""
        if led.band == ISOLATED and led.quarantined_until is None:
            led.quarantined_until = self.tick + C.QUARANTINE_TICKS
            led.probation = False
            return led.quarantined_until
        return None

    def quarantine_end(self, device_id: str) -> Optional[int]:
        """Tick at which this device's quarantine ends, or None if it is not quarantined."""
        led = self.ledgers.get(device_id)
        return led.quarantined_until if led else None

    # ----- helpers ------------------------------------------------------------
    def _derive_anomalies(self, led: DeviceLedger, event: ObservedEvent, kind: EventType,
                          zk_ok: bool, replay: bool) -> List[EventType]:
        """Turn observable facts into anomaly events. Nothing here looks at ground truth."""
        found: List[EventType] = []
        if not zk_ok:
            found.append(EventType.ZK_AUTH_FAIL)
        if replay:
            found.append(EventType.TOKEN_REPLAY)
        if led.baseline_rate is not None:
            expected = led.baseline_rate * C.WINDOW_TICKS
            if len(led.window) > max(C.RATE_SPIKE_FACTOR * expected, expected + C.RATE_MIN_EXCESS):
                found.append(EventType.RATE_SPIKE)
            usual = led.baseline_bytes.get(kind.value)
            if usual and event.details.get('bytes', 0) > C.PAYLOAD_FACTOR * usual:
                found.append(EventType.UNUSUAL_PAYLOAD)
        if len({w[2] for w in led.window if w[2] != CONTROLLER_ID}) > C.SCAN_DISTINCT_DSTS:
            found.append(EventType.DEVICE_SCAN)
        if len({w[3] for w in led.window}) > C.SCAN_DISTINCT_PORTS:
            found.append(EventType.PORT_SCAN)
        if event.dst != CONTROLLER_ID:
            linked = (event.src, event.dst) in self.edges
            target = self.ledgers.get(event.dst)
            if not linked:
                found.append(EventType.NO_TOPOLOGY_EDGE)
            if target is not None:
                usual_zones = {led.zone} | {self.ledgers[d].zone for s, d in self.edges if s == led.device_id and d in self.ledgers}
                if target.zone not in usual_zones:
                    found.append(EventType.CROSS_ZONE_ACCESS)
                if target.sensitivity >= C.SENSITIVE_TARGET and not linked:
                    found.append(EventType.SENSITIVE_TARGET_ACCESS)
        if is_night(event.tick) and kind in (EventType.AUTHORIZED_SERVICE, EventType.TOKEN_REQUEST):
            found.append(EventType.OFF_HOURS_ACCESS)
        return found

    def _evidence_facts(self, led: DeviceLedger, event: ObservedEvent, kind: EventType, token_count: int) -> dict:
        """The raw observed values behind each anomaly check (display only; read-only, mirrors _derive_anomalies)."""
        expected = None if led.baseline_rate is None else led.baseline_rate * C.WINDOW_TICKS
        linked = (event.src, event.dst) in self.edges
        target = self.ledgers.get(event.dst)
        usual_zones = sorted({led.zone} | {self.ledgers[d].zone for s, d in self.edges if s == led.device_id and d in self.ledgers})
        return {
            'window_requests': len(led.window),
            'baseline_rate': led.baseline_rate,
            'rate_threshold': None if expected is None else max(C.RATE_SPIKE_FACTOR * expected, expected + C.RATE_MIN_EXCESS),
            'distinct_destinations': len({w[2] for w in led.window if w[2] != CONTROLLER_ID}),
            'distinct_ports': len({w[3] for w in led.window}),
            'bytes': event.details.get('bytes', 0),
            'baseline_bytes': led.baseline_bytes.get(kind.value),
            'token_count': token_count,
            'to_controller': event.dst == CONTROLLER_ID,
            'linked': linked,
            'own_zone': led.zone, 'usual_zones': usual_zones,
            'target_zone': target.zone if target is not None else None,
            'target_sensitivity': target.sensitivity if target is not None else None,
            'night': is_night(event.tick),
        }

    @staticmethod
    def _event_term(event_type: EventType) -> float:
        """Weighted log-likelihood ratio of one observed event."""
        p_legit, p_malicious = likelihood(event_type)
        return C.EVENT_WEIGHT * math.log(p_legit / p_malicious)

    @staticmethod
    def _sensor_term(sensor: str, fired: bool) -> float:
        """Weighted log-likelihood ratio of one sensor's alarm / no-alarm."""
        true_positive, false_positive = C.SENSORS[sensor]
        p_legit = false_positive if fired else 1 - false_positive
        p_malicious = true_positive if fired else 1 - true_positive
        return C.SENSOR_WEIGHT * math.log(p_legit / p_malicious)

    def snapshot(self) -> Dict[str, dict]:
        """Current trust and band per device (what the status panel shows)."""
        return {d: {'trust': round(l.trust, 4), 'band': l.band, 'status': l.classification,
                    'quarantined_until': l.quarantined_until} for d, l in self.ledgers.items()}
