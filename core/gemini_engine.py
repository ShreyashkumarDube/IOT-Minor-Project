"""Google Gemini as an LLM provider (official `google-genai` SDK, imported lazily).

GeminiEvaluator has the same interface as LLMEvaluator (`evaluate_raw`, `evaluate`, `warm_up`), so the
LLMController, the twin worlds and the comparison work unchanged. All isolation rules stay the same: the
controller builds the same facts-only context and validates the reply strictly; failures become LLM_ERROR.

Speed and quota:
* compact JSON context, `response_json_schema` + JSON MIME type, `max_output_tokens`, low thinking level;
  no temperature / top_p / top_k (deprecated from Gemini 3 on);
* a client-side rate limiter (requests per minute) — calls wait for a free slot;
* HTTP 429 / RESOURCE_EXHAUSTED: exponential backoff with jitter (1 s, 2 s, 4 s + 0–0.5 s), at most 3 retries;
  a per-day quota error is `quota_exhausted` and takes the LLM world offline.
The API key is read from GEMINI_API_KEY (or .env) or passed in; it is never logged, returned or put in a context.
"""
from __future__ import annotations

import json
import os
import random
import threading
import time
from collections import deque
from pathlib import Path
from types import SimpleNamespace
from typing import Callable, List, Optional

from .llm_engine import LLMUnavailable, _json_from_text

DEFAULT_MODEL = 'gemini-flash-lite-latest'   # an alias Google keeps pointing at the current Flash-Lite model
BACKOFF = (1.0, 2.0, 4.0)                    # seconds, plus 0–0.5 s jitter
MAX_RETRIES = 3


class GeminiRateLimited(LLMUnavailable):
    """429 per-minute limit still hit after the retries."""


class GeminiQuotaExhausted(LLMUnavailable):
    """429 with a per-day quota: the LLM world goes offline."""


# ----- key handling --------------------------------------------------------------------------------------
def load_dotenv(path: Optional[Path] = None) -> None:
    """Tiny .env loader (KEY=VALUE lines); never overrides variables that are already set."""
    path = Path(path) if path else Path(__file__).resolve().parents[1] / '.env'
    if not path.exists():
        return
    for line in path.read_text(encoding='utf-8').splitlines():
        line = line.strip()
        if not line or line.startswith('#') or '=' not in line:
            continue
        k, v = line.split('=', 1)
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def mask_key(key: Optional[str]) -> str:
    """'AIza…last4' — the only form in which a key may be shown."""
    if not key:
        return '(none)'
    return f'{key[:4]}…{key[-4:]}' if len(key) > 8 else '…'


def scrub(text: str, key: Optional[str]) -> str:
    """Remove the key (in case an SDK error message echoes it)."""
    return text.replace(key, mask_key(key)) if key and text else text


# ----- rate limiting -------------------------------------------------------------------------------------
class RateLimiter:
    """At most `rpm` calls in any 60-second window. Thread-safe; `clock`/`sleep` are injectable for tests."""

    def __init__(self, rpm: int, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep):
        self.rpm, self.clock, self.sleep = max(1, int(rpm)), clock, sleep
        self.calls: deque = deque()
        self.lock = threading.Lock()

    def _prune(self, now: float) -> None:
        while self.calls and now - self.calls[0] >= 60.0:
            self.calls.popleft()

    def acquire(self) -> float:
        """Wait for a free slot; returns the seconds waited."""
        waited = 0.0
        while True:
            with self.lock:
                now = self.clock()
                self._prune(now)
                if len(self.calls) < self.rpm:
                    self.calls.append(now)
                    return waited
                pause = 60.0 - (now - self.calls[0]) + 0.01
            self.sleep(pause)
            waited += pause

    def calls_last_minute(self) -> int:
        with self.lock:
            self._prune(self.clock())
            return len(self.calls)


# ----- the evaluator -------------------------------------------------------------------------------------
class GeminiEvaluator:
    """Same interface as LLMEvaluator. `client` can be injected (tests); otherwise google-genai is imported lazily."""

    def __init__(self, model: str = '', api_key: Optional[str] = None, system_prompt: str = '', response_schema: Optional[dict] = None,
                 max_output_tokens: Optional[int] = 160, timeout: float = 30, rpm: int = 10, thinking_level: Optional[str] = 'low',
                 client=None, clock: Callable[[], float] = time.monotonic, sleep: Callable[[float], None] = time.sleep,
                 rng: Optional[random.Random] = None):
        model = model or os.getenv('GEMINI_MODEL') or DEFAULT_MODEL
        self.config = SimpleNamespace(provider='gemini', model=model)   # what LLMController shows; never the key
        self._key = api_key or os.getenv('GEMINI_API_KEY', '')
        self.system_prompt, self.response_schema = system_prompt, response_schema
        self.max_output_tokens, self.timeout, self.thinking_level = max_output_tokens, timeout, thinking_level
        self._client = client
        self.limiter = RateLimiter(rpm, clock, sleep)
        self.sleep, self.rng = sleep, rng or random.Random()
        self.stats = {'calls': 0, 'retries': 0, 'rate_limited': 0, 'quota_exhausted': 0}
        self.warnings: List[str] = []
        self.quota_exhausted = False    # once set, no further API calls are made in this run
        self.lock = threading.Lock()

    def __repr__(self) -> str:   # never print the key
        return f'GeminiEvaluator(model={self.config.model!r}, key={mask_key(self._key)})'

    @property
    def key_hint(self) -> str:
        return mask_key(self._key)

    # ----- SDK plumbing ---------------------------------------------------------------------------------
    def client(self):
        if self._client is None:
            try:
                from google import genai
                from google.genai import types
            except ImportError as exc:
                raise LLMUnavailable('google-genai is not installed (pip install google-genai)') from exc
            if not self._key:
                raise LLMUnavailable('no Gemini API key: set GEMINI_API_KEY or enter it in the LLM settings')
            self._client = genai.Client(api_key=self._key, http_options=types.HttpOptions(timeout=int(self.timeout * 1000)))
        return self._client

    def _config(self, with_thinking: bool, max_tokens: Optional[int] = None) -> dict:
        cfg = {'system_instruction': self.system_prompt, 'response_mime_type': 'application/json',
               'max_output_tokens': max_tokens if max_tokens is not None else self.max_output_tokens}
        if self.response_schema:
            cfg['response_json_schema'] = self.response_schema
        if with_thinking and self.thinking_level:
            cfg['thinking_config'] = {'thinking_level': self.thinking_level}
        return {k: v for k, v in cfg.items() if v is not None}

    @staticmethod
    def _error_info(exc: BaseException):
        code = getattr(exc, 'code', None)
        status = str(getattr(exc, 'status', '') or '')
        text = f'{getattr(exc, "message", "") or ""} {exc}'
        return code, status, text

    def _generate(self, contents: str, max_tokens: Optional[int] = None):
        """One generate_content call with the rate limiter, the thinking fallback and the 429 backoff."""
        with_thinking = bool(self.thinking_level)
        dropped_thinking = False
        attempt = 0
        while True:
            if self.quota_exhausted:
                raise GeminiQuotaExhausted('Gemini daily quota exhausted')
            self.limiter.acquire()
            with self.lock:
                self.stats['calls'] += 1
            try:
                return self.client().models.generate_content(model=self.config.model, contents=contents,
                                                             config=self._config(with_thinking, max_tokens))
            except LLMUnavailable:
                raise
            except Exception as exc:  # noqa: BLE001 — classify every SDK / network error
                code, status, text = self._error_info(exc)
                low = text.lower()
                if code == 429 or 'resource_exhausted' in low or status == 'RESOURCE_EXHAUSTED':
                    if 'per day' in low or 'perday' in low or 'per_day' in low or 'daily' in low:
                        with self.lock:
                            self.stats['quota_exhausted'] += 1
                            self.quota_exhausted = True
                        raise GeminiQuotaExhausted('Gemini daily quota exhausted') from None
                    with self.lock:
                        self.stats['rate_limited'] += 1
                    if attempt >= MAX_RETRIES:
                        raise GeminiRateLimited(f'Gemini rate limit: still 429 after {MAX_RETRIES} retries') from None
                    with self.lock:
                        self.stats['retries'] += 1
                    self.sleep(BACKOFF[attempt] + self.rng.uniform(0, 0.5))
                    attempt += 1
                    continue
                if code == 400 and with_thinking and not dropped_thinking:
                    with_thinking, dropped_thinking = False, True
                    self.warnings.append('Gemini rejected a config field (HTTP 400); retried without thinking_config')
                    with self.lock:
                        self.stats['retries'] += 1
                    continue
                if 'timeout' in low or 'timed out' in low:
                    raise LLMUnavailable('LLM request failed: timed out') from None
                raise LLMUnavailable(scrub(f'Gemini request failed: HTTP {code} {text.strip()[:200]}', self._key)) from None

    # ----- interface used by LLMController and Classic mode ---------------------------------------------
    def evaluate_raw(self, context: dict) -> dict:
        t0 = time.perf_counter()
        resp = self._generate(json.dumps(context, separators=(',', ':')))
        latency_ms = (time.perf_counter() - t0) * 1000
        text = getattr(resp, 'text', None) or ''
        result = _json_from_text(text)   # ValueError -> invalid_json
        if not isinstance(result, dict):
            raise ValueError('LLM JSON is not an object')
        result['_latency_ms'] = latency_ms
        result['_raw_text'] = text
        usage = getattr(resp, 'usage_metadata', None)
        if usage is not None:
            result['_usage'] = {'prompt_tokens': getattr(usage, 'prompt_token_count', None),
                                'output_tokens': getattr(usage, 'candidates_token_count', None)}
        return result

    def evaluate(self, context: dict) -> dict:
        """Classic mode path: lenient validation like LLMEvaluator.evaluate."""
        from .llm_engine import LLMEvaluator
        return LLMEvaluator.validate(self.evaluate_raw(context))

    def warm_up(self) -> float:
        t0 = time.perf_counter()
        self._generate('{"ping":true}', max_tokens=16)
        return (time.perf_counter() - t0) * 1000

    def list_models(self) -> List[str]:
        """Models that support generateContent (names change often, so they are never hard-coded)."""
        try:
            models = list(self.client().models.list())
        except LLMUnavailable:
            raise
        except Exception as exc:  # noqa: BLE001
            raise LLMUnavailable(scrub(f'could not list Gemini models: {exc}', self._key)) from None
        out = []
        for m in models:
            actions = getattr(m, 'supported_actions', None) or []
            if 'generateContent' in actions:
                out.append(str(getattr(m, 'name', '')).replace('models/', ''))
        return sorted(out)

    def counters(self) -> dict:
        return {'calls this minute': self.limiter.calls_last_minute(), 'total calls': self.stats['calls'],
                'retries': self.stats['retries'], 'rate-limit errors': self.stats['rate_limited'],
                'quota errors': self.stats['quota_exhausted']}
