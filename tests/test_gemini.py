"""Google Gemini provider — no network and no real key: the SDK client is mocked."""
import builtins
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))

from core.gemini_engine import GeminiEvaluator, RateLimiter, load_dotenv, mask_key  # noqa: E402
from core.live_llm import LIVE_RESPONSE_SCHEMA, LIVE_SYSTEM_PROMPT, LLMController, make_live_evaluator  # noqa: E402
from core.metrics import log_rows, to_csv  # noqa: E402
from core.twin import TwinSimulation  # noqa: E402
from fake_llm import RuleBasedFakeLLM  # noqa: E402

KEY = 'AIzaSyTESTKEY-0123456789abcdefWXYZ'
FAKE = RuleBasedFakeLLM()


class APIError(Exception):
    """Shaped like google.genai.errors.APIError (code, status, message)."""

    def __init__(self, code, message, status=''):
        super().__init__(f'{code} {status}. {message}')
        self.code, self.status, self.message = code, status, message


class FakeModels:
    """Records every generate_content call; answers with the rule-based fake (or scripted errors)."""

    def __init__(self, script=None, reply=None):
        self.calls, self.script, self.reply = [], list(script or []), reply

    def generate_content(self, model, contents, config):
        self.calls.append({'model': model, 'contents': contents, 'config': config})
        if self.script:
            item = self.script.pop(0)
            if isinstance(item, BaseException):
                raise item
            text = item
        elif self.reply is not None:
            text = self.reply
        else:
            text = json.dumps(FAKE.evaluate(json.loads(contents)))
        return SimpleNamespace(text=text, usage_metadata=SimpleNamespace(prompt_token_count=321, candidates_token_count=42))

    def list(self):
        return [SimpleNamespace(name='models/gemini-flash-lite-x', supported_actions=['generateContent', 'countTokens']),
                SimpleNamespace(name='models/embedding-x', supported_actions=['embedContent'])]


def evaluator(models, **kw):
    kw.setdefault('sleep', lambda s: None)
    kw.setdefault('rpm', 100_000)   # the limiter has its own tests; avoid real waits here
    return GeminiEvaluator('gemini-test-model', KEY, LIVE_SYSTEM_PROMPT, LIVE_RESPONSE_SCHEMA, max_output_tokens=160,
                           client=SimpleNamespace(models=models), **kw)


CTX = {'device': {'id': 'DEV-001', 'type': 'Sensor'}, 'window_facts': {'requests': 3}, 'omitted_fields_mean_zero': True}


def test_request_shape_schema_json_tokens_no_temperature():
    models = FakeModels()
    out = evaluator(models).evaluate_raw(CTX)
    call = models.calls[0]
    cfg = call['config']
    assert call['model'] == 'gemini-test-model'
    assert call['contents'] == json.dumps(CTX, separators=(',', ':'))
    assert cfg['system_instruction'] == LIVE_SYSTEM_PROMPT and cfg['response_mime_type'] == 'application/json'
    assert cfg['response_json_schema'] == LIVE_RESPONSE_SCHEMA and cfg['max_output_tokens'] == 160
    assert cfg['thinking_config'] == {'thinking_level': 'low'}
    assert not {'temperature', 'top_p', 'top_k'} & set(cfg)
    assert out['_usage'] == {'prompt_tokens': 321, 'output_tokens': 42} and out['_latency_ms'] >= 0


def test_config_is_accepted_by_the_real_sdk_types():
    types = pytest.importorskip('google.genai.types')
    cfg = types.GenerateContentConfig(**evaluator(FakeModels())._config(True))
    assert cfg.thinking_config.thinking_level is not None and cfg.response_json_schema == LIVE_RESPONSE_SCHEMA


def test_valid_reply_is_parsed_by_the_controller():
    ctl = LLMController(evaluator(FakeModels(reply=json.dumps({'trust_score': 0.2, 'action': 'BLOCK', 'confidence': 0.8,
                                                                 'reason': 'replayed nonce', 'factors': ['replay']}))), synchronous=True)
    ctl.enroll('DEV-001', 1, 'z', 0.2)
    rec = ctl._apply('DEV-001', CTX, ctl.evaluator.evaluate_raw(CTX), 5.0)
    assert rec['ok'] and rec['llm_action'] == 'BLOCK' and rec['llm_trust'] == 0.2
    assert rec['llm_model'] == 'gemini-test-model (gemini)' and rec['llm_usage']['output_tokens'] == 42


def test_invalid_json_is_llm_error():
    ev = evaluator(FakeModels(reply='not json at all'))
    ctl = LLMController(ev, synchronous=True)
    ctl.enroll('DEV-001', 1, 'z', 0.2)
    with pytest.raises(ValueError) as exc:
        ev.evaluate_raw(CTX)
    rec = ctl._apply('DEV-001', CTX, exc.value, 5.0)
    assert not rec['ok'] and rec['error'] == 'invalid_json' and rec['llm_trust'] is None


def test_400_on_thinking_config_retries_once_without_it():
    good = json.dumps({'trust_score': 0.9, 'action': 'ALLOW', 'confidence': 0.7, 'reason': 'ok', 'factors': []})
    models = FakeModels(script=[APIError(400, 'Invalid JSON payload: unknown field thinking_level', 'INVALID_ARGUMENT'), good])
    ev = evaluator(models)
    assert ev.evaluate_raw(CTX)['action'] == 'ALLOW'
    assert 'thinking_config' in models.calls[0]['config'] and 'thinking_config' not in models.calls[1]['config']
    assert len(models.calls) == 2 and ev.warnings and ev.stats['retries'] == 1


def test_400_twice_is_an_error_not_a_crash():
    models = FakeModels(script=[APIError(400, 'bad', 'INVALID_ARGUMENT'), APIError(400, 'bad', 'INVALID_ARGUMENT')])
    from core.llm_engine import LLMUnavailable
    with pytest.raises(LLMUnavailable):
        evaluator(models).evaluate_raw(CTX)
    assert len(models.calls) == 2


def test_two_429_then_success_backs_off_and_counts_retries():
    slept = []
    good = json.dumps({'trust_score': 0.9, 'action': 'ALLOW', 'confidence': 0.7, 'reason': 'ok', 'factors': []})
    rate = 'Quota exceeded for metric: generate_content_free_tier_requests, quotaId: GenerateRequestsPerMinutePerProjectPerModel-FreeTier'
    models = FakeModels(script=[APIError(429, rate, 'RESOURCE_EXHAUSTED'), APIError(429, rate, 'RESOURCE_EXHAUSTED'), good])
    ev = evaluator(models, sleep=slept.append)
    assert ev.evaluate_raw(CTX)['action'] == 'ALLOW'
    assert ev.stats['retries'] == 2 and ev.stats['rate_limited'] == 2 and ev.stats['calls'] == 3
    assert 1.0 <= slept[0] <= 1.5 and 2.0 <= slept[1] <= 2.5


def test_429_after_three_retries_is_rate_limited():
    rate = 'rate limit per minute'
    models = FakeModels(script=[APIError(429, rate, 'RESOURCE_EXHAUSTED')] * 4)
    ev = evaluator(models)
    ctl = LLMController(ev, synchronous=True)
    ctl.enroll('DEV-001', 1, 'z', 0.2)
    try:
        raw = ev.evaluate_raw(CTX)
    except Exception as exc:  # noqa: BLE001
        raw = exc
    rec = ctl._apply('DEV-001', CTX, raw, 1.0)
    assert rec['error'] == 'rate_limited' and ev.stats['retries'] == 3 and len(models.calls) == 4 and not ctl.quota_exhausted


def test_daily_quota_takes_the_llm_world_offline_and_the_world_keeps_running():
    daily = APIError(429, 'Quota exceeded for metric: generate_content_free_tier_requests, limit: 250, '
                          'quotaId: GenerateRequestsPerDayPerProjectPerModel-FreeTier', 'RESOURCE_EXHAUSTED')

    class Models(FakeModels):
        def generate_content(self, model, contents, config):
            if len(self.calls) >= 5:
                self.calls.append(None)
                raise daily
            return super().generate_content(model, contents, config)

    models = Models()
    ev = evaluator(models)
    twin = TwinSimulation.from_scenario('compromised_valid_token', seed=42, mode='both',
                                        llm_controller_factory=lambda: LLMController(ev, synchronous=True))
    twin.run(80)
    ctl = twin.llm.controller
    assert twin.tick == 80 and ctl.quota_exhausted and ctl.offline and ctl.offline_message == 'Gemini daily quota exhausted'
    assert ctl.errors['quota_exhausted'] >= 1
    assert len(models.calls) == 6    # no further API calls after the daily quota error
    assert twin.bayes.tick == twin.llm.tick == 80
    twin.shutdown()


def test_rate_limiter_never_exceeds_rpm_with_a_fake_clock():
    now = [0.0]
    sleeps = []

    def sleep(s):
        sleeps.append(s)
        now[0] += s

    rl = RateLimiter(3, clock=lambda: now[0], sleep=sleep)
    stamps = []
    for _ in range(10):
        rl.acquire()
        stamps.append(now[0])
        now[0] += 0.5
    for t in stamps:
        assert sum(1 for x in stamps if t <= x < t + 60) <= 3
    assert sleeps


def test_evaluator_rate_limiter_paces_calls():
    now = [0.0]
    ev = evaluator(FakeModels(), rpm=2, clock=lambda: now[0], sleep=lambda s: now.__setitem__(0, now[0] + s))
    for _ in range(5):
        ev.evaluate_raw(CTX)
    assert now[0] >= 120 and ev.counters()['total calls'] == 5


def test_key_never_appears_in_records_terminal_exports_or_context(capsys):
    models = FakeModels()
    ev = make_live_evaluator('gemini', 'gemini-test-model', 'Google Gemini API', KEY)
    ev._client = SimpleNamespace(models=models)
    ev.sleep = lambda s: None
    ev.limiter.rpm = 100_000
    twin = TwinSimulation.from_scenario('compromised_valid_token', seed=42, mode='both',
                                        llm_controller_factory=lambda: LLMController(ev, synchronous=True))
    twin.run(60)
    from core.comparison import compare, report_markdown, summary_table
    cmp = compare(twin)
    blobs = [json.dumps(twin.llm.records, default=str), to_csv(log_rows(twin.llm)), report_markdown(cmp),
             json.dumps(summary_table(cmp), default=str), repr(ev), str(ev.counters()), capsys.readouterr().out]
    blobs += [c['contents'] for c in models.calls] + [json.dumps(c['config'], default=str) for c in models.calls]
    assert models.calls
    for b in blobs:
        assert KEY not in b and KEY[-12:] not in b
    assert 'gemini-test-model (gemini)' in report_markdown(cmp)
    assert mask_key(KEY) == 'AIza…WXYZ'
    twin.shutdown()


def test_error_messages_scrub_the_key():
    from core.llm_engine import LLMUnavailable
    models = FakeModels(script=[APIError(500, f'server echoed key={KEY}', 'INTERNAL')])
    with pytest.raises(LLMUnavailable) as exc:
        evaluator(models).evaluate_raw(CTX)
    assert KEY not in str(exc.value) and 'AIza…WXYZ' in str(exc.value)


def test_list_models_filters_generate_content():
    assert evaluator(FakeModels()).list_models() == ['gemini-flash-lite-x']


def test_isolation_and_no_fallback_with_gemini():
    """The Gemini world sees only its facts-only context; an error never falls back to Bayesian values."""
    models = FakeModels(reply='{"broken": ')
    ev = evaluator(models)
    twin = TwinSimulation.from_scenario('compromised_valid_token', seed=42, mode='both',
                                        llm_controller_factory=lambda: LLMController(ev, synchronous=True))
    twin.run(50)
    evals = [r for r in twin.llm.records if r.get('kind') == 'llm_eval']
    assert evals and all(not r['ok'] and r['llm_trust'] is None for r in evals)
    banned = ('trust_score', 'log_odds', 'bayes', 'ground_truth', 'is_attacker', 'attacker')
    for c in models.calls:
        low = c['contents'].lower()
        assert not any(b in low for b in banned), low[:200]
    reqs = [r for r in twin.llm.records if r.get('kind') == 'request']
    assert all(r.get('trust_after') is None or r['trust_after'] != r.get('bayes_trust') for r in reqs)
    twin.shutdown()


def test_other_providers_and_the_app_work_without_the_sdk(monkeypatch):
    real_import = builtins.__import__

    def no_genai(name, *a, **kw):
        if name.startswith('google'):
            raise ImportError('no google-genai')
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, '__import__', no_genai)
    for m in [m for m in sys.modules if m.startswith('google')]:
        monkeypatch.delitem(sys.modules, m)
    ollama = make_live_evaluator('ollama', 'llama3.2:3b', 'http://localhost:11434/api/chat')
    assert ollama.config.model == 'llama3.2:3b'
    ev = make_live_evaluator('gemini', 'gemini-test-model', '', KEY)
    from core.llm_engine import LLMUnavailable
    with pytest.raises(LLMUnavailable, match='google-genai is not installed'):
        ev.evaluate_raw(CTX)


def test_dotenv_loader(tmp_path, monkeypatch):
    monkeypatch.delenv('GEMINI_MODEL', raising=False)
    monkeypatch.setenv('GEMINI_API_KEY', 'already-set')
    (tmp_path / '.env').write_text('# comment\nGEMINI_API_KEY=from-file\nGEMINI_MODEL="gemini-x"\n')
    load_dotenv(tmp_path / '.env')
    assert os.environ['GEMINI_API_KEY'] == 'already-set' and os.environ['GEMINI_MODEL'] == 'gemini-x'


def test_run_batch_gemini_args(monkeypatch):
    monkeypatch.setenv('GEMINI_API_KEY', KEY)
    import run_batch
    from core.live_llm import resolve_llm_args
    import argparse
    args = resolve_llm_args(argparse.Namespace(provider='gemini', model='gemini-test-model', endpoint=None))
    assert args.llm_note == 'gemini-test-model (gemini) @ Google Gemini API' and KEY not in args.llm_note
    assert callable(run_batch.llm_factory)


def test_app_offers_gemini_and_keeps_ollama_default():
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=90).run()
    assert not at.exception
    at.radio(key='live_run_mode').set_value('both').run()
    sel = at.selectbox(key='live_llm_provider')
    assert sel.options == ['Ollama (local)', 'Google Gemini (API)', 'OpenAI-compatible API'] and sel.value == 'Ollama (local)'
    sel.set_value('Google Gemini (API)').run()
    assert not at.exception
    assert any("Simulated device data is sent to Google's API" in c.value for c in at.caption)
    assert at.text_input(key='live_gemini_key').value == ''
