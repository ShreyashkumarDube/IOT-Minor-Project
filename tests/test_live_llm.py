"""Step 2: the LLM world is fully isolated from the Bayesian world (tests 1-7 of the isolation prompt)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import pytest

from core.controller import ZeroTrustController
from core.live_llm import LIVE_SYSTEM_PROMPT, LLMController, parse_llm_output, InvalidLLMOutput
from core.live_sim import render_record
from core.llm_engine import LLMConfig, LLMEvaluator, LLMUnavailable
from core.twin import TwinSimulation

FORBIDDEN_SUBSTRINGS = ('bayes', 'band', 'role', 'is_attacker', 'ground_truth', 'attack_type', 'suspicious', 'malicious')


class FakeLLM:
    """Configurable stand-in for a real model; records every context it receives."""

    def __init__(self, trust=0.37, action='STEP-UP', delay=0.0, raise_exc=None, rule=None, reply=None):
        self.trust, self.action, self.delay, self.raise_exc, self.rule, self.reply = trust, action, delay, raise_exc, rule, reply
        self.contexts = []
        self.lock = threading.Lock()

    def evaluate(self, context):
        with self.lock:
            self.contexts.append(json.loads(json.dumps(context)))
        if self.delay:
            time.sleep(self.delay)
        if self.raise_exc:
            raise self.raise_exc
        if self.reply is not None:
            return self.reply
        trust, action = self.rule(context) if self.rule else (self.trust, self.action)
        return {'trust_score': trust, 'action': action, 'risk_score': 1 - trust, 'confidence': 0.66,
                'reason': f'fake {action}', 'factors': ['token_requests', 'zk_failures'], 'behavior_assessment': []}


def twin(mode, fake=None, scenario='multi_attacker', seed=3, synchronous=True, **kw):
    factory = (lambda: LLMController(fake, synchronous=synchronous, **kw)) if fake is not None else None
    return TwinSimulation.from_scenario(scenario, seed=seed, mode=mode, llm_controller_factory=factory)


def walk(obj, path=''):
    """Yield (path, key-or-None, value) for every nested key and leaf value."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            yield path, k, None
            yield from walk(v, f'{path}.{k}')
    elif isinstance(obj, list):
        for v in obj:
            yield from walk(v, path + '[]')
    else:
        yield path, None, obj


# 1 ---------------------------------------------------------------------------------------------------------
def test_isolation_by_sabotage(monkeypatch):
    def boom(*a, **k):
        raise RuntimeError('Bayesian controller sabotaged')
    monkeypatch.setattr(ZeroTrustController, 'process', boom)
    t = twin('llm', FakeLLM(trust=0.8, action='ALLOW'))
    t.run(40)
    assert t.bayes is None and any(r.get('kind') == 'llm_eval' and r['ok'] for r in t.llm.records)
    monkeypatch.undo()

    plain = twin('bayesian'); plain.run(60)
    broken = FakeLLM(raise_exc=RuntimeError('LLM sabotaged'))
    monkeypatch.setattr(LLMController, '__init__', lambda *a, **k: (_ for _ in ()).throw(RuntimeError('never built')))
    again = TwinSimulation.from_scenario('multi_attacker', seed=3, mode='bayesian',
                                         llm_controller_factory=lambda: LLMController(broken))
    again.run(60)
    assert plain.bayes.records == again.bayes.records


# 2 ---------------------------------------------------------------------------------------------------------
def test_no_copying_llm_values_are_its_own():
    t = twin('both', FakeLLM(trust=0.37, action='STEP-UP'))
    t.run(80)
    evals = [r for r in t.llm.records if r.get('kind') == 'llm_eval']
    assert evals and all(r['ok'] and r['llm_trust'] == 0.37 and r['llm_action'] == 'STEP-UP' for r in evals)
    for device_id, s in t.llm.controller.snapshot().items():
        assert s['trust'] in (None, 0.37) and s['decision'] in ('PENDING', 'STEP-UP')
    bayes_trusts = {s['trust'] for s in t.bayes.controller.snapshot().values()}
    assert 0.37 not in bayes_trusts
    assert all('0.37' in render_record(r)[0] and 'STEP-UP' in render_record(r)[0] for r in evals)
    assert all(r.get('engine') == 'llm' for r in t.llm.records) and all(r.get('engine') == 'bayesian' for r in t.bayes.records)


# 3 ---------------------------------------------------------------------------------------------------------
@pytest.mark.parametrize('scenario', ['multi_attacker', 'compromised_valid_token', 'replay_attack'])
def test_context_purity(scenario):
    fake = FakeLLM(rule=lambda c: (0.2, 'BLOCK') if c['window_facts'].get('zk_failures', 0) > 2 else (0.8, 'ALLOW'))
    t = twin('both', fake, scenario=scenario, seed=1)
    t.run(160)
    assert fake.contexts
    for ctx in fake.contexts:
        for path, key, value in walk(ctx):
            if key is not None:
                assert not any(bad in key.lower() for bad in FORBIDDEN_SUBSTRINGS), (path, key)
            if isinstance(value, str):
                assert not any(bad in value.lower() for bad in ('attacker', 'compromised', 'scanning', 'flood', 'slow_low', 'honest')), (path, value)
        assert 'name' not in ctx['device'] and 'trust' not in ctx['device']


# 4 ---------------------------------------------------------------------------------------------------------
def test_no_shared_state_between_worlds():
    t = twin('both', FakeLLM(trust=0.9, action='ALLOW'))
    t.run(30)
    b, l = t.bayes, t.llm
    assert b is not l and b.controller is not l.controller and type(b.controller) is not type(l.controller)
    assert b.records is not l.records and b.devices is not l.devices and b.agents is not l.agents
    for d in b.devices:
        assert b.devices[d] is not l.devices[d]
        lb, ll = b.controller.ledgers[d], l.controller.ledgers[d]
        assert lb is not ll and lb.seen_nonces is not ll.seen_nonces and lb.window is not ll.window
    l.controller.ledgers['DEV-001'].seen_nonces.add('only-in-L')
    l.devices['DEV-001'].zone = 'changed'
    assert 'only-in-L' not in b.controller.ledgers['DEV-001'].seen_nonces and b.devices['DEV-001'].zone != 'changed'


# 5 ---------------------------------------------------------------------------------------------------------
def test_same_inputs_before_divergence():
    never_block = lambda: ZeroTrustController(allow_threshold=0.0, block_threshold=0.0)  # noqa: E731
    t = TwinSimulation.from_scenario('multi_attacker', seed=4, mode='both', bayes_controller_factory=never_block,
                                     llm_controller_factory=lambda: LLMController(FakeLLM(trust=0.9, action='ALLOW'), synchronous=True))
    t.run(200)
    key = lambda r: (r['rid'], r['tick'], r['src'], r['dst'], r['event_type'], r['port'], r['bytes'], r['zk'], r['replay'],  # noqa: E731
                     tuple(sorted(r['alarms'].items())))
    sb = [key(r) for r in t.bayes.records if r.get('kind') == 'request']
    sl = [key(r) for r in t.llm.records if r.get('kind') == 'request']
    assert sb and sb == sl
    assert not any(r.get('kind') == 'dropped' for r in t.bayes.records + t.llm.records)


# 6 ---------------------------------------------------------------------------------------------------------
def test_no_fallback_on_timeouts_sync():
    fake = FakeLLM(raise_exc=LLMUnavailable('LLM request failed: timed out'))
    t = twin('both', fake)
    t.run(60)
    evals = [r for r in t.llm.records if r.get('kind') == 'llm_eval']
    assert evals and all(not r['ok'] and r['error'] == 'timeout' and r['llm_trust'] is None for r in evals)
    assert t.llm.controller.errors['timeout'] == len(evals)
    for s in t.llm.controller.snapshot().values():
        assert s['trust'] is None and s['decision'] == 'PENDING'
    assert any(s['error'] == 'timeout' for s in t.llm.controller.snapshot().values())
    reqs = [r for r in t.llm.records if r.get('kind') == 'request']
    assert all(r['trust_before'] is None for r in reqs)


def test_no_fallback_on_real_wall_clock_timeout():
    fake = FakeLLM(delay=1.0)
    t = twin('llm', fake, synchronous=False, timeout_s=0.1, every_n=3)
    for _ in range(12):
        t.step()
    evals = [r for r in t.llm.records if r.get('kind') == 'llm_eval']
    assert evals and all(r['error'] == 'timeout' for r in evals)
    t.shutdown()


def test_invalid_outputs_are_errors_and_clamps_are_counted():
    with pytest.raises(InvalidLLMOutput):
        parse_llm_output({'action': 'ALLOW'})
    with pytest.raises(InvalidLLMOutput):
        parse_llm_output({'trust_score': 0.5, 'action': 'MAYBE'})
    clean, clamped = parse_llm_output({'trust_score': 1.4, 'action': 'step_up', 'confidence': -0.2})
    assert clean['trust_score'] == 1.0 and clean['action'] == 'STEP-UP' and clean['confidence'] == 0.0 and clamped == 2
    t = twin('llm', FakeLLM(reply={'trust': 0.5}))
    t.run(30)
    assert t.llm.controller.errors['invalid_schema'] > 0
    assert all(s['trust'] is None for s in t.llm.controller.snapshot().values())


# 7 ---------------------------------------------------------------------------------------------------------
def test_bayesian_world_is_deterministic_and_untouched_by_llm():
    a = twin('both', FakeLLM(trust=0.1, action='BLOCK')); a.run(120)
    b = twin('both', FakeLLM(trust=0.9, action='ALLOW')); b.run(120)
    c = twin('bayesian'); c.run(120)
    assert a.bayes.records == b.bayes.records == c.bayes.records
    assert a.bayes.trust_history == c.bayes.trust_history


# enforcement, pacing, probation, real HTTP ----------------------------------------------------------------------
def test_llm_world_enforces_its_own_block_and_probation():
    rule = lambda c: (0.1, 'BLOCK') if c['window_facts'].get('token_requests', 0) > 6 else (0.85, 'ALLOW')  # noqa: E731
    # the schedule this test was written for (every 5 events, no minimum gap)
    t = twin('both', FakeLLM(rule=rule), scenario='token_flood_attack', seed=2, every_n=5, min_ticks_between=0)
    t.run(110)
    q = [r for r in t.llm.records if r.get('kind') == 'llm_eval' and r.get('quarantined_until')]
    assert q and q[0]['src'] == 'DEV-007'
    assert any(r.get('kind') == 'dropped' and r['src'] == 'DEV-007' and 'LLM world' in r['reason'] for r in t.llm.records)
    assert any('PROBATION' in r.get('text', '') for r in t.llm.records)
    assert any(r.get('kind') == 'llm_eval' and r['trigger'] == 'released from quarantine' for r in t.llm.records)
    # current defaults: same enforcement, and a released device is evaluated in its release tick
    t = twin('both', FakeLLM(rule=rule), scenario='token_flood_attack', seed=2)
    t.run(110)
    q = [r for r in t.llm.records if r.get('kind') == 'llm_eval' and r.get('quarantined_until')]
    assert q and q[0]['src'] == 'DEV-007'
    releases = [r['tick'] for r in t.llm.records if r.get('kind') == 'note' and 'DEV-007 released' in r['text']]
    assert releases and all(any(e.get('kind') == 'llm_eval' and e['src'] == 'DEV-007' and e['tick'] == rt for e in t.llm.records)
                            for rt in releases)


def test_pending_default_before_first_evaluation():
    t = twin('llm', FakeLLM(trust=0.9, action='ALLOW'), every_n=1000)
    t.run(10)
    reqs = [r for r in t.llm.records if r.get('kind') == 'request']
    assert reqs and all(r['reason'].startswith('PENDING') for r in reqs if r['zk'] == 'PASS' and not r['replay'])
    assert all(s['decision'] == 'PENDING' and s['trust'] is None for s in t.llm.controller.snapshot().values()
               if s['error'] is None)


def test_tick_waits_for_concurrent_llm_calls_and_reports_progress():
    seen = []
    t = twin('both', FakeLLM(trust=0.9, action='ALLOW', delay=0.3), synchronous=False, every_n=1)
    t0 = time.perf_counter()
    t.step(progress=seen.append)
    elapsed = time.perf_counter() - t0
    assert seen and seen[0] > 1 and 0.3 <= elapsed < 0.3 * seen[0]   # waited, and calls ran concurrently
    assert all(r['ok'] for r in t.llm.records if r.get('kind') == 'llm_eval')
    t.shutdown()


def test_real_http_path_with_local_stub():
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append(body)
            reply = {'choices': [{'message': {'content': json.dumps({'trust_score': 0.61, 'action': 'STEP-UP', 'confidence': 0.7,
                                                                       'reason': 'stub', 'factors': ['requests']})}}]}
            data = json.dumps(reply).encode()
            self.send_response(200); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *a):
            pass

    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    try:
        cfg = LLMConfig('openai_compatible', 'stub', f'http://127.0.0.1:{server.server_port}/v1/chat/completions', 'k', 5, 0.0)
        evaluator = LLMEvaluator(cfg, system_prompt=LIVE_SYSTEM_PROMPT)
        t = TwinSimulation.from_scenario('normal_day', seed=8, mode='llm', llm_controller_factory=lambda: LLMController(evaluator))
        t.run(12)
        evals = [r for r in t.llm.records if r.get('kind') == 'llm_eval']
        assert evals and all(r['ok'] and r['llm_trust'] == 0.61 and r['llm_action'] == 'STEP-UP' for r in evals)
        assert seen[0]['messages'][0]['content'] == LIVE_SYSTEM_PROMPT
        t.shutdown()
    finally:
        server.shutdown()


def test_unreachable_endpoint_marks_offline_and_world_keeps_running():
    cfg = LLMConfig('ollama', 'x', 'http://127.0.0.1:9/api/chat', '', 2, 0.0)
    t = TwinSimulation.from_scenario('normal_day', seed=6, mode='both',
                                     llm_controller_factory=lambda: LLMController(LLMEvaluator(cfg, LIVE_SYSTEM_PROMPT), every_n=2))
    t.run(15)
    assert t.llm.controller.offline and t.llm.controller.errors['unreachable'] > 0 and t.llm.tick == 15
    t.shutdown()
