"""Part A: a faster LLM world — compact context, short schema-constrained replies, fewer calls."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'tests'))

from core import live_config as C  # noqa: E402
from core.live_llm import (LIVE_RESPONSE_SCHEMA, LIVE_SYSTEM_PROMPT, LLMController, compact_facts, make_live_evaluator,  # noqa: E402
                           parse_llm_output)
from core.twin import TwinSimulation  # noqa: E402
from fake_llm import RuleBasedFakeLLM  # noqa: E402
from test_live_llm import FORBIDDEN_SUBSTRINGS, walk  # noqa: E402

REPLY = {'trust_score': 0.61, 'action': 'STEP-UP', 'confidence': 0.7, 'reason': 'stub.', 'factors': ['requests']}


class Recorder:
    """Rule-based fake that also records every context it receives."""
    def __init__(self):
        self.rule, self.contexts = RuleBasedFakeLLM(), []

    def evaluate(self, ctx):
        self.contexts.append(json.loads(json.dumps(ctx)))
        return self.rule.evaluate(ctx)


def run_llm(scenario, seed, ticks, fake=None, **kw):
    fake = fake or Recorder()
    t = TwinSimulation.from_scenario(scenario, seed=seed, mode='llm', llm_controller_factory=lambda: LLMController(fake, synchronous=True, **kw))
    t.run(ticks)
    return t, fake


def serve(handler_reply):
    """A local HTTP stub. handler_reply(body) -> (status, dict). Returns (server, list of received bodies)."""
    seen = []

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
            seen.append({'body': body, 'auth': self.headers.get('Authorization')})
            status, out = handler_reply(body)
            data = json.dumps(out).encode()
            self.send_response(status); self.send_header('Content-Type', 'application/json')
            self.send_header('Content-Length', str(len(data))); self.end_headers(); self.wfile.write(data)

        def log_message(self, *a):
            pass
    server = HTTPServer(('127.0.0.1', 0), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, seen


def ollama_ok(body):
    return 200, {'message': {'content': json.dumps(REPLY)}, 'prompt_eval_count': 321, 'eval_count': 42}


# ----- compact context ----------------------------------------------------------------------------------------
@pytest.mark.parametrize('scenario', ['multi_attacker', 'replay_attack', 'normal_day'])
def test_compact_context_has_no_forbidden_keys_and_no_zero_fields(scenario):
    t, fake = run_llm(scenario, 4, 120)
    assert fake.contexts
    for ctx in fake.contexts:
        for path, key, value in walk(ctx):
            if key is not None:
                assert not any(bad in key.lower() for bad in FORBIDDEN_SUBSTRINGS), (path, key)
        assert compact_facts(ctx['window_facts']) == ctx['window_facts']   # nothing zero or empty left
        assert len(ctx['recent_requests']) <= C.LLM_RECENT_REQUESTS and ctx['omitted_fields_mean_zero'] is True
    t.shutdown()


def test_compact_json_is_much_shorter_than_indented():
    t, fake = run_llm('compromised_valid_token', 42, 60)
    ev = make_live_evaluator('ollama', 'm', 'http://x')
    ctx = fake.contexts[-1]
    assert ev.user_prompt(ctx) == json.dumps(ctx, separators=(',', ':'))
    assert len(ev.user_prompt(ctx)) < 0.8 * len(json.dumps(ctx, indent=2))   # measured ≈ 0.71
    t.shutdown()


# ----- request shape ---------------------------------------------------------------------------------------------
def test_ollama_payload_sends_schema_output_cap_and_runtime_options():
    body = make_live_evaluator('ollama', 'llama3.2:3b', 'http://x')._payload('{}')
    assert isinstance(body['format'], dict) and body['format'] == LIVE_RESPONSE_SCHEMA
    assert set(body['format']['required']) == {'trust_score', 'action', 'confidence', 'reason', 'factors'}
    assert body['format']['properties']['action']['enum'] == ['ALLOW', 'STEP-UP', 'BLOCK']
    assert body['format']['properties']['factors']['maxItems'] == 4
    assert 'behavior_assessment' not in body['format']['properties']
    assert body['options'] == {'temperature': 0.0, 'num_predict': C.LLM_MAX_OUTPUT_TOKENS, 'num_ctx': C.LLM_NUM_CTX}
    assert body['keep_alive'] == C.LLM_KEEP_ALIVE and body['think'] is False and body['stream'] is False
    assert body['model'] == 'llama3.2:3b'


def test_openai_payload_sends_max_tokens_and_json_schema():
    body = make_live_evaluator('openai_compatible', 'any', 'http://x')._payload('{}')
    assert body['max_tokens'] == C.LLM_MAX_OUTPUT_TOKENS
    assert body['response_format']['type'] == 'json_schema'
    assert body['response_format']['json_schema']['schema'] == LIVE_RESPONSE_SCHEMA
    assert 'keep_alive' not in body and 'think' not in body and 'options' not in body


def test_real_ollama_http_path_sends_everything_and_reads_token_counts():
    server, seen = serve(ollama_ok)
    try:
        ev = make_live_evaluator('ollama', 'llama3.2:3b', f'http://127.0.0.1:{server.server_port}/api/chat', timeout=5)
        t = TwinSimulation.from_scenario('normal_day', seed=8, mode='llm', llm_controller_factory=lambda: LLMController(ev))
        t.run(25)
        evals = [r for r in t.llm.records if r.get('kind') == 'llm_eval']
        assert evals and all(r['ok'] for r in evals)
        assert all(r['llm_usage'] == {'prompt_tokens': 321, 'output_tokens': 42} for r in evals)
        b = seen[0]['body']
        assert isinstance(b['format'], dict) and b['options']['num_predict'] == C.LLM_MAX_OUTPUT_TOKENS and b['think'] is False
        assert b['messages'][0]['content'] == LIVE_SYSTEM_PROMPT
        assert '\n' not in b['messages'][1]['content']          # compact JSON
        t.shutdown()
    finally:
        server.shutdown()


def test_think_field_is_dropped_when_the_model_rejects_it():
    def reply(body):
        if 'think' in body:
            return 400, {'error': '"llama3.2:3b" does not support thinking'}
        return ollama_ok(body)
    server, seen = serve(reply)
    try:
        ev = make_live_evaluator('ollama', 'llama3.2:3b', f'http://127.0.0.1:{server.server_port}/api/chat', timeout=5)
        assert ev.evaluate_raw({'x': 1})['trust_score'] == 0.61
        assert ev.evaluate_raw({'x': 2})['trust_score'] == 0.61
        assert ['think' in s['body'] for s in seen] == [True, False, False]   # one retry, then remembered
        assert ev.think_supported is False
    finally:
        server.shutdown()


def test_openai_falls_back_to_json_object_and_works_without_api_key():
    def reply(body):
        if body['response_format']['type'] == 'json_schema':
            return 400, {'error': 'response_format json_schema not supported'}
        return 200, {'choices': [{'message': {'content': json.dumps(REPLY)}}], 'usage': {'prompt_tokens': 300, 'completion_tokens': 40}}
    server, seen = serve(reply)
    try:
        ev = make_live_evaluator('openai_compatible', 'local', f'http://127.0.0.1:{server.server_port}/v1/chat/completions', api_key='', timeout=5)
        out = ev.evaluate_raw({'x': 1})
        assert out['action'] == 'STEP-UP' and out['_usage'] == {'prompt_tokens': 300, 'output_tokens': 40}
        assert [s['body']['response_format']['type'] for s in seen] == ['json_schema', 'json_object']
        assert all(s['auth'] is None for s in seen) and all(s['body']['max_tokens'] == C.LLM_MAX_OUTPUT_TOKENS for s in seen)
    finally:
        server.shutdown()


def test_warm_up_sends_a_tiny_request_with_keep_alive():
    server, seen = serve(lambda b: (200, {'message': {'content': '{}'}}))
    try:
        ev = make_live_evaluator('ollama', 'llama3.2:3b', f'http://127.0.0.1:{server.server_port}/api/chat', timeout=5)
        assert ev.warm_up() >= 0
        b = seen[0]['body']
        assert b['options']['num_predict'] == 1 and b['keep_alive'] == C.LLM_KEEP_ALIVE and len(b['messages']) == 1
    finally:
        server.shutdown()


# ----- reply parsing -----------------------------------------------------------------------------------------------
def test_parsing_works_without_behavior_assessment_and_risk_score():
    out, clamped = parse_llm_output(dict(REPLY))
    assert out['behavior_assessment'] == [] and out['action'] == 'STEP-UP' and clamped == 0
    assert abs(out['risk_score'] - (1 - 0.61)) < 1e-12
    assert 'behavior_assessment' not in LIVE_SYSTEM_PROMPT


# ----- fewer calls ----------------------------------------------------------------------------------------------------
def test_defaults_changed_as_requested():
    assert (C.LLM_EVERY_N, C.LLM_MIN_TICKS_BETWEEN_EVALS, C.LLM_RECENT_REQUESTS, C.LLM_MAX_OUTPUT_TOKENS) == (10, 3, 5, 160)
    assert (C.LLM_KEEP_ALIVE, C.LLM_NUM_CTX, C.LLM_THINK, C.LLM_TEMPERATURE) == ('30m', 4096, False, 0.0)


@pytest.mark.parametrize('scenario,seed', [('multi_attacker', 1), ('replay_attack', 3), ('normal_day', 2)])
def test_min_ticks_between_evaluations_and_hard_events_stay_immediate(scenario, seed):
    t, _ = run_llm(scenario, seed, 150)
    evals = [r for r in t.llm.records if r.get('kind') == 'llm_eval']
    assert evals
    last = {}
    for e in evals:
        if e['src'] in last and not e['trigger'].startswith('hard event'):
            assert e['tick'] - last[e['src']] >= C.LLM_MIN_TICKS_BETWEEN_EVALS, (e['src'], e['tick'], last[e['src']])
        last[e['src']] = e['tick']
    # every hard event (failed proof, replayed nonce, failed step-up) is evaluated at the end of that same tick
    for r in t.llm.records:
        if r.get('kind') != 'request':
            continue
        hard = r['zk'] == 'FAIL' or r['replay'] or (r.get('step_up') and r['action'] == 'BLOCK')
        if hard:
            assert any(e['src'] == r['src'] and e['tick'] == r['tick'] for e in evals), (r['rid'], r['reason'])
    t.shutdown()


def test_hard_event_inside_the_gap_is_still_evaluated_at_once():
    t, _ = run_llm('replay_attack', 0, 120)
    evals = [r for r in t.llm.records if r.get('kind') == 'llm_eval']
    by_dev = {}
    for e in evals:
        by_dev.setdefault(e['src'], []).append(e)
    close_hard = [b for es in by_dev.values() for a, b in zip(es, es[1:])
                  if b['tick'] - a['tick'] < C.LLM_MIN_TICKS_BETWEEN_EVALS and b['trigger'].startswith('hard event')]
    assert close_hard, 'expected at least one hard-event evaluation inside the minimum gap'
    t.shutdown()


def test_fewer_evaluations_than_the_old_schedule():
    old, _ = run_llm('compromised_valid_token', 42, 60, every_n=5, min_ticks_between=0)
    new, _ = run_llm('compromised_valid_token', 42, 60)
    n_old = sum(r.get('kind') == 'llm_eval' for r in old.llm.records)
    n_new = sum(r.get('kind') == 'llm_eval' for r in new.llm.records)
    assert n_old == 127 and n_new < 0.7 * n_old
    old.shutdown(); new.shutdown()


def test_bayesian_world_is_identical_under_the_new_llm_defaults():
    from golden_logs import GOLDEN, all_fingerprints
    golden = json.loads(GOLDEN.read_text(encoding='utf-8'))
    now = all_fingerprints(llm_settings={})
    assert {k: v['bayesian'] for k, v in now.items()} == {k: v['bayesian'] for k, v in golden.items()}


# ----- speed check script ----------------------------------------------------------------------------------------------
def test_speed_check_runs_with_the_fake_llm(capsys):
    import llm_speed_check
    assert llm_speed_check.main(['--fake-llm', '--n', '20', '--compare']) == 0
    out = capsys.readouterr().out
    assert 'latency median' in out and 'errors' in out and 'old (before speed changes)' in out and '| 127 | 70 |' in out
