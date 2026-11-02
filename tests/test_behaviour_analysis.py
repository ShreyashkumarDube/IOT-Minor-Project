"""Tests for the calculation view, attack-kind inference and alarm breakdown (core/behaviour_analysis.py).

1 logs unchanged · 2 re-addition matches · 3 every fact traceable · 4 no leakage · 5 attack inference
6 alarm statistics · 8 batch outputs.  (7, the AppTest, is in test_calc_ui.py.)
"""
import ast
import json
import random
import subprocess
import sys
from collections import Counter
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
sys.path.insert(0, str(ROOT / 'tests'))

from core import behaviour_analysis as BA  # noqa: E402
from core import live_config as C  # noqa: E402
from core.agents import ATTACKER_TYPES  # noqa: E402
from core.controller import ZeroTrustController  # noqa: E402
from core.live_llm import LLMController  # noqa: E402
from core.live_sim import list_scenarios, simulation_from_scenario  # noqa: E402
from core.sensors import SensorSuite  # noqa: E402
from core.twin import TwinSimulation  # noqa: E402
from fake_llm import RuleBasedFakeLLM  # noqa: E402


def fake_twin(scenario, seed, ticks=None):
    twin = TwinSimulation.from_scenario(scenario, seed=seed, mode='both',
                                        llm_controller_factory=lambda: LLMController(RuleBasedFakeLLM(), synchronous=True))
    twin.run(ticks or twin.primary.scenario.get('ticks_suggested', 160))
    return twin


# 1 ------------------------------------------------------------------------------------------------------
def test_logs_unchanged_against_golden_fingerprints():
    """Bayesian and FakeLLM-world logs are identical to the ones recorded BEFORE the calculation view existed."""
    from golden_logs import GOLDEN, all_fingerprints
    assert all_fingerprints() == json.loads(GOLDEN.read_text(encoding='utf-8'))


# 2 ------------------------------------------------------------------------------------------------------
def test_readding_the_shown_terms_reproduces_the_stored_trust():
    live = simulation_from_scenario('multi_attacker', seed=42)
    live.run(200)
    checked, ok = BA.all_bayes_rechecks(live)
    assert checked > 1000 and ok == checked
    for r in live.records:
        if r.get('kind') == 'request':
            steps = BA.bayes_steps(r)
            final = steps['step_up']['trust_after'] if steps['step_up'] else steps['trust_after']
            assert round(final, 4) == r['trust_after']           # the logged (rounded) value is the same number
            assert all(abs(row['term value'] - row['weight'] * row['ln(ratio)']) < 1e-12 for row in steps['evidence'])


def test_decay_is_reproduced_from_the_previous_update():
    live = simulation_from_scenario('compromised_valid_token', seed=42)
    live.run(60)
    checks = [BA.bayes_steps(r, live)['decay_check'] for r in live.records if r.get('kind') == 'request' and r['calc']['probation'] is False]
    checks = [c for c in checks if c is not None]
    assert len(checks) > 100 and sum(checks) / len(checks) > 0.98   # misses only where a probation reset happened in between


# 3 ------------------------------------------------------------------------------------------------------
def test_every_llm_window_fact_is_traceable_to_request_ids():
    twin = fake_twin('multi_attacker', 42)
    evals = [e for e in twin.llm.records if e.get('kind') == 'llm_eval']
    assert len(evals) > 100
    for e in evals:
        for row in BA.window_trace(twin.llm, e):
            assert row['matches'], (e['tick'], e['src'], row['fact'])
            if row['type'] in ('count', 'distinct'):
                assert row['n contributing'] == row['value sent to LLM'], (e['tick'], e['src'], row['fact'])
    twin.shutdown()


# 4 ------------------------------------------------------------------------------------------------------
LEAK_WORDS = ('sensor_draws', 'request_is_malicious', 'request_truth', 'ground_truth') + tuple(ATTACKER_TYPES) + \
             ('noisy_honest', 'noisy_benign') + tuple(BA.KIND_LABEL.values())


def test_no_new_examiner_data_reaches_a_controller_or_the_llm(monkeypatch):
    seen = []
    for cls in (ZeroTrustController, LLMController):
        original = cls.process

        def spy(self, event, alarms, _orig=original):
            seen.append((event, dict(alarms)))
            return _orig(self, event, alarms)
        monkeypatch.setattr(cls, 'process', spy)
    twin = fake_twin('multi_attacker', 0, ticks=120)
    assert seen
    for event, alarms in seen:
        assert set(alarms) <= set(C.SENSORS) and all(isinstance(v, bool) for v in alarms.values())   # never u or p
        assert set(event.details) <= {'port', 'bytes', 'zone_from', 'zone_to', 'sensitivity'}
    for e in twin.llm.records:
        if e.get('kind') == 'llm_eval':
            text = json.dumps(e['context']).lower()
            assert '"u"' not in text and '"p"' not in text
            for word in LEAK_WORDS:
                assert word.lower() not in text, word
    twin.shutdown()


def test_behaviour_analysis_is_never_imported_by_a_controller():
    for module in ('controller.py', 'live_llm.py', 'events.py', 'sensors.py', 'zk_auth.py'):
        tree = ast.parse((ROOT / 'core' / module).read_text(encoding='utf-8'))
        names = {a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names} | \
                {(n.module or '') for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)} | \
                {a.name for n in ast.walk(tree) if isinstance(n, ast.ImportFrom) for a in n.names}
        assert not any('behaviour_analysis' in x for x in names), module
    code = 'import sys; import core.controller, core.live_llm, core.live_sim; print("core.behaviour_analysis" in sys.modules)'
    out = subprocess.run([sys.executable, '-c', code], cwd=ROOT, capture_output=True, text=True, check=True)
    assert out.stdout.strip() == 'False'


def test_attack_inference_reads_no_ground_truth(monkeypatch):
    """Scramble every bit of ground truth after the run: the inferred kinds must not change."""
    live = simulation_from_scenario('multi_attacker', seed=3)
    live.run(200)
    before = {d: v['kind'] for d, v in BA.infer_world(live).items()}
    live.ground_truth = {d: {'agent': 'honest', 'attack_start': None} for d in live.ground_truth}
    live.request_truth = {k: False for k in live.request_truth}
    live.sensor_draws = {}
    live.agents = {}
    assert {d: v['kind'] for d, v in BA.infer_world(live).items()} == before


# 5 ------------------------------------------------------------------------------------------------------
ATTACKER_MIN = 0.90   # X: share of attacker-runs whose kind is recognised (measured 78/80 = 97.5% when written)
PER_KIND_MIN = 0.70   # every attack kind on its own (slow_low measured 8/10)
HONEST_MIN = 0.95     # Y: honest / noisy-honest device-runs NOT labelled as an attack (measured 550/550)


def test_attack_kind_inference_accuracy_on_every_preset():
    per_kind, honest = Counter(), Counter()
    for scenario in list_scenarios():
        for seed in range(10):
            live = simulation_from_scenario(scenario, seed=seed)
            live.run(live.scenario.get('ticks_suggested', 160))
            for row in BA.score_inference(live):
                if row['truth'] in ('honest', 'noisy_honest'):
                    honest[row['correct']] += 1
                else:
                    per_kind[(row['truth'], row['correct'])] += 1
    att_ok = sum(n for (k, ok), n in per_kind.items() if ok)
    att_all = sum(per_kind.values())
    report = {k: f"{per_kind[(k, True)]}/{per_kind[(k, True)] + per_kind[(k, False)]}" for k in sorted({k for k, _ in per_kind})}
    print('attack-kind recognition:', report, f'honest {honest[True]}/{sum(honest.values())}')
    assert att_ok / att_all >= ATTACKER_MIN, report
    for k in {k for k, _ in per_kind}:
        assert per_kind[(k, True)] / (per_kind[(k, True)] + per_kind[(k, False)]) >= PER_KIND_MIN, (k, report)
    assert honest[True] / sum(honest.values()) >= HONEST_MIN


# 6 ------------------------------------------------------------------------------------------------------
def test_sensor_draw_recording_keeps_the_same_random_stream():
    for seed in range(50):
        for malicious in (True, False):
            a = SensorSuite(random.Random(0)).observe(malicious, random.Random(seed))
            b, draws = SensorSuite(random.Random(0)).observe_detailed(malicious, random.Random(seed))
            assert a == b
            assert all((draws[s]['u'] < draws[s]['p']) == b[s] for s in C.SENSORS)
            assert all(draws[s]['p'] == C.SENSORS[s][0 if malicious else 1] for s in C.SENSORS)


def test_both_worlds_record_identical_sensor_draws_and_alarms_match_them():
    twin = fake_twin('scanning_attack', 1, ticks=100)
    assert twin.bayes.sensor_draws == twin.llm.sensor_draws
    for r in twin.bayes.records:
        if r.get('kind') == 'request':
            d = twin.bayes.sensor_draws[r['rid']]
            assert r['alarms'] == {s: d[s]['u'] < d[s]['p'] for s in C.SENSORS}
    twin.shutdown()


def test_observed_alarm_rates_match_the_configured_rates_over_long_runs():
    mal = Counter(); ben = Counter(); n_mal = n_ben = 0
    for seed in range(5):
        live = simulation_from_scenario('multi_attacker', seed=seed)
        live.run(200)
        for rid, draws in live.sensor_draws.items():
            m = live.request_truth[rid]
            n_mal += m; n_ben += not m
            for s, d in draws.items():
                (mal if m else ben)[s] += d['u'] < d['p']
    assert n_mal > 1000 and n_ben > 5000
    for s, (tp, fp) in C.SENSORS.items():
        assert abs(mal[s] / n_mal - tp) < 0.04, s
        assert abs(ben[s] / n_ben - fp) < 0.02, s
    stats = BA.alarm_stats(live)
    assert {r['sensor'] for r in stats} == set(C.SENSORS)


def test_alarm_combinations_use_the_controller_terms():
    ctl = ZeroTrustController()
    rows = BA.alarm_combinations()
    assert len(rows) == 8
    for row in rows:
        fired = set(row['alarms fired'].split(' + ')) if row['alarms fired'] != 'none' else set()
        expected = sum(ctl._sensor_term(s, s in fired) for s in C.SENSORS)
        assert abs(row['combined Bayesian term'] - round(expected, 4)) < 1e-12


# 8 ------------------------------------------------------------------------------------------------------
def test_batch_writes_attack_kind_comparison_and_alarm_stats(tmp_path):
    import run_batch
    assert run_batch.main(['--mode', 'both', '--fake-llm', '--scenario', 'compromised_valid_token', '--seeds', '1',
                           '--ticks', '70', '--out', str(tmp_path)]) == 0
    for name in ('attack_kind_comparison.csv', 'attack_kind_comparison.md', 'alarm_stats.csv', 'attack_kind_detection_rate.png'):
        assert (tmp_path / name).exists(), name
    md = (tmp_path / 'attack_kind_comparison.md').read_text(encoding='utf-8')
    assert 'FakeLLM' in md and 'compromised_valid' in md
    single = tmp_path / 'single'
    assert run_batch.main(['--scenario', 'normal_day', '--seeds', '1', '--ticks', '40', '--no-plots', '--out', str(single)]) == 0
    assert (single / 'alarm_stats.csv').exists()
