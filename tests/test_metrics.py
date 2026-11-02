"""Stage 6: metrics from ground truth after the fact, exports, batch runner and Results tab."""
import csv
import io
import json
import sys
from pathlib import Path

from streamlit.testing.v1 import AppTest

from core.live_llm import LLMController
from core.twin import TwinSimulation
from core.live_sim import simulation_from_scenario
from core.metrics import compute_metrics, log_rows, metrics_json, summary_markdown, to_csv, trust_csv

ROOT = Path(__file__).resolve().parents[1]


class FakeLLM:
    def evaluate(self, context):
        odd = context['window_facts'].get('requests_without_configured_link', 0) > 3
        return {'trust_score': 0.2 if odd else 0.9, 'action': 'BLOCK' if odd else 'ALLOW', 'risk_score': .5,
                'reason': 'stub', 'confidence': .7, 'factors': [], 'behavior_assessment': []}


def test_metrics_on_multi_attacker_run():
    live = simulation_from_scenario('multi_attacker', seed=1); live.run(200)
    m = compute_metrics(live)
    assert {a['device'] for a in m['attackers']} == {'DEV-002', 'DEV-007', 'DEV-008'}
    assert m['confusion']['TP'] + m['confusion']['FN'] == 3 and m['confusion']['TP'] + m['confusion']['FP'] + m['confusion']['TN'] + m['confusion']['FN'] == 9
    for a in m['attackers']:
        assert a['detected'] == (a['ttd_block'] is not None)
        if a['ttd_block'] is not None:
            assert a['ttd_block'] >= 0 and a['ttd_step_up'] <= a['ttd_block']


def test_unstarted_attack_is_not_counted():
    live = simulation_from_scenario('multi_attacker', seed=1); live.run(50)   # only the scanner (t=35) has started
    assert [a['device'] for a in compute_metrics(live)['attackers']] == ['DEV-002']


def test_normal_day_has_no_attackers_and_f1_undefined():
    live = simulation_from_scenario('normal_day', seed=0); live.run(100)
    m = compute_metrics(live)
    assert m['attackers'] == [] and m['detection_rate'] is None and m['f1'] is None
    assert 0 <= m['fpr_block'] <= 1 and len(m['honest']) == 9


def test_same_metrics_work_for_the_llm_world():
    t = TwinSimulation.from_scenario('scanning_attack', seed=2, mode='llm',
                                     llm_controller_factory=lambda: LLMController(FakeLLM(), synchronous=True))
    t.run(80)
    m = compute_metrics(t.llm)
    assert m['engine'] == 'llm' and m['llm']['evaluations'] > 0 and m['llm']['errors'] == 0
    scanner = next(a for a in m['attackers'] if a['device'] == 'DEV-002')
    assert scanner['detected']   # FakeLLM blocks whenever anomalies-like facts appear


def test_exports_are_well_formed():
    live = simulation_from_scenario('token_flood_attack', seed=3); live.run(60)
    m = compute_metrics(live)
    rows = list(csv.DictReader(io.StringIO(to_csv(log_rows(live)))))
    assert rows and {'tick', 'kind', 'src', 'action', 'text'} <= set(rows[0])
    trust_rows = list(csv.DictReader(io.StringIO(trust_csv(live))))
    assert len(trust_rows) == 60 and 'DEV-007' in trust_rows[0]
    assert json.loads(metrics_json(m))['scenario'] == 'token_flood_attack'
    md = summary_markdown(m)
    assert md.startswith('# Run summary') and 'DEV-007' in md


def test_batch_runner_writes_results(tmp_path):
    sys.path.insert(0, str(ROOT / 'experiments'))
    import run_batch
    assert run_batch.main(['--scenario', 'token_flood_attack', '--seeds', '2', '--ticks', '70', '--out', str(tmp_path)]) == 0
    for name in ('runs_summary.csv', 'runs_devices.csv', 'summary_by_scenario.csv', 'summary.md',
                 'detection_time_boxplot.png', 'fpr_bar.png', 'trust_curves_examples.png'):
        assert (tmp_path / name).exists() and (tmp_path / name).stat().st_size > 0, name


def test_results_tab_renders():
    at = AppTest.from_file(str(ROOT / 'app.py'), default_timeout=60).run()
    at.selectbox(key='live_scenario').set_value('token_flood_attack').run()
    at.button(key='live_reset').click().run()
    for _ in range(45):
        at.button(key='live_step').click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert 'Attackers detected' in labels and 'Honest quarantined (FPR)' in labels


def test_batch_runner_both_mode_with_fake_llm(tmp_path):
    sys.path.insert(0, str(ROOT / 'experiments'))
    import run_batch
    assert run_batch.main(['--mode', 'both', '--fake-llm', '--scenario', 'scanning_attack', '--seeds', '2',
                           '--ticks', '60', '--repeat', '2', '--out', str(tmp_path)]) == 0
    for name in ('comparison_runs.csv', 'comparison_mean_std.csv', 'comparison_mean_std.md', 'consistency.csv'):
        assert (tmp_path / name).stat().st_size > 0, name
    assert (tmp_path / 'scanning_attack_seed0' / 'comparison_report.md').exists()
    rows = list(csv.DictReader(io.StringIO((tmp_path / 'comparison_mean_std.csv').read_text())))
    assert any(r['parameter'] == 'Detection rate' and r['n_seeds'] == '2' for r in rows)
    cons = list(csv.DictReader(io.StringIO((tmp_path / 'consistency.csv').read_text())))
    assert {r['engine'] for r in cons} == {'bayesian', 'llm'}
    assert all(float(r['decision_consistency']) == 1.0 for r in cons if r['engine'] == 'bayesian')
    assert 'FAKE' in (tmp_path / 'comparison_mean_std.md').read_text()


def test_batch_runner_llm_only_mode(tmp_path):
    sys.path.insert(0, str(ROOT / 'experiments'))
    import run_batch
    assert run_batch.main(['--mode', 'llm', '--fake-llm', '--scenario', 'replay_attack', '--seeds', '2', '--ticks', '60',
                           '--no-plots', '--out', str(tmp_path)]) == 0
    assert 'world: llm' in (tmp_path / 'summary.md').read_text()
