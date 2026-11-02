"""Step 4: the comparison report computes every parameter for every preset (test 8), plus consistency and exports."""
import os
import random
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

from core.comparison import compare, repeat_consistency, summary_table, write_exports
from core.live_llm import LLMController
from core.live_sim import list_scenarios
from core.twin import TwinSimulation

APP = str(Path(__file__).resolve().parents[1] / 'app.py')
REQUIRED = [
    'Detection rate', 'Mean ticks to first STEP-UP (attackers)', 'Mean ticks to first BLOCK (detected attackers)',
    'Attackers blocked first', 'True positives', 'False positives', 'True negatives', 'False negatives', 'Precision',
    'Recall', 'F1', 'Accuracy (devices)', 'Specificity', 'FPR (quarantined) — honest devices',
    'FPR (quarantined) — noisy-honest devices', 'False-negative rate (missed attackers)',
    'Wrongful quarantines of honest devices', 'Total wrongful quarantine duration (ticks)',
    'Mean trust — honest devices', 'Mean trust — attackers while active', 'Separation gap (honest − attacker)',
    'Trust volatility (mean std of tick-to-tick change)', 'State changes per device (mean)',
    'Noisy-honest recovery to ALLOW (mean ticks)', 'Brier score (trust vs truth)', 'Per-request accuracy',
    'Decision agreement rate (same request)', 'Step-up challenges issued', 'Step-up challenges passed',
    'Step-up challenges failed', 'Evaluations performed', 'Latency per evaluation — mean (ms)',
    'Latency per evaluation — median (ms)', 'Latency per evaluation — p95 (ms)', 'Latency per evaluation — max (ms)',
    'LLM errors — timeout', 'LLM errors — invalid_json', 'Out-of-range values clamped', 'Mean confidence',
    'Mean confidence on correct verdicts', 'Mean confidence on incorrect verdicts', 'Throughput (ticks per second)',
    'LLM factor mentions supported by its window', 'LLM factor mentions NOT in its window']


class RuleLLM:
    """Deterministic fake that reasons over the facts (so the comparison has real differences)."""
    def __init__(self, noise=False):
        self.noise = random.Random(0) if noise else None

    def evaluate(self, c):
        w = {k: 0 for k in ('zk_failures', 'nonce_reuse_detected', 'requests_without_configured_link',
                            'token_requests', 'distinct_ports')} | c['window_facts']   # omitted fields mean zero
        bad = [f for f, hit in (('ZK failures', w['zk_failures'] > 1), ('nonce reuse', w['nonce_reuse_detected'] > 0),
                                ('requests without configured link', w['requests_without_configured_link'] > 2),
                                ('token flood', w['token_requests'] > 6), ('port scan', w['distinct_ports'] > 5)) if hit]
        trust = max(0.05, 0.92 - 0.3 * len(bad)) - (self.noise.random() * 0.3 if self.noise else 0)
        action = 'ALLOW' if trust >= 0.72 else ('STEP-UP' if trust >= 0.45 else 'BLOCK')
        return {'trust_score': trust, 'action': action, 'confidence': 0.8 if bad else 0.6,
                'reason': '; '.join(bad) or 'normal', 'factors': bad or ['within baseline']}


def run_both(scenario, seed=1, ticks=None, llm=None):
    t = TwinSimulation.from_scenario(scenario, seed=seed, mode='both',
                                     llm_controller_factory=lambda: LLMController(llm or RuleLLM(), synchronous=True))
    t.run(ticks or t.primary.scenario.get('ticks_suggested', 160))
    return t


@pytest.mark.parametrize('scenario', list_scenarios())
def test_report_computes_every_parameter_for_every_preset(scenario):
    t = run_both(scenario)
    cmp = compare(t)
    names = {r['parameter'] for r in cmp['rows']}
    missing = [p for p in REQUIRED if p not in names]
    assert not missing, missing
    table = summary_table(cmp)
    assert all({'Bayesian', 'LLM', 'difference (LLM − Bayes)', 'better'} <= set(r) for r in table)
    assert len(cmp['per_device']) == len(t.bayes.devices)
    assert 0 <= (cmp['agreement']['rate'] or 0) <= 1
    if 'slow' in scenario:
        assert 'Slow-and-low detected' in names


def test_repeat_consistency_bayesian_identical_llm_measured():
    make = lambda llm: (lambda: TwinSimulation.from_scenario('scanning_attack', seed=2, mode='both',  # noqa: E731
                                                            llm_controller_factory=lambda: LLMController(llm(), synchronous=True)))
    steady = repeat_consistency(make(RuleLLM), 3, 60)
    assert steady['bayesian']['decision_consistency'] == 1.0 and steady['bayesian']['mean_trust_std'] == 0.0
    assert steady['llm']['decision_consistency'] == 1.0
    counter = iter(range(100))
    noisy = repeat_consistency(make(lambda: RuleLLM(noise=True) if next(counter) % 2 else RuleLLM()), 2, 60)
    assert noisy['bayesian']['decision_consistency'] == 1.0
    assert noisy['llm']['mean_trust_std'] > 0


def test_exports_written(tmp_path):
    t = run_both('compromised_valid_token', ticks=90)
    files = write_exports(compare(t), t, str(tmp_path))
    names = {os.path.basename(f) for f in files}
    assert {'comparison_summary.csv', 'per_device_comparison.csv', 'disagreements.csv', 'bayesian_world_log.csv',
            'llm_world_log.csv', 'comparison_report.md', 'trust_curves.png', 'time_to_detect.png', 'false_positives.png',
            'latency_box.png', 'agreement_over_time.png'} <= names
    md = (tmp_path / 'comparison_report.md').read_text()
    assert '## Detection' in md and '![trust_curves.png](trust_curves.png)' in md
    assert (tmp_path / 'comparison_summary.csv').read_text().startswith('section,parameter,Bayesian,LLM')


def test_comparison_tab_in_app(monkeypatch):
    from core import llm_engine
    rule = RuleLLM()
    monkeypatch.setattr(llm_engine.LLMEvaluator, 'evaluate_raw', lambda self, ctx: rule.evaluate(ctx))
    at = AppTest.from_file(APP, default_timeout=120).run()
    at.radio(key='live_run_mode').set_value('both').run()
    at.button(key='live_reset').click().run()
    for _ in range(15):
        at.button(key='live_step').click().run()
    assert not at.exception
    labels = [m.label for m in at.metric]
    assert 'Attackers caught (B / L)' in labels and 'Decision agreement' in labels
    at.number_input(key='cmp_repeats').set_value(2).run()
    at.button(key='cmp_repeat').click().run()
    assert not at.exception
    at.button(key='cmp_export').click().run()
    assert not at.exception and at.session_state.cmp_zip[1][:2] == b'PK'
