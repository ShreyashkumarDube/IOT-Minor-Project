"""AppTest for Live Network mode: loads by default, Step advances, Reset works, Classic still reachable."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def test_live_mode_step_reset_and_classic_switch():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    assert at.button_group(key='app_mode').value == 'Live network'   # Live is the default
    assert at.toggle(key='live_reveal').value is False                # ground truth hidden by default
    live = at.session_state.live_sim
    assert live.tick == 0
    for _ in range(5):
        at.button(key='live_step').click().run()
    assert not at.exception
    live = at.session_state.live_sim
    assert live.tick == 5 and len(live.records) > 0
    assert 'Ground truth' not in at.dataframe[0].value.columns
    at.toggle(key='live_reveal').set_value(True).run()
    assert 'Ground truth' in at.dataframe[0].value.columns
    at.button(key='live_reset').click().run()
    assert not at.exception and at.session_state.live_sim.tick == 0
    at.button_group(key='app_mode').set_value('Classic evaluator').run()
    assert not at.exception and any(b.key == 'classic_run' for b in at.sidebar.button)


def test_live_flow_deep_dive_panels_render():
    at = AppTest.from_file(APP, default_timeout=60).run()
    at.selectbox(key='live_scenario').set_value('Current network (from the builder)').run()
    at.button(key='live_reset').click().run()
    for _ in range(31):
        at.button(key='live_step').click().run()
    at.selectbox(key='live_flow_device').set_value('DEV-005').run()
    for stage in ['Request packet', 'Topology gate', 'Zero-Knowledge', 'Behavior inspection',
                  'Trust calculation', 'Policy engine', 'Final decision']:
        at.radio(key='live_flow_stage').set_value(stage).run()
        assert not at.exception, stage
    assert at.session_state.live_sim.controller.ledgers['DEV-005'].classification == 'Quarantined'


def test_presets_load_and_trigger_attack():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert at.selectbox(key='live_scenario').value == 'normal_day'
    at.selectbox(key='live_scenario').set_value('compromised_valid_token').run()
    at.button(key='live_reset').click().run()
    live = at.session_state.live_sim
    assert live.scenario_label == 'compromised_valid_token' and len(live.devices) == 9
    at.selectbox(key='live_attack_device').set_value('DEV-001').run()
    at.selectbox(key='live_attack_type').set_value('scanning').run()
    at.button(key='live_trigger').click().run()
    assert not at.exception
    assert at.session_state.live_sim.ground_truth['DEV-001'] == {'agent': 'scanning', 'attack_start': 1}


def _both_mode_run(at, steps):
    at.radio(key='live_run_mode').set_value('both').run()
    at.button(key='live_reset').click().run()
    for _ in range(steps):
        at.button(key='live_step').click().run()
    assert not at.exception
    return next(d.value for d in at.dataframe if 'LLM trust' in d.value.columns)


def test_both_mode_table_shows_only_llm_own_values(monkeypatch):
    from core import llm_engine
    monkeypatch.setattr(llm_engine.LLMEvaluator, 'evaluate_raw',
                        lambda self, ctx: {'trust_score': 0.37, 'action': 'STEP-UP', 'confidence': 0.5, 'reason': 'fake', 'factors': []})
    at = AppTest.from_file(APP, default_timeout=90).run()
    table = _both_mode_run(at, 12)
    # Both mode: a Bayes group and an LLM group (quarantine shows in the status text, e.g. "● Quarantined (until t = 70)")
    assert {'Bayes trust', 'Bayes status', 'Bayes action', 'LLM trust', 'LLM status', 'LLM action',
            'LLM confidence', 'LLM last eval', 'Agree'} <= set(table.columns)
    assert set(table['LLM trust']) <= {'0.37', 'PENDING'} and '0.37' in set(table['LLM trust'])
    assert all(a.startswith(('STEP-UP', 'PENDING', 'BLOCK (quarantined)')) for a in table['LLM action'])
    assert '0.37' not in set(table['Bayes trust'])
    twin = at.session_state.live_twin
    snap_b, snap_l = twin.bayes.controller.snapshot(), twin.llm.controller.snapshot()   # each column = its own world
    assert list(table['Bayes trust']) == [f"{snap_b[d]['trust']:.2f}" for d in table['ID']]
    assert list(table['LLM trust']) == ['PENDING' if snap_l[d]['trust'] is None else f"{snap_l[d]['trust']:.2f}" for d in table['ID']]
    assert twin.mode == 'both' and twin.bayes is not None and twin.llm is not None


def test_both_mode_timeouts_never_show_bayesian_numbers(monkeypatch):
    from core import llm_engine

    def timeout(self, ctx):
        raise llm_engine.LLMUnavailable('LLM request failed: timed out')
    monkeypatch.setattr(llm_engine.LLMEvaluator, 'evaluate_raw', timeout)
    at = AppTest.from_file(APP, default_timeout=90).run()
    table = _both_mode_run(at, 10)
    assert set(table['LLM trust']) <= {'PENDING', 'LLM_ERROR'} and 'LLM_ERROR' in set(table['LLM trust'])
    assert any('LLM_ERROR (timeout)' in a for a in table['LLM action'])
    assert at.session_state.live_twin.llm.controller.errors['timeout'] > 0
