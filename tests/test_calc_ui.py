"""AppTest for the 🧮 Trust calculation tab: renders in every run mode, examiner-only columns only with Reveal on."""
import sys
from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'experiments'))
from fake_llm import RuleBasedFakeLLM  # noqa: E402

APP = str(ROOT / 'app.py')
EXAMINER_COLUMNS = ('random draw u (examiner)', 'probability used p (examiner)', 'request really malicious? (examiner)')


def _button(at, label):
    return next(b for b in at.button if b.label == label)


def _columns(at):
    cols = set()
    for df in at.dataframe:
        cols |= set(map(str, df.value.columns))
    return cols


@pytest.mark.parametrize('mode', ['bayesian', 'llm', 'both'])
def test_calculation_tab_renders_and_hides_examiner_data(monkeypatch, mode):
    from core import llm_engine
    rule = RuleBasedFakeLLM()
    monkeypatch.setattr(llm_engine.LLMEvaluator, 'evaluate_raw', lambda self, ctx: rule.evaluate(ctx))
    at = AppTest.from_file(APP, default_timeout=180).run()
    at.selectbox(key='live_scenario').set_value('compromised_valid_token').run()
    at.radio(key='live_run_mode').set_value(mode).run()
    at.button(key='live_reset').click().run()
    for _ in range(42):
        at.button(key='live_step').click().run()
    assert not at.exception
    at.selectbox(key='calc_device').set_value('DEV-008').run()
    assert not at.exception
    cols = _columns(at)
    assert not any(c in cols for c in EXAMINER_COLUMNS), 'examiner-only columns shown with Reveal off'
    assert 'fact' in cols if mode != 'bayesian' else 'P(e|legit)' in cols
    at.toggle(key='live_reveal').set_value(True).run()
    assert not at.exception
    cols = _columns(at)
    assert all(c in cols for c in EXAMINER_COLUMNS)
    # pick an older request too
    options = at.selectbox(key='calc_rid').options
    at.selectbox(key='calc_rid').set_value(at.selectbox(key='calc_rid').value).run()
    assert len(options) > 1 and not at.exception
    success = [s.value for s in at.success]
    if mode != 'llm':
        assert any('✔ matches the stored trust' in s for s in success), success
