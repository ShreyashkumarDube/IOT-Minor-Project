"""Streamlit AppTest: app loads, catalog/custom add works, a run with the new device completes."""
from pathlib import Path

from streamlit.testing.v1 import AppTest

APP = str(Path(__file__).resolve().parents[1] / 'app.py')


def _by_label(widgets, label):
    return next(w for w in widgets if w.label == label)


def test_app_catalog_add_and_run():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.button_group(key='app_mode').set_value('Classic evaluator').run()  # Live mode is now the default
    assert not at.exception

    # Catalog add: pick a type, form auto-fills, submit.
    at.selectbox(key='catalog_choice').set_value('smart_lock').run()
    assert at.text_input(key='cat_smart_lock_0_name').value == 'Smart Lock'
    _by_label(at.button, 'Add device').click().run()
    assert not at.exception
    sim = at.session_state.sim
    assert sim.devices[-1].id == 'DEV-006' and sim.devices[-1].name == 'Smart Lock'
    source = _by_label(at.sidebar.selectbox, 'Source device')
    assert 'DEV-006' in source.options[-1] or 'Smart Lock · Lock · DEV-006' in source.options

    # Custom add with a bad IP shows errors and adds nothing.
    at.selectbox(key='catalog_choice').set_value('__custom__').run()
    n = at.session_state.catalog_nonce
    k = f'cat___custom___{n}_'
    for field, value in [('name', 'Lab Box'), ('type', 'Sensor'), ('zone', 'Lab'), ('ip', '300.1.1.1'), ('fw', 'v0.1')]:
        at.text_input(key=k + field).set_value(value)
    at.selectbox(key=k + 'proto').set_value('CoAP')
    at.selectbox(key=k + 'power').set_value('Mains')
    at.selectbox(key=k + 'posture').set_value('healthy')
    at.multiselect(key=k + 'beh').set_value(['read_sensor'])
    _by_label(at.button, 'Add device').click().run()
    assert not at.exception and any('not a valid IP' in e.value for e in at.error)
    assert len(at.session_state.sim.devices) == 6

    # Run Bayesian evaluation from the new device to the gateway.
    _by_label(at.sidebar.selectbox, 'Source device').set_value('DEV-006').run()
    _by_label(at.sidebar.selectbox, 'Target device/resource').set_value('DEV-003').run()
    _by_label(at.sidebar.radio, 'Run').set_value('Bayesian only').run()
    at.button(key='classic_run').click().run()
    assert not at.exception
    last = at.session_state.last
    assert last['device_id'] == 'DEV-006' and last['bayesian']['action'] == 'ALLOW'

    # Reset survives and restores the five defaults.
    at.button(key='classic_reset').click().run()
    assert not at.exception and len(at.session_state.sim.devices) == 5
