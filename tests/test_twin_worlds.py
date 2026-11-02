"""Step 1: twin worlds share nothing and, with identical controllers, produce identical streams."""
from core.controller import ZeroTrustController
from core.live_sim import simulation_from_scenario, stream_rng
from core.twin import TwinSimulation


def test_stream_rng_is_keyed_and_reproducible():
    a = stream_rng(42, 'DEV-001', 7, 'agent').random()
    assert a == stream_rng(42, 'DEV-001', 7, 'agent').random()
    assert a != stream_rng(42, 'DEV-001', 8, 'agent').random()
    assert a != stream_rng(42, 'DEV-002', 7, 'agent').random()
    assert a != stream_rng(43, 'DEV-001', 7, 'agent').random()


def test_two_bayesian_worlds_are_separate_objects_with_identical_streams():
    """Two worlds with the same (Bayesian) controller type must behave identically and share nothing."""
    twin = TwinSimulation.from_scenario('multi_attacker', seed=3, mode='both', llm_controller_factory=ZeroTrustController)
    a, b = twin.bayes, twin.llm
    assert a is not b and a.controller is not b.controller and a.records is not b.records
    assert a.devices is not b.devices and a.devices['DEV-001'] is not b.devices['DEV-001']
    assert a.agents['DEV-001'] is not b.agents['DEV-001']
    twin.run(120)
    strip = lambda recs: [{k: v for k, v in r.items() if k != 'engine'} for r in recs]  # noqa: E731
    assert strip(a.records) == strip(b.records)
    assert a.controller.ledgers['DEV-001'] is not b.controller.ledgers['DEV-001']
    assert a.controller.ledgers['DEV-001'].seen_nonces is not b.controller.ledgers['DEV-001'].seen_nonces
    a.controller.ledgers['DEV-001'].seen_nonces.add('mutated')
    assert 'mutated' not in b.controller.ledgers['DEV-001'].seen_nonces


def test_single_engine_mode_creates_only_one_world():
    twin = TwinSimulation.from_scenario('normal_day', seed=1, mode='bayesian')
    assert set(twin.worlds) == {'bayesian'} and twin.llm is None


def test_every_record_carries_engine_and_request_id():
    live = simulation_from_scenario('scanning_attack', seed=2)
    live.run(50)
    assert all(r['engine'] == 'bayesian' for r in live.records)
    reqs = [r for r in live.records if r.get('kind') in ('request', 'dropped')]
    assert reqs and len({r['rid'] for r in reqs}) == len(reqs)
    assert set(live.request_truth) == {r['rid'] for r in reqs}
    assert not any('malicious' in r for r in live.records)
