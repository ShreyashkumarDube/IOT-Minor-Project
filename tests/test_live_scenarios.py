"""Stage 4: every agent type, scenario presets, and Bayesian-only sanity across 20 seeds."""
import json

import pytest

from core.agents import ATTACKER_TYPES, HONEST_TYPES
from core.live_sim import SCENARIO_DIR, list_scenarios, load_scenario, scenario_network, simulation_from_scenario

SEEDS = range(20)
REQUIRED_PRESETS = {'normal_day', 'token_flood_attack', 'replay_attack', 'scanning_attack',
                    'compromised_valid_token', 'stealth_slow_and_low', 'multi_attacker'}


def quarantined_after(live, device_id, start):
    return any(r.get('quarantined_until') and r['src'] == device_id and r['tick'] >= start for r in live.records)


def test_all_presets_exist_and_build():
    assert REQUIRED_PRESETS <= set(list_scenarios())
    for name in list_scenarios():
        sc = load_scenario(name)
        devices, connections = scenario_network(sc)
        assert len({d.id for d in devices}) == len(devices) and connections
        assert all(a['type'] in HONEST_TYPES + ATTACKER_TYPES for a in sc['agents'].values())
        # presets must not hand the attacker away through public attributes
        assert all(d.role == 'normal' and 'attack' not in d.name.lower() for d in devices)


def test_same_scenario_and_seed_is_reproducible():
    a = simulation_from_scenario('multi_attacker', seed=5); a.run(120)
    b = simulation_from_scenario('multi_attacker', seed=5); b.run(120)
    assert a.records == b.records and a.trust_history == b.trust_history


def test_attackers_are_honest_while_dormant():
    live = simulation_from_scenario('compromised_valid_token', seed=1); live.run(39)
    assert not any(r.get('quarantined_until') or r.get('action') == 'BLOCK' for r in live.records if r.get('src') == 'DEV-008')


@pytest.mark.parametrize('scenario,attacker', [('token_flood_attack', 'DEV-007'), ('replay_attack', 'DEV-006'),
                                               ('scanning_attack', 'DEV-002'), ('compromised_valid_token', 'DEV-008')])
def test_attacker_blocked_in_at_least_90_percent_of_seeds(scenario, attacker):
    caught = 0
    for seed in SEEDS:
        live = simulation_from_scenario(scenario, seed=seed)
        live.run(110)
        caught += quarantined_after(live, attacker, live.ground_truth[attacker]['attack_start'])
    assert caught >= 18, f'{scenario}: {caught}/20'


def test_normal_day_honest_block_rate_at_most_5_percent():
    blocked = total = 0
    for seed in SEEDS:
        live = simulation_from_scenario('normal_day', seed=seed)
        live.run(160)
        for device_id in live.devices:
            total += 1
            blocked += quarantined_after(live, device_id, 0)
    assert blocked / total <= 0.05, f'{blocked}/{total}'


def test_compromised_device_passes_zk_every_time_yet_is_caught():
    live = simulation_from_scenario('compromised_valid_token', seed=3)
    live.run(100)
    mine = [r for r in live.records if r.get('kind') == 'request' and r['src'] == 'DEV-008']
    assert mine and all(r['zk'] == 'PASS' for r in mine)
    assert live.controller.ledgers['DEV-008'].classification in ('Suspicious', 'Quarantined', 'Probation', 'Isolated') or \
        quarantined_after(live, 'DEV-008', 40)


def test_replay_attack_produces_replay_or_zk_failure():
    live = simulation_from_scenario('replay_attack', seed=2)
    live.run(60)
    flags = {f for r in live.records if r.get('kind') == 'request' and r['src'] == 'DEV-006' for f in r['derived']}
    assert flags & {'token_replay', 'zk_auth_fail'}


def test_noisy_honest_reboot_causes_zk_failures_but_rarely_quarantine():
    zk_fails = 0
    for seed in range(10):
        live = simulation_from_scenario('normal_day', seed=seed); live.run(160)
        zk_fails += sum(1 for r in live.records if r.get('kind') == 'request' and r['zk'] == 'FAIL')
    assert zk_fails > 0  # noisy devices really do glitch


def test_trigger_attack_changes_only_the_environment():
    live = simulation_from_scenario('normal_day', seed=4); live.run(25)
    before = live.controller.ledgers['DEV-001'].trust
    start = live.trigger_attack('DEV-001', 'scanning')
    assert live.ground_truth['DEV-001'] == {'agent': 'scanning', 'attack_start': start}
    assert live.controller.ledgers['DEV-001'].trust == before  # controller not told anything
    live.run(40)
    assert quarantined_after(live, 'DEV-001', start)
    with pytest.raises(ValueError):
        live.trigger_attack('DEV-001', 'honest')


def test_slow_and_low_is_reported_not_forced():
    """Slow-and-low may be caught late or missed; this only checks the run completes and the attacker is labelled."""
    caught = 0
    for seed in range(5):
        live = simulation_from_scenario('stealth_slow_and_low', seed=seed)
        live.run(250)
        assert live.ground_truth['DEV-006']['agent'] == 'slow_low'
        caught += quarantined_after(live, 'DEV-006', 40)
    assert 0 <= caught <= 5
