"""Stage 1 tests: no ground-truth leakage, determinism, trust bounds/recovery, replay, step-up, flooding."""
import ast
import dataclasses
import inspect
import random
from pathlib import Path

import pytest

import core.live_config as C
from core.controller import ISOLATED, SUSPICIOUS, TRUSTED, ZeroTrustController, classify
from core.events import CONTROLLER_ID, FORBIDDEN_KEYS, EventType, ObservedEvent
from core.live_sim import LiveSimulation, render_record
from core.simulator import ZTIoTSimulator
from core.zk_auth import prove, public_key, verify

ROOT = Path(__file__).resolve().parents[1]
NO_ALARMS = {name: False for name in C.SENSORS}
ALL_ALARMS = {name: True for name in C.SENSORS}


def make_live(seed=0, ticks=0):
    sim = ZTIoTSimulator()
    live = LiveSimulation(sim.devices, sim.connections, seed=seed)
    live.run(ticks)
    return live


def solo_controller(secret=37):
    ctl = ZeroTrustController()
    ctl.enroll('D1', public_key(secret), 'Public IoT', 0.25)
    return ctl


def event(tick, nonce, secret=37, kind=EventType.NORMAL_TELEMETRY, dst=CONTROLLER_ID):
    proof = prove(secret, 'D1', nonce, rng=random.Random(tick))
    return ObservedEvent(tick, 'D1', dst, kind, {'port': 1883, 'bytes': 200}, nonce=nonce, proof=proof)


# ---------------- no ground-truth leakage ----------------
def test_observed_event_has_no_ground_truth_fields():
    names = {f.name for f in dataclasses.fields(ObservedEvent)}
    assert names == {'tick', 'src', 'dst', 'event_type', 'details', 'zk_result', 'nonce', 'proof'}
    assert not names & FORBIDDEN_KEYS
    assert {'role', 'is_attacker', 'ground_truth', 'attack_type'} <= FORBIDDEN_KEYS
    with pytest.raises(ValueError):
        ObservedEvent(1, 'D1', CONTROLLER_ID, EventType.NORMAL_TELEMETRY, {'role': 'attacker'})


def test_controller_public_api_accepts_only_observed_data():
    for name, method in inspect.getmembers(ZeroTrustController, inspect.isfunction):
        if name.startswith('_'):
            continue
        params = set(inspect.signature(method).parameters) - {'self'}
        assert not params & FORBIDDEN_KEYS, (name, params)
    assert list(inspect.signature(ZeroTrustController.enroll).parameters) == ['self', 'device_id', 'public_key', 'zone', 'target_sensitivity']
    ctl = solo_controller()
    with pytest.raises(TypeError):
        ctl.process({'src': 'D1', 'role': 'attacker'}, NO_ALARMS)
    with pytest.raises(TypeError):
        ctl.process(event(1, 'n1'), {**NO_ALARMS, 'ground_truth': True})


def test_controller_module_never_imports_environment_code():
    tree = ast.parse((ROOT / 'core' / 'controller.py').read_text())
    imported = {node.module for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)}
    assert not imported & {'agents', 'sensors', 'live_sim', 'simulator'}


def test_simulation_hands_controller_only_observed_events(monkeypatch):
    live = make_live(seed=3)
    seen = []
    original = live.controller.process

    def spy(ev, alarms):
        seen.append((type(ev), set(alarms), set(ev.details)))
        return original(ev, alarms)

    monkeypatch.setattr(live.controller, 'process', spy)
    records = live.run(40)
    assert seen and all(t is ObservedEvent and a == set(C.SENSORS) and not d & FORBIDDEN_KEYS for t, a, d in seen)
    for r in records:
        assert not set(r) & FORBIDDEN_KEYS


def test_classic_attacker_profile_is_not_given_to_controller():
    live = make_live()
    led = live.controller.ledgers['DEV-005']
    assert led.trust == live.controller.ledgers['DEV-001'].trust == C.LIVE_PRIOR  # same prior for everyone
    assert not {f.name for f in dataclasses.fields(led)} & (FORBIDDEN_KEYS | {'name', 'status', 'posture'})


# ---------------- determinism ----------------
def test_same_seed_gives_identical_run():
    a, b = make_live(seed=11, ticks=60), make_live(seed=11, ticks=60)
    assert a.records == b.records and a.trust_history == b.trust_history
    c = make_live(seed=12, ticks=60)
    assert c.records != a.records


def test_seeded_prove_is_reproducible_and_valid():
    p1 = prove(37, 'D1', 'n', rng=random.Random(5))
    assert p1 == prove(37, 'D1', 'n', rng=random.Random(5)) and verify(public_key(37), p1, 'D1')


# ---------------- trust bounds and recovery ----------------
def test_trust_stays_in_bounds():
    live = make_live(seed=4, ticks=120)
    values = [v for row in live.trust_history for k, v in row.items() if k != 'tick']
    assert min(values) >= C.TRUST_MIN and max(values) <= C.TRUST_MAX


def test_trust_recovers_toward_prior_after_bad_events_stop():
    ctl = solo_controller()
    for i in range(3):
        ctl.begin_tick(i + 1)
        ctl.process(event(i + 1, f'bad{i}', kind=EventType.TOKEN_REQUEST), ALL_ALARMS)
    low = ctl.ledgers['D1'].trust
    assert low < C.BLOCK_THRESHOLD
    for t in range(4, 400):
        ctl.begin_tick(t)
    assert abs(ctl.ledgers['D1'].trust - C.LIVE_PRIOR) < 0.02 and ctl.ledgers['D1'].band == TRUSTED


def test_hysteresis_prevents_flicker():
    allow, block, h = C.ALLOW_THRESHOLD, C.BLOCK_THRESHOLD, C.HYSTERESIS
    assert classify(allow - h / 2, TRUSTED, allow, block, h) == TRUSTED      # just below: stays
    assert classify(allow - 2 * h, TRUSTED, allow, block, h) == SUSPICIOUS   # clearly below: moves
    assert classify(allow + h / 2, SUSPICIOUS, allow, block, h) == SUSPICIOUS
    assert classify(block - 2 * h, SUSPICIOUS, allow, block, h) == ISOLATED


# ---------------- ZK, replay, step-up ----------------
def test_replayed_nonce_is_detected_and_blocked():
    ctl = solo_controller()
    ctl.begin_tick(1)
    first = event(1, 'same-nonce')
    assert ctl.process(first, NO_ALARMS)['action'] == 'ALLOW'
    again = ctl.process(first, NO_ALARMS)
    assert again['replay'] and 'token_replay' in again['derived'] and again['action'] == 'BLOCK'


def test_wrong_secret_fails_zk_and_blocks():
    ctl = solo_controller(secret=37)
    ctl.begin_tick(1)
    r = ctl.process(event(1, 'n1', secret=38), NO_ALARMS)
    assert r['zk'] == 'FAIL' and 'zk_auth_fail' in r['derived'] and r['action'] == 'BLOCK'


def test_step_up_needs_fresh_proof_on_challenge_and_mfa():
    cases = [(True, 'challenge', 37, 'ALLOW'), (False, 'challenge', 37, 'BLOCK'),
             (True, 'stale', 37, 'BLOCK'), (True, 'challenge', 38, 'BLOCK')]
    for mfa, nonce_choice, secret, expected in cases:
        ctl = solo_controller()
        ctl.ledgers['D1'].trust, ctl.ledgers['D1'].band = 0.60, SUSPICIOUS
        record = {'src': 'D1', 'action': 'STEP-UP', 'challenge': 'D1-challenge-1-n1'}
        nonce = record['challenge'] if nonce_choice == 'challenge' else 'stale'
        ctl.complete_step_up(record, prove(secret, 'D1', nonce), mfa)
        assert record['action'] == expected, (mfa, nonce_choice, secret)
        moved_up = record['step_up']['trust_after'] > record['step_up']['trust_before']
        assert moved_up == (expected == 'ALLOW')


def test_suspicious_band_maps_to_step_up():
    from core.controller import BAND_ACTION
    assert BAND_ACTION == {TRUSTED: 'ALLOW', SUSPICIOUS: 'STEP-UP', ISOLATED: 'BLOCK'}


# ---------------- behaviour across seeds (Bayesian only) ----------------
def test_token_flood_attacker_blocked_and_honest_devices_allowed_across_seeds():
    blocked, honest_blocks, honest_total = 0, 0, 0
    for seed in range(20):
        live = make_live(seed=seed, ticks=80)
        attack_start = live.ground_truth['DEV-005']['attack_start']
        dev5 = [r for r in live.records if r.get('src') == 'DEV-005']
        assert all(r['action'] == 'ALLOW' for r in dev5 if r['tick'] < attack_start)  # dormant = looks honest
        blocked += any(r['action'] == 'BLOCK' for r in dev5 if r['tick'] >= attack_start)
        honest = [r for r in live.records if r.get('kind') == 'request' and r['src'] != 'DEV-005']
        honest_total += len(honest)
        honest_blocks += sum(r['action'] == 'BLOCK' for r in honest)
    assert blocked >= 18                       # >= 90% of seeds
    assert honest_blocks / honest_total <= 0.05


def test_render_record_shows_calculation():
    live = make_live(seed=7, ticks=31)
    r = next(r for r in live.records if r.get('src') == 'DEV-005' and r['event_type'] == 'token_request' and r['token_count'] > C.TOKEN_FREE_PER_WINDOW)
    text = '\n'.join(render_record(r))
    assert 'Bayes trust' in text and 'token_request×' in text and 'ZK:PASS' in text


# ---------------- Stage 3: quarantine and probation ----------------
def test_isolation_quarantines_then_probation_requires_step_up():
    live = make_live(seed=7, ticks=31)
    led = live.controller.ledgers['DEV-005']
    start = next(r for r in live.records if r.get('quarantined_until'))
    assert start['src'] == 'DEV-005' and led.classification == 'Quarantined'
    end = start['quarantined_until']
    assert end - start['tick'] == C.QUARANTINE_TICKS
    live.run(end - live.tick - 1)
    dropped = [r for r in live.records if r.get('kind') == 'dropped' and r['tick'] > start['tick']]
    assert dropped and all(r['src'] == 'DEV-005' or r['dst'] == 'DEV-005' for r in dropped)
    assert all(r.get('kind') != 'request' or r['src'] != 'DEV-005' for r in live.records if start['tick'] < r['tick'] < end)
    live.step()  # release tick
    assert any('PROBATION' in r.get('text', '') for r in live.records if r['tick'] == end)


def test_probation_device_gets_step_up_then_allow():
    ctl = solo_controller()
    led = ctl.ledgers['D1']
    led.probation, led.trust, led.band = True, 0.90, TRUSTED
    ctl.begin_tick(1)
    r = ctl.process(event(1, 'p1'), NO_ALARMS)
    assert r['action'] == 'STEP-UP' and r['reason'].startswith('on probation')
    ctl.complete_step_up(r, prove(37, 'D1', r['challenge']), True)
    assert r['action'] == 'ALLOW' and not led.probation
    assert ctl.process(event(1, 'p2'), NO_ALARMS)['action'] == 'ALLOW'
