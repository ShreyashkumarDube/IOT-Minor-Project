"""Tick engine for Live mode — one simulated WORLD. Runs headless (tests, batch) or under Streamlit.

A world is the ENVIRONMENT (agents = ground truth, sensors, clock) plus ONE controller
(the Bayesian ZeroTrustController or the LLMController). Each tick it turns every agent
Action into an ObservedEvent (dropping ground truth), asks the sensors for alarms, and
hands both to its controller. Ground truth is kept in `self.ground_truth` and
`self.request_truth` for scoring and the optional examiner view only.

Reproducibility and twin worlds: every random draw comes from a fresh RNG derived from
(seed, device, tick, stream, request index) — see `stream_rng`. There is no global RNG
whose consumption order could differ, so two worlds built from the same scenario and
seed generate exactly the same traffic, sensor alarms, ZK proofs and MFA outcomes. Only
their controllers' decisions (and therefore what gets dropped) can differ.
"""
from __future__ import annotations

import copy
import hashlib
import json
import random
import time
from collections import deque
from pathlib import Path
from typing import Deque, Dict, Iterable, List, Optional, Sequence, Tuple

from . import live_config as C
from .agents import AGENT_TYPES, Action, HonestAgent, make_agent
from .controller import ZeroTrustController
from .events import CONTROLLER_ID, ObservedEvent
from .sensors import SensorSuite
from .simulator import Device, ZTIoTSimulator
from .zk_auth import Q, prove

SCENARIO_DIR = Path(__file__).resolve().parents[1] / 'scenarios'
CAPTURE_PER_DEVICE = 20  # proofs per device an eavesdropper keeps (used only by replay attackers)


def stream_rng(seed: int, device_id: str, tick: int, stream: str, index: int = 0) -> random.Random:
    """A private RNG for one (seed, device, tick, stream, request) — identical in every world."""
    digest = hashlib.sha256(f'{seed}|{device_id}|{tick}|{stream}|{index}'.encode()).digest()
    return random.Random(int.from_bytes(digest[:8], 'big'))


def default_scenario(devices: Sequence[Device]) -> dict:
    """Scenario built from the Classic device list: the old `role` becomes ground truth only."""
    agents = {d.id: {'type': 'token_flood' if d.role == 'attacker' else 'honest'} for d in devices}
    return {'name': 'default (from Classic devices)', 'agents': agents}


class LiveSimulation:
    """One world. Call step() to advance one tick; it returns the new log records."""

    def __init__(self, devices: Sequence[Device], connections: Iterable[Tuple[str, str]],
                 scenario: Optional[dict] = None, seed: int = 0, controller=None):
        # Everything is deep-copied: a world never shares devices, lists or dicts with another world.
        self.devices = {d.id: copy.deepcopy(d) for d in devices}
        self.connections = [tuple(e) for e in connections]
        self.scenario = copy.deepcopy(scenario) if scenario else default_scenario(devices)
        self.scenario_label = self.scenario.get('name', 'custom')
        self.seed = seed
        self.sensors = SensorSuite(random.Random(seed))  # per-request RNGs are passed in; this one is unused
        self.controller = controller or ZeroTrustController()
        self.engine = getattr(self.controller, 'engine', 'bayesian')
        self.controller.set_topology(self.connections)
        self.tick = 0
        self.records: List[dict] = []
        self.trust_history: List[Dict[str, Optional[float]]] = []   # per tick: device -> trust (None = PENDING)
        self.status_history: List[Dict[str, str]] = []              # per tick: device -> classification
        self.request_truth: Dict[str, bool] = {}   # ENVIRONMENT-ONLY: request id -> really malicious?
        self.sensor_draws: Dict[str, Dict[str, dict]] = {}  # ENVIRONMENT-ONLY: request id -> {sensor: {'p', 'u'}}
        self.step_seconds = 0.0                     # wall time spent stepping (for throughput)
        self.eval_ms: Dict[str, float] = {}         # request id -> controller time (kept out of records so logs stay reproducible)
        self.captured: Dict[str, Deque[dict]] = {}  # proofs seen on the wire, per device (eavesdropper's view)

        # Environment-only state: agents and ground truth.
        self.agents: Dict[str, HonestAgent] = {}
        self.ground_truth: Dict[str, dict] = {}
        spec = self.scenario.get('agents', {})
        for device_id in sorted(self.devices):
            device = self.devices[device_id]
            entry = spec.get(device_id, {'type': 'honest'})
            out_links = [dst for src, dst in self.connections if src == device_id and dst in self.devices]
            agent = make_agent(entry.get('type', 'honest'), device, out_links, random.Random(0), entry.get('attack_start'))
            agent.world = self
            self.agents[device_id] = agent
            self.ground_truth[device_id] = {'agent': agent.ground_truth, 'attack_start': agent.attack_start}
            # Enrollment hands over PUBLIC data only (no name, role, status or prior trust).
            self.controller.enroll(device_id, device.public_identity(), device.zone, device.target_sensitivity)
            if hasattr(self.controller, 'register_profile'):   # LLM world: public, non-identifying attributes only
                self.controller.register_profile(device_id, {'type': device.type, 'protocol': device.protocol,
                                                             'firmware': device.firmware})

    # ----- one tick -------------------------------------------------------------
    def step(self, progress=None) -> List[dict]:
        """Advance one tick and return the log records it produced.

        `progress(n)` is called before the controller's end-of-tick work (the LLM world
        evaluates due devices there and the tick waits for them)."""
        t0 = time.perf_counter()
        self.tick += 1
        first = len(self.records)
        for note in self.controller.begin_tick(self.tick):
            self.records.append({'tick': self.tick, 'kind': 'note', 'text': note, 'engine': self.engine})
        for device_id in sorted(self.agents):
            agent = self.agents[device_id]
            agent.rng = stream_rng(self.seed, device_id, self.tick, 'agent')  # same draws in every world
            for index, action in enumerate(agent.actions(self.tick)):
                record = self._handle(device_id, action, index)
                self.records.append(record)
        end_tick = getattr(self.controller, 'end_tick', None)
        if end_tick is not None:
            for r in end_tick(self.tick, progress):
                r['engine'] = self.engine
                self.records.append(r)
        snap = self.controller.snapshot()
        self.trust_history.append({'tick': self.tick, **{d: s['trust'] for d, s in snap.items()}})
        self.status_history.append({'tick': self.tick, **{d: s['status'] for d, s in snap.items()}})
        self.step_seconds += time.perf_counter() - t0
        return self.records[first:]

    def trigger_attack(self, device_id: str, agent_type: str) -> int:
        """Examiner control: turn a device into an attacker of `agent_type` from the next tick.
        Only the environment changes; the controller is not told. Returns the attack start tick."""
        if agent_type not in AGENT_TYPES or agent_type in ('honest', 'noisy_honest'):
            raise ValueError(f'not an attacker type: {agent_type}')
        old = self.agents[device_id]
        start = self.tick + 1
        agent = make_agent(agent_type, self.devices[device_id], old.out_links, random.Random(0), start)
        agent.world = self
        self.agents[device_id] = agent
        self.ground_truth[device_id] = {'agent': agent.ground_truth, 'attack_start': start}
        return start

    def run(self, ticks: int) -> List[dict]:
        """Advance several ticks (headless helper)."""
        out = []
        for _ in range(ticks):
            out.extend(self.step())
        return out

    @property
    def throughput(self) -> Optional[float]:
        """Ticks per second of wall time in this world."""
        return self.tick / self.step_seconds if self.step_seconds else None

    def _handle(self, device_id: str, action: Action, index: int) -> dict:
        """Action (ground truth) -> ObservedEvent (no ground truth) -> controller."""
        device = self.devices[device_id]
        rid = f'{self.tick}-{device_id}-{index}'          # same request id in every world
        rng = lambda stream: stream_rng(self.seed, device_id, self.tick, stream, index)  # noqa: E731
        proof = action.proof
        if proof is None:
            proof = prove(device.secret, device_id, f'{device_id}-{self.tick}-{index}', rng=rng('zk'))
            if action.corrupt_proof:  # e.g. a device mid-reboot sends a damaged response
                proof = {**proof, 'response': (proof['response'] + 1) % Q}
        self.captured.setdefault(device_id, deque(maxlen=CAPTURE_PER_DEVICE)).append(dict(proof))
        target = self.devices.get(action.dst)
        event = ObservedEvent(
            tick=self.tick, src=device_id, dst=action.dst, event_type=action.kind,
            details={'port': action.port, 'bytes': action.bytes, 'zone_from': device.zone,
                     'zone_to': target.zone if target else CONTROLLER_ID,
                     'sensitivity': target.target_sensitivity if target else 0.0},
            nonce=proof.get('nonce'), proof=dict(proof))
        alarms, draws = self.sensors.observe_detailed(action.malicious, rng('sensor'))  # controller only gets booleans
        self.request_truth[rid] = action.malicious
        self.sensor_draws[rid] = draws   # ENVIRONMENT-ONLY: rate used and random draw per sensor
        t0 = time.perf_counter()
        record = self.controller.process(event, alarms)
        self.eval_ms[rid] = (time.perf_counter() - t0) * 1000
        record['engine'], record['rid'] = self.engine, rid
        record['policy_action'] = record['action']   # decision before any step-up outcome
        if record['action'] == 'DROPPED':
            record['kind'] = 'dropped'
            record.pop('kind_hint', None)
            return record
        if record['action'] == 'STEP-UP':
            challenge_proof = prove(device.secret, device_id, record['challenge'], rng=rng('challenge'))
            attacking = self.agents[device_id].is_active(self.tick)
            mfa_ok = rng('mfa').random() < (C.MFA_PASS_MALICIOUS if attacking else C.MFA_PASS_HONEST)
            self.controller.complete_step_up(record, challenge_proof, mfa_ok)
        record['kind'] = 'request'
        return record


# ----- scenarios ------------------------------------------------------------------------
def list_scenarios() -> List[str]:
    """Names of the preset scenario files in scenarios/."""
    return sorted(p.stem for p in SCENARIO_DIR.glob('*.json'))


def load_scenario(name_or_path: str) -> dict:
    """Read a scenario JSON by preset name or path."""
    path = Path(name_or_path)
    if not path.suffix:
        path = SCENARIO_DIR / f'{name_or_path}.json'
    data = json.loads(path.read_text(encoding='utf-8'))
    data.setdefault('name', path.stem)
    return data


def scenario_network(scenario: dict) -> Tuple[List[Device], List[Tuple[str, str]]]:
    """Build the device list and links a scenario describes (default devices and/or catalog types)."""
    from .device_catalog import CATALOG, catalog_defaults, derive_secret
    defaults = {d.id: d for d in ZTIoTSimulator()._default_devices()}
    devices: List[Device] = []
    for spec in scenario['devices']:
        if 'default' in spec:
            devices.append(copy.deepcopy(defaults[spec['default']]))
            continue
        key, device_id = spec['catalog'], spec['id']
        if key not in CATALOG:
            raise ValueError(f'unknown catalog key {key!r}')
        f = catalog_defaults(key, devices)
        devices.append(Device(id=device_id, name=spec.get('name', f['name']), type=f['type'], zone=spec.get('zone', f['zone']),
                              ip=f['ip'], firmware=f['firmware'], protocol=f['protocol'], power=f['power'], status='Online',
                              secret=derive_secret(device_id), trust=f['trust'], posture=f['posture'],
                              target_sensitivity=f['target_sensitivity'], role='normal', behaviors=f['behaviors']))
    ids = {d.id for d in devices}
    connections = [tuple(e) for e in scenario.get('connections', []) if e[0] in ids and e[1] in ids]
    return devices, connections


def simulation_from_scenario(name_or_path: str, seed: Optional[int] = None, **kwargs) -> LiveSimulation:
    """Convenience: load a preset and build a LiveSimulation (seed defaults to the file's seed)."""
    scenario = load_scenario(name_or_path)
    devices, connections = scenario_network(scenario)
    live = LiveSimulation(devices, connections, scenario=scenario, seed=scenario.get('seed', 0) if seed is None else seed, **kwargs)
    return live


# ----- text rendering for the live terminal ------------------------------------
def fmt_trust(x) -> str:
    """Trust for display: PENDING when the LLM has not evaluated the device yet."""
    return 'PENDING' if x is None else f'{x:.2f}'


def render_record(r: dict) -> List[str]:
    """Human-readable lines for one record (Bayesian or LLM world), e.g. for the live terminal."""
    if r.get('kind') == 'note':
        return [f"[t={r['tick']:03d}] {r['text']}"]
    if r.get('kind') == 'llm_eval':
        head = f"[t={r['tick']:03d}] LLM eval({r['src']}) [{r['trigger']}] {r['latency_ms']:.0f} ms: "
        if not r['ok']:
            return [head + f"LLM_ERROR ({r['error']}) — keeping {r['decision_before']}  {r.get('message', '')[:80]}"]
        conf = '—' if r['confidence'] is None else f"{r['confidence']:.2f}"
        lines = [head + f"trust {fmt_trust(r['trust_before'])} → {r['llm_trust']:.2f}, {r['llm_action']}, conf {conf}"
                 f" — \"{r['reason'][:90]}\"",
                 '        factors: ' + ('; '.join(r['factors'][:4]) or '(none returned)')]
        if r.get('quarantined_until'):
            lines.append(f"[t={r['tick']:03d}] {r['src']} QUARANTINED by LLM for {r['quarantined_until'] - r['tick']} ticks — links cut")
        return lines
    if r.get('kind') == 'dropped':
        return [f"[t={r['tick']:03d}] {r['src']} → {r['dst']}  {r['event_type']}  DROPPED — {r['reason']}"]
    count = f" ×{r['token_count']} (window {C.WINDOW_TICKS})" if r['event_type'] == 'token_request' else ''
    alarms = '  '.join(f"{k}:{'alert' if v else '-'}" for k, v in r['alarms'].items())
    head = f"[t={r['tick']:03d}] {r['src']} → {r['dst']}  {r['event_type']}{count}   ZK:{r['zk']}  {alarms}"
    if r['derived']:
        head += '   flags: ' + ', '.join(r['derived'])
    if r.get('engine') == 'llm':
        lines = [head, f"        LLM trust {fmt_trust(r['trust_before'])} · state {r['band_before']}"]
        if r['step_up']:
            s = r['step_up']
            lines.append(f"        STEP-UP challenge → ZK:{s['zk']} MFA:{s['mfa']}  → {r['action']} ({r['reason']})")
        else:
            lines.append(f"        → {r['action']} ({r['reason']})")
        return lines
    terms = ', '.join(f'{label} {delta:+.2f}' for label, delta in r['terms'])
    first_trust = r['step_up']['trust_before'] if r['step_up'] else r['trust_after']
    lines = [head, f"        Bayes trust {r['trust_before']:.2f} → {first_trust:.2f}  (log-odds: {terms})"
                   f"  band: {r['band_before']} → {r.get('band_before_step_up', r['band_after'])}"]
    if r['step_up']:
        s = r['step_up']
        lines.append(f"        STEP-UP challenge → ZK:{s['zk']} MFA:{s['mfa']}  ({s['term'][0]} {s['term'][1]:+.2f})"
                     f"  trust {s['trust_before']:.2f} → {s['trust_after']:.2f}  band → {r['band_after']}  → {r['action']}")
    else:
        lines.append(f"        → {r['action']} ({r['reason']})")
    if r.get('quarantined_until'):
        lines.append(f"[t={r['tick']:03d}] {r['src']} QUARANTINED for {r['quarantined_until'] - r['tick']} ticks — links cut")
    return lines
