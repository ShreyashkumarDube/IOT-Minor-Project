"""Device behaviour generators — the ENVIRONMENT side. Ground truth lives here.

An agent decides what its device sends each tick. Whether a device is honest or
an attacker (and when the attack starts) is known only to the agent and the
simulator; it is used to drive the sensors and to score results afterwards.
The controller never sees an Agent or an Action, only the ObservedEvent built
from an Action.

Agent types: honest, noisy_honest (legitimate but occasionally odd), and five attackers that
start dormant (behaving honestly) and activate at `attack_start`: token_flood, credential_replay,
scanning, compromised_valid, slow_low.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from .events import CONTROLLER_ID, EventType, is_night
from . import live_config as C
from .live_config import DEFAULT_DORMANT_TICKS, FLOOD_BURST, HONEST_REQUESTS_PER_TICK, HONEST_TOKEN_EVERY
from .simulator import Device

# Wire port per protocol (assumption: standard default ports).
PROTOCOL_PORTS = {'MQTT': 1883, 'HTTPS': 443, 'HTTP': 80, 'CoAP': 5683, 'TCP': 9000, 'UDP': 9001,
                  'Zigbee': 8080, 'BLE': 8081, 'Modbus': 502, 'LoRaWAN': 1700}
# Typical payload sizes in bytes per request kind (assumption).
BASE_BYTES = {EventType.NORMAL_TELEMETRY: 200, EventType.READ_SENSOR: 300, EventType.AUTHORIZED_SERVICE: 800,
              EventType.TOKEN_REQUEST: 150, EventType.FIRMWARE_UPDATE: 5000, EventType.REBOOT: 100}
ORDINARY_KINDS = (EventType.NORMAL_TELEMETRY, EventType.AUTHORIZED_SERVICE, EventType.READ_SENSOR)


@dataclass
class Action:
    """One request an agent wants to send. ENVIRONMENT-INTERNAL: `malicious` is ground truth."""
    dst: str
    kind: EventType
    port: int
    bytes: int
    malicious: bool = False
    proof: Optional[dict] = None  # None -> the simulator makes a fresh real ZK proof
    corrupt_proof: bool = False   # True -> the simulator damages the fresh proof (e.g. a device mid-reboot)


class HonestAgent:
    """Ordinary traffic along the device's configured links, at its own baseline rate."""
    ground_truth = 'honest'

    def __init__(self, device: Device, out_links: List[str], rng: random.Random, attack_start: Optional[int] = None):
        self.world = None  # set by the simulator: read-only view of the network (devices, links, captured proofs)
        self.device = device
        self.out_links = sorted(out_links)
        self.rng = rng
        self.attack_start = attack_start
        self.port = PROTOCOL_PORTS.get(device.protocol, 9000)
        kinds = [EventType(b) for b in device.behaviors if b in {k.value for k in ORDINARY_KINDS}]
        self.kinds = kinds or [EventType.NORMAL_TELEMETRY]

    def is_active(self, tick: int) -> bool:
        """Honest devices never attack."""
        return False

    def _request(self, dst: str, kind: EventType, malicious: bool = False, port: Optional[int] = None,
                 size_factor: float = 1.0) -> Action:
        size = int(BASE_BYTES[kind] * size_factor * self.rng.uniform(0.85, 1.15))
        return Action(dst=dst, kind=kind, port=port or self.port, bytes=size, malicious=malicious)

    # Helpers shared by agents that move around the network (read from the simulator's world view).
    def _other_devices(self) -> List[str]:
        return sorted(d for d in self.world.devices if d != self.device.id)

    def _unlinked_other_zone(self) -> List[str]:
        linked = set(self.out_links)
        return [d for d in self._other_devices() if d not in linked and self.world.devices[d].zone != self.device.zone]

    def _sensitive_unlinked(self) -> List[str]:
        linked = set(self.out_links)
        return [d for d in self._other_devices() if d not in linked and self.world.devices[d].target_sensitivity >= C.SENSITIVE_TARGET]

    def honest_actions(self, tick: int) -> List[Action]:
        """Baseline behaviour, shared by honest devices and dormant attackers."""
        actions = []
        destinations = self.out_links + [CONTROLLER_ID]
        night_kinds = [k for k in self.kinds if k != EventType.AUTHORIZED_SERVICE] or [EventType.NORMAL_TELEMETRY]
        for _ in range(self.rng.randint(*HONEST_REQUESTS_PER_TICK)):
            kind = self.rng.choice(night_kinds if is_night(tick) else self.kinds)
            actions.append(self._request(self.rng.choice(destinations), kind))
        if not is_night(tick) and self.rng.random() < 1 / HONEST_TOKEN_EVERY:
            actions.append(self._request(CONTROLLER_ID, EventType.TOKEN_REQUEST))
        return actions

    def actions(self, tick: int) -> List[Action]:
        return self.honest_actions(tick)




class NoisyHonestAgent(HonestAgent):
    """Honest, but sometimes updates firmware, reboots (1-2 failed proofs) or does maintenance access.
    These are the legitimate events that produce false positives."""
    ground_truth = 'noisy_honest'

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pending_glitches = 0  # failed proofs still to come after a reboot

    def actions(self, tick: int) -> List[Action]:
        actions = self.honest_actions(tick)
        if self.pending_glitches:
            for a in actions[:self.pending_glitches]:
                a.corrupt_proof = True
            self.pending_glitches -= min(self.pending_glitches, len(actions))
        roll = self.rng.random()
        if roll < C.NOISY_FIRMWARE_P:
            actions.append(self._request(CONTROLLER_ID, EventType.FIRMWARE_UPDATE))
        elif roll < C.NOISY_FIRMWARE_P + C.NOISY_REBOOT_P:
            actions.append(self._request(CONTROLLER_ID, EventType.REBOOT))
            self.pending_glitches = self.rng.randint(1, 2)
        elif roll < C.NOISY_FIRMWARE_P + C.NOISY_REBOOT_P + C.NOISY_MAINTENANCE_P:
            targets = self._unlinked_other_zone() or self._other_devices()
            if targets:
                actions.append(self._request(self.rng.choice(targets), EventType.AUTHORIZED_SERVICE))
        return actions


class AttackerAgent(HonestAgent):
    """Base for attackers: honest cover traffic, plus attack moves once active."""

    def is_active(self, tick: int) -> bool:
        return self.attack_start is not None and tick >= self.attack_start

    def attack_moves(self, tick: int) -> List[Action]:
        return []

    def actions(self, tick: int) -> List[Action]:
        actions = self.honest_actions(tick)
        if self.is_active(tick):
            actions.extend(self.attack_moves(tick))
        return actions


class TokenFloodAgent(AttackerAgent):
    """Attacker 1: bursts of token requests (which also push its request rate far above baseline)."""
    ground_truth = 'token_flood'

    def attack_moves(self, tick: int) -> List[Action]:
        return [self._request(CONTROLLER_ID, EventType.TOKEN_REQUEST, malicious=True) for _ in range(self.rng.randint(*FLOOD_BURST))]


class CredentialReplayAgent(AttackerAgent):
    """Attacker 2: resends captured proofs — its own old ones (nonce reuse) or other devices' (wrong key)."""
    ground_truth = 'credential_replay'

    def attack_moves(self, tick: int) -> List[Action]:
        moves = []
        for _ in range(self.rng.randint(*C.REPLAY_PER_TICK)):
            own = self.world.captured.get(self.device.id, [])
            others = [p for d, ps in sorted(self.world.captured.items()) if d != self.device.id for p in ps]
            pool = own if (own and (not others or self.rng.random() < 0.5)) else others
            if not pool:
                continue
            action = self._request(CONTROLLER_ID, EventType.TOKEN_REQUEST, malicious=True)
            action.proof = dict(self.rng.choice(pool))
            moves.append(action)
        return moves


class ScanningAgent(AttackerAgent):
    """Attacker 3: probes many destinations and ports, mostly without configured links."""
    ground_truth = 'scanning'

    def attack_moves(self, tick: int) -> List[Action]:
        targets = self._other_devices()
        return [self._request(self.rng.choice(targets), EventType.READ_SENSOR, malicious=True, port=self.rng.choice(C.SCAN_PORTS))
                for _ in range(self.rng.randint(*C.SCAN_PER_TICK))] if targets else []


class CompromisedValidAgent(AttackerAgent):
    """Attacker 4: owns a VALID secret (ZK always passes) but moves laterally, hits sensitive
    targets and exfiltrates large payloads. Shows that ZK alone is not enough."""
    ground_truth = 'compromised_valid'

    def malicious_move(self) -> Optional[Action]:
        options = []
        lateral = self._unlinked_other_zone()
        sensitive = self._sensitive_unlinked()
        if lateral:
            options.append(lambda: self._request(self.rng.choice(lateral), EventType.AUTHORIZED_SERVICE, malicious=True))
        if sensitive:
            options.append(lambda: self._request(self.rng.choice(sensitive), EventType.AUTHORIZED_SERVICE, malicious=True))
        exfil_dst = self.rng.choice(self.out_links + [CONTROLLER_ID])
        options.append(lambda: self._request(exfil_dst, self.rng.choice(self.kinds), malicious=True, size_factor=C.EXFIL_FACTOR))
        return self.rng.choice(options)()

    def attack_moves(self, tick: int) -> List[Action]:
        return [self.malicious_move() for _ in range(self.rng.randint(*C.COMPROMISED_PER_TICK))]


class SlowLowAgent(CompromisedValidAgent):
    """Attacker 5: the same malicious moves as a compromised device, but rarely, hidden in normal traffic."""
    ground_truth = 'slow_low'

    def attack_moves(self, tick: int) -> List[Action]:
        return [self.malicious_move()] if self.rng.random() < C.SLOW_LOW_P else []


AGENT_TYPES: Dict[str, type] = {
    'honest': HonestAgent, 'noisy_honest': NoisyHonestAgent, 'token_flood': TokenFloodAgent,
    'credential_replay': CredentialReplayAgent, 'scanning': ScanningAgent,
    'compromised_valid': CompromisedValidAgent, 'slow_low': SlowLowAgent,
}
HONEST_TYPES = ('honest', 'noisy_honest')
ATTACKER_TYPES = tuple(k for k in AGENT_TYPES if k not in HONEST_TYPES)


def make_agent(agent_type: str, device: Device, out_links: List[str], rng: random.Random,
               attack_start: Optional[int] = None) -> HonestAgent:
    """Build an agent; attackers default to DEFAULT_DORMANT_TICKS of honest behaviour."""
    cls = AGENT_TYPES[agent_type]
    if agent_type not in HONEST_TYPES and attack_start is None:
        attack_start = DEFAULT_DORMANT_TICKS
    return cls(device, out_links, rng, attack_start)
