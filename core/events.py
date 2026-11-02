"""What the Zero-Trust controller is allowed to see.

`ObservedEvent` is the ONLY input the live controller gets about a request
(plus per-request sensor alarms). It describes what a network tap could record:
who sent what to whom, on which port, how many bytes, and the ZK proof.

Deliberately absent: role, label, attacker type, "is malicious", ground truth.
Those live only in the simulated environment (core/agents.py, core/live_sim.py)
and are used for sensors and scoring, never handed to the controller.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Optional

from .live_config import EVENT_LIKELIHOODS, NIGHT_START, TICKS_PER_DAY


class EventType(str, Enum):
    """Request kinds seen on the wire, plus anomalies the controller derives from them."""
    # Kinds a device can send (what the request itself is)
    NORMAL_TELEMETRY = 'normal_telemetry'
    AUTHORIZED_SERVICE = 'authorized_service'
    READ_SENSOR = 'read_sensor'
    TOKEN_REQUEST = 'token_request'
    FIRMWARE_UPDATE = 'firmware_update'
    REBOOT = 'reboot'
    # Anomalies the controller derives itself from observable facts
    ZK_AUTH_FAIL = 'zk_auth_fail'
    TOKEN_REPLAY = 'token_replay'
    RATE_SPIKE = 'rate_spike'
    PORT_SCAN = 'port_scan'
    DEVICE_SCAN = 'device_scan'
    CROSS_ZONE_ACCESS = 'cross_zone_access'
    SENSITIVE_TARGET_ACCESS = 'sensitive_target_access'
    NO_TOPOLOGY_EDGE = 'no_topology_edge'
    UNUSUAL_PAYLOAD = 'unusual_payload'
    OFF_HOURS_ACCESS = 'off_hours_access'
    # Outcomes of a step-up challenge
    STEP_UP_PASS = 'step_up_pass'
    STEP_UP_FAIL = 'step_up_fail'


# Kinds an agent may put on the wire (the rest are derived by the controller).
REQUEST_KINDS = frozenset({
    EventType.NORMAL_TELEMETRY, EventType.AUTHORIZED_SERVICE, EventType.READ_SENSOR,
    EventType.TOKEN_REQUEST, EventType.FIRMWARE_UPDATE, EventType.REBOOT,
})

# The controller is the destination for token requests and device-to-controller traffic.
CONTROLLER_ID = 'CONTROLLER'

# Field names that must never appear on anything the controller or LLM receives.
FORBIDDEN_KEYS = frozenset({'role', 'is_attacker', 'ground_truth', 'attack_type', 'malicious',
                            'label', 'attacker', 'suspicious', 'agent', 'secret'})


@dataclass(frozen=True)
class ObservedEvent:
    """One observed request. Contains only facts a network tap could record."""
    tick: int
    src: str
    dst: str
    event_type: EventType
    details: Dict[str, Any] = field(default_factory=dict)  # port, bytes, zone_from, zone_to, sensitivity
    zk_result: Optional[bool] = None                       # filled in by the controller after verifying
    nonce: Optional[str] = None
    proof: Optional[Dict[str, Any]] = None                 # public Schnorr proof: commitment, challenge, response, nonce

    def __post_init__(self) -> None:
        # Guard rail: refuse to carry ground-truth-looking keys even inside `details`.
        leaked = FORBIDDEN_KEYS & set(self.details)
        if leaked:
            raise ValueError(f'ObservedEvent.details must not contain ground-truth keys: {sorted(leaked)}')


def likelihood(event_type: str) -> tuple:
    """(P(event | legitimate), P(event | malicious)) from the single table in live_config."""
    return EVENT_LIKELIHOODS[str(getattr(event_type, 'value', event_type))]


def is_night(tick: int) -> bool:
    """True during the simulated night window (public clock, known to everyone)."""
    return tick % TICKS_PER_DAY >= NIGHT_START
