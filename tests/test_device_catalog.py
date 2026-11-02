"""Tests for the device catalog, custom-device validation and simulator compatibility."""
import os
import subprocess
import sys
from pathlib import Path

import pytest

from core.bayesian_engine import BEHAVIOR_LIKELIHOODS
from core.device_catalog import (CATALOG, CUSTOM_OPTION, POSTURES, POWER_OPTIONS, PROTOCOLS, SENSITIVITY_LEVELS,
                                 SUSPICIOUS_BEHAVIORS, build_device, catalog_defaults, derive_secret,
                                 next_device_id, validate_device_fields)
from core.simulator import Device, ZTIoTSimulator
from core.zk_auth import Q

ROOT = Path(__file__).resolve().parents[1]
REQUIRED = {'domain', 'label', 'type', 'zone', 'protocol', 'power', 'firmware', 'sensitivity', 'posture', 'behaviors'}


def custom_fields(**overrides):
    f = {'name': 'Lab Sensor', 'type': 'Sensor', 'zone': 'Lab', 'ip': '10.0.4.50', 'firmware': 'v1.0.0',
         'protocol': 'MQTT', 'power': 'Battery', 'posture': 'healthy', 'target_sensitivity': 0.3,
         'trust': 0.8, 'behaviors': ['read_sensor']}
    f.update(overrides)
    return f


def test_catalog_entries_are_complete_and_valid():
    assert 15 <= len(CATALOG) <= 20
    for key, e in CATALOG.items():
        assert REQUIRED <= set(e), key
        assert e['protocol'] in PROTOCOLS and e['power'] in POWER_OPTIONS, key
        assert e['sensitivity'] in SENSITIVITY_LEVELS and e['posture'] in POSTURES, key
        assert e['behaviors'] and set(e['behaviors']) <= set(BEHAVIOR_LIKELIHOODS), key
        assert not set(e['behaviors']) & SUSPICIOUS_BEHAVIORS, f'{key} normal profile must not be suspicious'
        assert 'role' not in e and 'secret' not in e


@pytest.mark.parametrize('key', list(CATALOG))
def test_every_catalog_entry_builds_a_valid_device(key):
    sim = ZTIoTSimulator()
    fields = catalog_defaults(key, sim.devices)
    assert validate_device_fields(fields, sim.devices) == []
    dev = build_device(fields, sim.devices)
    assert isinstance(dev, Device) and dev.id == 'DEV-006' and dev.role == 'normal'
    assert 1 <= dev.secret < Q and dev.secret == derive_secret('DEV-006')


def test_adding_twice_gives_unique_id_name_and_ip():
    sim = ZTIoTSimulator()
    first = build_device(catalog_defaults('smart_lock', sim.devices), sim.devices)
    sim.add_device(first)
    second_fields = catalog_defaults('smart_lock', sim.devices)
    assert second_fields['name'] == 'Smart Lock 2' and second_fields['ip'] != first.ip
    second = build_device(second_fields, sim.devices)
    assert second.id == 'DEV-007' == next_device_id(sim.devices)


@pytest.mark.parametrize('bad_ip', ['999.1.1.1', 'abc', '10.0.0', '10.0.0.1/24'])
def test_validation_rejects_bad_ip(bad_ip):
    errors = validate_device_fields(custom_fields(ip=bad_ip), ZTIoTSimulator().devices)
    assert any('not a valid IP' in e for e in errors)


def test_validation_rejects_duplicates_and_missing_fields():
    devices = ZTIoTSimulator().devices
    dup = validate_device_fields(custom_fields(name='  temperature SENSOR ', ip='10.0.1.11'), devices)
    assert any('already exists' in e for e in dup) and any('already used' in e for e in dup)
    empty = validate_device_fields(catalog_defaults(CUSTOM_OPTION, devices), devices)
    for msg in ['Name is required', 'Ip is required', 'Choose a protocol', 'Choose a power source',
                'Choose a posture', 'at least one behavior']:
        assert any(msg in e for e in empty), msg
    with pytest.raises(ValueError):
        build_device(custom_fields(ip='bad'), devices)
    assert validate_device_fields(custom_fields(), devices) == []


def test_secret_is_stable_across_processes():
    code = 'from core.device_catalog import derive_secret; print(derive_secret("DEV-006"), derive_secret("DEV-123"))'
    outs = set()
    for seed in ('1', '2', '987'):
        env = {**os.environ, 'PYTHONHASHSEED': seed}
        outs.add(subprocess.run([sys.executable, '-c', code], cwd=ROOT, env=env, capture_output=True,
                                text=True, check=True).stdout.strip())
    assert outs == {f'{derive_secret("DEV-006")} {derive_secret("DEV-123")}'}


class FakeLLM:
    """Stand-in for LLMEvaluator so the 'both' path runs without a model server."""
    def __init__(self):
        self.contexts = []

    def evaluate(self, context):
        self.contexts.append(context)
        return {'trust_score': .9, 'risk_score': .1, 'action': 'ALLOW', 'reason': 'stub', 'confidence': .5,
                'required_validation': 'ZK proof', 'factors': [], 'behavior_assessment': [], '_latency_ms': 1.0}


def test_run_with_new_device_through_existing_simulator():
    sim = ZTIoTSimulator()
    dev = build_device(catalog_defaults('wearable_monitor', sim.devices), sim.devices)
    sim.add_device(dev)
    sim.set_connection(dev.id, 'DEV-003', True)
    llm = FakeLLM()
    for method in ('bayesian', 'llm', 'both'):
        r = sim.run(dev.id, 'DEV-003', method, False, llm)
        assert r['device_id'] == dev.id and r['zk_valid'] is True
    assert r['bayesian']['action'] == 'ALLOW' and r['llm']['action'] == 'ALLOW'
    assert llm.contexts[-1]['direct_connection'] is True and llm.contexts[-1]['device']['role'] == 'normal'
    assert sim.run(dev.id, 'DEV-003', 'bayesian', True)['bayesian']['action'] == 'BLOCK'  # invalid ZK still blocks


def test_original_five_devices_unchanged():
    sim = ZTIoTSimulator()
    assert [(d.id, d.name, d.secret, d.role) for d in sim.devices] == [
        ('DEV-001', 'Temperature Sensor', 37, 'normal'), ('DEV-002', 'Smart Camera', 61, 'normal'),
        ('DEV-003', 'Gateway Controller', 73, 'normal'), ('DEV-004', 'Industrial Actuator', 89, 'normal'),
        ('DEV-005', 'Attacker Simulator', 17, 'attacker')]
    actions = {d.id: sim.run(d.id, None, 'bayesian')['bayesian']['action'] for d in sim.devices}
    assert actions == {'DEV-001': 'ALLOW', 'DEV-002': 'ALLOW', 'DEV-003': 'ALLOW', 'DEV-004': 'ALLOW', 'DEV-005': 'BLOCK'}
