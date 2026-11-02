"""Catalog of common IoT device types plus helpers to create/validate devices.

Single source of truth for the "Add device from catalog" UI.

MODELLING ASSUMPTIONS: every default below (zone, protocol, power source,
firmware string, target sensitivity, initial trust, normal behavior profile)
is a plausible, hand-picked assumption for a teaching simulator. None of it is
measured data, vendor data or a standard. Justify or adjust these values in the
project report before drawing conclusions from them.
"""
from __future__ import annotations

import hashlib
import ipaddress
import re
from typing import Dict, List, Optional, Sequence

from .bayesian_engine import BEHAVIOR_LIKELIHOODS, SUSPICIOUS_BEHAVIORS
from .simulator import Device
from .zk_auth import Q

# Shown as the last option in the catalog dropdown.
CUSTOM_OPTION = '__custom__'
CUSTOM_LABEL = 'Add other device (custom)'

# Superset of the protocols the original editor offered (MQTT, HTTPS, CoAP, TCP,
# UDP). MQTT stays first because the original editor falls back to index 0.
PROTOCOLS: List[str] = ['MQTT', 'HTTPS', 'CoAP', 'TCP', 'UDP', 'HTTP', 'Zigbee', 'BLE', 'Modbus', 'LoRaWAN']

# Power *source* for catalog/custom devices. The five original devices use
# Low/Medium/High (power *draw*); both kinds are accepted by the editor.
POWER_OPTIONS: List[str] = ['Battery', 'Mains']

POSTURES: List[str] = ['healthy', 'unknown', 'compromised']

# The existing code stores target_sensitivity as a float in [0, 1]
# (defaults use 0.25 ... 0.95). The labelled levels map onto that same scale.
SENSITIVITY_LEVELS: Dict[str, float] = {'low': 0.25, 'medium': 0.50, 'high': 0.75, 'critical': 0.95}


# Subnet prefix per zone, following the IPs of the five original devices.
ZONE_SUBNETS: Dict[str, str] = {
    'Public IoT': '10.0.1', 'Private IoT': '10.0.2', 'Critical Zone': '10.0.3', 'Untrusted': '10.0.9',
}
FALLBACK_SUBNET = '10.0.4'

# Assumption: a newly enrolled device starts slightly below the vetted defaults (0.91-0.96).
DEFAULT_NEW_TRUST = 0.85

# ---------------------------------------------------------------------------
# Catalog (assumed defaults — see module docstring)
# ---------------------------------------------------------------------------
CATALOG: Dict[str, Dict] = {
    # Smart home
    'temperature_sensor': dict(domain='Smart home', label='Temperature Sensor', type='Sensor', zone='Public IoT', protocol='MQTT', power='Battery', firmware='v2.4.0', sensitivity='low', posture='healthy', behaviors=['read_sensor', 'normal_telemetry']),
    'smart_camera': dict(domain='Smart home', label='Smart Camera', type='Camera', zone='Public IoT', protocol='HTTPS', power='Mains', firmware='v5.1.0', sensitivity='medium', posture='healthy', behaviors=['read_sensor', 'authorized_service']),
    'smart_lock': dict(domain='Smart home', label='Smart Lock', type='Lock', zone='Private IoT', protocol='BLE', power='Battery', firmware='v1.3.2', sensitivity='high', posture='healthy', behaviors=['authorized_service', 'normal_telemetry']),
    'smart_thermostat': dict(domain='Smart home', label='Smart Thermostat', type='Thermostat', zone='Public IoT', protocol='MQTT', power='Mains', firmware='v3.0.1', sensitivity='medium', posture='healthy', behaviors=['read_sensor', 'normal_telemetry', 'authorized_service']),
    'smart_speaker': dict(domain='Smart home', label='Smart Speaker', type='Voice Assistant', zone='Public IoT', protocol='HTTPS', power='Mains', firmware='v8.2.0', sensitivity='medium', posture='healthy', behaviors=['authorized_service', 'normal_telemetry']),
    'smart_plug': dict(domain='Smart home', label='Smart Plug', type='Plug', zone='Public IoT', protocol='Zigbee', power='Mains', firmware='v1.1.4', sensitivity='low', posture='healthy', behaviors=['normal_telemetry', 'authorized_service']),
    'door_sensor': dict(domain='Smart home', label='Door Sensor', type='Contact Sensor', zone='Public IoT', protocol='Zigbee', power='Battery', firmware='v1.0.6', sensitivity='low', posture='healthy', behaviors=['read_sensor', 'normal_telemetry']),
    # Industrial / IIoT
    'plc': dict(domain='Industrial / IIoT', label='PLC Controller', type='PLC', zone='Critical Zone', protocol='Modbus', power='Mains', firmware='v4.2.1', sensitivity='critical', posture='healthy', behaviors=['authorized_service', 'normal_telemetry']),
    'industrial_actuator': dict(domain='Industrial / IIoT', label='Industrial Actuator', type='Actuator', zone='Critical Zone', protocol='CoAP', power='Mains', firmware='v1.9.7', sensitivity='critical', posture='healthy', behaviors=['authorized_service']),
    'vibration_sensor': dict(domain='Industrial / IIoT', label='Vibration Sensor', type='Sensor', zone='Private IoT', protocol='MQTT', power='Battery', firmware='v2.0.3', sensitivity='medium', posture='healthy', behaviors=['read_sensor', 'normal_telemetry']),
    # Healthcare
    'wearable_monitor': dict(domain='Healthcare', label='Wearable Health Monitor', type='Wearable', zone='Private IoT', protocol='BLE', power='Battery', firmware='v2.7.0', sensitivity='high', posture='healthy', behaviors=['read_sensor', 'normal_telemetry']),
    'infusion_pump': dict(domain='Healthcare', label='Infusion Pump', type='Medical Device', zone='Critical Zone', protocol='HTTPS', power='Mains', firmware='v6.0.2', sensitivity='critical', posture='healthy', behaviors=['authorized_service', 'normal_telemetry']),
    # Smart city / energy
    'smart_meter': dict(domain='Smart city / energy', label='Smart Meter', type='Meter', zone='Private IoT', protocol='LoRaWAN', power='Mains', firmware='v3.4.0', sensitivity='high', posture='healthy', behaviors=['read_sensor', 'normal_telemetry']),
    'traffic_sensor': dict(domain='Smart city / energy', label='Traffic Sensor', type='Sensor', zone='Public IoT', protocol='LoRaWAN', power='Mains', firmware='v1.6.0', sensitivity='medium', posture='healthy', behaviors=['read_sensor', 'normal_telemetry']),
    'smart_streetlight': dict(domain='Smart city / energy', label='Smart Streetlight', type='Streetlight', zone='Public IoT', protocol='CoAP', power='Mains', firmware='v2.2.5', sensitivity='medium', posture='healthy', behaviors=['normal_telemetry', 'authorized_service']),
    'drone': dict(domain='Smart city / energy', label='Survey Drone', type='Drone', zone='Public IoT', protocol='MQTT', power='Battery', firmware='v4.0.0', sensitivity='high', posture='healthy', behaviors=['normal_telemetry', 'authorized_service']),
    # Gateways / controllers
    'edge_gateway': dict(domain='Gateways / controllers', label='Edge Gateway', type='Gateway', zone='Private IoT', protocol='HTTPS', power='Mains', firmware='v3.8.2', sensitivity='high', posture='healthy', behaviors=['authorized_service']),
    'smart_home_hub': dict(domain='Gateways / controllers', label='Smart Home Hub', type='Hub', zone='Private IoT', protocol='Zigbee', power='Mains', firmware='v2.9.1', sensitivity='high', posture='healthy', behaviors=['authorized_service', 'normal_telemetry']),
}

REQUIRED_TEXT_FIELDS = ('name', 'type', 'zone', 'ip', 'firmware')


def option_label(key: str) -> str:
    """Readable dropdown label for a catalog key (grouped by domain prefix)."""
    if key == CUSTOM_OPTION:
        return CUSTOM_LABEL
    entry = CATALOG[key]
    return f"{entry['domain']} — {entry['label']}"


def device_label(device: Device) -> str:
    """Readable label 'name · type · id' for source/target selectors."""
    return f'{device.name} · {device.type} · {device.id}'


def derive_secret(device_id: str) -> int:
    """Stable ZK secret in [1, Q-1] derived from SHA-256 of the id (never Python's hash())."""
    digest = hashlib.sha256(device_id.encode('utf-8')).digest()
    return int.from_bytes(digest, 'big') % (Q - 1) + 1


def next_device_id(devices: Sequence[Device]) -> str:
    """Next free id continuing the DEV-xxx numbering."""
    ids = {d.id for d in devices}
    nums = [int(m.group(1)) for d in devices if (m := re.fullmatch(r'DEV-(\d+)', d.id))]
    n = max(nums, default=0) + 1
    while f'DEV-{n:03d}' in ids:
        n += 1
    return f'DEV-{n:03d}'


def unique_name(base: str, devices: Sequence[Device]) -> str:
    """Return base, or 'base 2', 'base 3', ... if the name is taken (case-insensitive)."""
    taken = {d.name.strip().lower() for d in devices}
    name, i = base, 2
    while name.lower() in taken:
        name, i = f'{base} {i}', i + 1
    return name


def next_free_ip(zone: str, devices: Sequence[Device]) -> str:
    """First unused address (from .30) in the subnet assumed for this zone."""
    prefix = ZONE_SUBNETS.get(zone, FALLBACK_SUBNET)
    used = {d.ip for d in devices}
    for host in range(30, 255):
        ip = f'{prefix}.{host}'
        if ip not in used:
            return ip
    return ''


def catalog_defaults(key: str, devices: Sequence[Device]) -> Dict:
    """Form defaults for a catalog entry, or an empty form for the custom option."""
    if key == CUSTOM_OPTION:
        return {'name': '', 'type': '', 'zone': '', 'ip': '', 'firmware': '', 'protocol': None,
                'power': None, 'posture': None, 'target_sensitivity': 0.5, 'trust': 0.8, 'behaviors': []}
    e = CATALOG[key]
    return {
        'name': unique_name(e['label'], devices), 'type': e['type'], 'zone': e['zone'],
        'ip': next_free_ip(e['zone'], devices), 'firmware': e['firmware'], 'protocol': e['protocol'],
        'power': e['power'], 'posture': e['posture'],
        'target_sensitivity': SENSITIVITY_LEVELS[e['sensitivity']], 'trust': DEFAULT_NEW_TRUST,
        'behaviors': list(e['behaviors']),
    }


def validate_device_fields(fields: Dict, devices: Sequence[Device]) -> List[str]:
    """Return a list of human-readable errors (empty list means valid)."""
    errors: List[str] = []
    for f in REQUIRED_TEXT_FIELDS:
        if not str(fields.get(f) or '').strip():
            errors.append(f'{f.capitalize()} is required.')
    ip = str(fields.get('ip') or '').strip()
    if ip:
        try:
            ipaddress.ip_address(ip)
        except ValueError:
            errors.append(f'"{ip}" is not a valid IP address.')
    if fields.get('protocol') not in PROTOCOLS:
        errors.append('Choose a protocol.')
    if fields.get('power') not in POWER_OPTIONS:
        errors.append('Choose a power source.')
    if fields.get('posture') not in POSTURES:
        errors.append('Choose a posture.')
    for f in ('target_sensitivity', 'trust'):
        try:
            if not 0.0 <= float(fields.get(f)) <= 1.0:
                errors.append(f'{f} must be between 0 and 1.')
        except (TypeError, ValueError):
            errors.append(f'{f} must be a number between 0 and 1.')
    behaviors = fields.get('behaviors') or []
    if not behaviors:
        errors.append('Select at least one behavior.')
    unknown = [b for b in behaviors if b not in BEHAVIOR_LIKELIHOODS]
    if unknown:
        errors.append(f'Unknown behaviors: {", ".join(unknown)}.')
    name = str(fields.get('name') or '').strip().lower()
    if name and any(d.name.strip().lower() == name for d in devices):
        errors.append(f'A device named "{fields["name"].strip()}" already exists.')
    if ip and any(d.ip == ip for d in devices):
        errors.append(f'IP {ip} is already used by another device.')
    return errors


def build_device(fields: Dict, devices: Sequence[Device]) -> Device:
    """Create a validated Device with the next DEV-xxx id, role 'normal' and a stable secret."""
    errors = validate_device_fields(fields, devices)
    if errors:
        raise ValueError(' '.join(errors))
    device_id = next_device_id(devices)
    return Device(
        id=device_id, name=fields['name'].strip(), type=fields['type'].strip(), zone=fields['zone'].strip(),
        ip=fields['ip'].strip(), firmware=fields['firmware'].strip(), protocol=fields['protocol'],
        power=fields['power'], status='Online', secret=derive_secret(device_id),
        trust=float(fields['trust']), posture=fields['posture'],
        target_sensitivity=float(fields['target_sensitivity']), role='normal',
        behaviors=list(fields['behaviors']),
    )


def validate_editor_fields(device_id: str, name: str, ip: str, devices: Sequence[Device],
                           editing_id: Optional[str] = None) -> List[str]:
    """Checks for the original 'Add / edit device' editor: required id/name, valid IP, no duplicates.

    `editing_id` is the device being edited (it may keep its own id, name and IP).
    """
    errors: List[str] = []
    others = [d for d in devices if d.id != editing_id]
    device_id, name, ip = (device_id or '').strip(), (name or '').strip(), (ip or '').strip()
    if not device_id:
        errors.append('Device ID is required.')
    elif any(d.id == device_id for d in others):
        errors.append(f'Device ID {device_id} already exists.')
    if not name:
        errors.append('Name is required.')
    elif any(d.name.strip().lower() == name.lower() for d in others):
        errors.append(f'A device named "{name}" already exists.')
    try:
        ipaddress.ip_address(ip)
        if any(d.ip == ip for d in others):
            errors.append(f'IP {ip} is already used by another device.')
    except ValueError:
        errors.append(f'"{ip}" is not a valid IP address.')
    return errors
