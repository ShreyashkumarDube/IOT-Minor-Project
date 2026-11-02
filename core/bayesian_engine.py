"""Deterministic Bayesian trust evaluation and policy engine."""
from dataclasses import dataclass
from typing import List

BEHAVIOR_LIKELIHOODS = {
    'normal_telemetry': 0.98,
    'authorized_service': 0.97,
    'read_sensor': 0.96,
    'firmware_update': 0.90,
    'cross_zone_access': 0.62,
    'rapid_scanning': 0.28,
    'credential_reuse': 0.18,
    'critical_target_probe': 0.08,
    'unexpected_port': 0.22,
    'data_exfiltration': 0.05,
}

# Behaviors that mark a request as suspicious in Classic mode (single source of truth;
# app.py, simulator.py and device_catalog.py import this set).
SUSPICIOUS_BEHAVIORS = frozenset({
    'cross_zone_access', 'rapid_scanning', 'credential_reuse',
    'critical_target_probe', 'unexpected_port', 'data_exfiltration',
})

@dataclass
class Evidence:
    name: str
    suspicious: bool
    true_positive: float
    false_positive: float

def bayes_evidence(trust: float, e: Evidence) -> float:
    p_legit = e.false_positive if e.suspicious else 1-e.false_positive
    p_att = e.true_positive if e.suspicious else 1-e.true_positive
    num = p_legit * trust
    den = num + p_att * (1-trust)
    return trust if den == 0 else max(0.0, min(1.0, num/den))

def behavior_update(trust: float, behavior: str) -> float:
    score = BEHAVIOR_LIKELIHOODS.get(behavior, 0.5)
    p_legit = max(0.01, score)
    p_att = max(0.01, 1-score)
    num = p_legit * trust
    den = num + p_att * (1-trust)
    return trust if den == 0 else max(0.0, min(1.0, num/den))

def evaluate_trust(initial_trust: float, behaviors: List[str], evidences: List[Evidence]):
    trust = max(0.0, min(1.0, initial_trust))
    trace = [{'stage':'initial','trust':trust}]
    for b in behaviors:
        trust = behavior_update(trust, b)
        trace.append({'stage':f'behavior:{b}','trust':trust})
    for e in evidences:
        trust = bayes_evidence(trust, e)
        trace.append({'stage':f'evidence:{e.name}','trust':trust})
    return trust, trace

def policy_decide(trust, zk_valid, posture, sensitivity, suspicious, allow_threshold=0.72, step_up_threshold=0.88):
    risk = round(1-trust, 4)
    if posture == 'compromised':
        return {'action':'BLOCK','allowed':False,'reason':'Device posture is compromised.','required_validation':'ZK proof + quarantine','risk':risk}
    if not zk_valid:
        return {'action':'BLOCK','allowed':False,'reason':'Zero-knowledge proof failed.','required_validation':'Valid ZK proof required','risk':risk}
    if suspicious and trust < step_up_threshold:
        if trust >= allow_threshold and sensitivity < 0.80:
            return {'action':'STEP-UP','allowed':True,'reason':'Suspicious behavior observed; trust is sufficient for step-up on this target.','required_validation':'ZK proof + MFA simulation','risk':risk}
        return {'action':'BLOCK','allowed':False,'reason':'Suspicious behavior lowered trust below the policy threshold.','required_validation':'ZK proof + MFA simulation','risk':risk}
    if trust >= allow_threshold:
        return {'action':'ALLOW','allowed':True,'reason':'Trust, posture and ZK proof satisfy policy.','required_validation':'ZK proof','risk':risk}
    return {'action':'BLOCK','allowed':False,'reason':'Trust score is below the minimum access threshold.','required_validation':'ZK proof + MFA simulation','risk':risk}
