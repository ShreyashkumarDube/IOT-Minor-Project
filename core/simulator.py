from dataclasses import dataclass, asdict, field
from typing import List, Dict
import time
from .zk_auth import public_key, prove, verify
from .bayesian_engine import Evidence, evaluate_trust, policy_decide, SUSPICIOUS_BEHAVIORS

@dataclass
class Device:
    id: str
    name: str
    type: str
    zone: str
    ip: str
    firmware: str
    protocol: str
    power: str
    status: str
    secret: int
    trust: float
    posture: str
    target_sensitivity: float
    role: str = 'normal'
    behaviors: List[str] = field(default_factory=lambda:['normal_telemetry'])

    def public_identity(self): return public_key(self.secret)
    def to_dict(self): return asdict(self)

class ZTIoTSimulator:
    def __init__(self):
        self.devices = self._default_devices()
        self.connections = [('DEV-001','DEV-003'),('DEV-002','DEV-003'),('DEV-003','DEV-004'),('DEV-003','DEV-005')]
        self.events = []
        self.allow_threshold = 0.72
        self.step_up_threshold = 0.88

    def _default_devices(self):
        return [
            Device('DEV-001','Temperature Sensor','Sensor','Public IoT','10.0.1.11','v2.4.1','MQTT','Low','Online',37,.93,'healthy',.25,'normal',['read_sensor','normal_telemetry']),
            Device('DEV-002','Smart Camera','Camera','Public IoT','10.0.1.12','v5.1.0','MQTT','Medium','Online',61,.91,'healthy',.45,'normal',['read_sensor','authorized_service']),
            Device('DEV-003','Gateway Controller','Gateway','Private IoT','10.0.2.10','v3.8.2','HTTPS','High','Online',73,.96,'healthy',.65,'normal',['authorized_service']),
            Device('DEV-004','Industrial Actuator','Actuator','Critical Zone','10.0.3.20','v1.9.7','CoAP','High','Online',89,.94,'healthy',.95,'normal',['authorized_service']),
            Device('DEV-005','Attacker Simulator','Unknown','Untrusted','10.0.9.99','unknown','TCP','Medium','Suspicious',17,.50,'unknown',.90,'attacker',['credential_reuse','rapid_scanning','cross_zone_access','critical_target_probe']),
        ]

    def reset(self):
        self.__init__()

    def add_device(self, device):
        if any(d.id == device.id for d in self.devices): raise ValueError('Device ID already exists')
        self.devices.append(device)

    def remove_device(self, device_id):
        self.devices = [d for d in self.devices if d.id != device_id]
        self.connections = [e for e in self.connections if device_id not in e]

    def set_connection(self, src, dst, enabled):
        edge=(src,dst)
        if enabled and edge not in self.connections: self.connections.append(edge)
        if not enabled and edge in self.connections: self.connections.remove(edge)

    def _evidences(self, suspicious):
        return [
            Evidence('IDS',suspicious,.92,.04),
            Evidence('SIEM',suspicious,.88,.06),
            Evidence('Behavior Monitor',suspicious,.90,.05),
        ]

    def build_context(self, device, target, zk_valid, suspicious, topology_path=None):
        return {
            'device': device.to_dict(),
            'target': target.to_dict() if target else None,
            'zk_valid': zk_valid,
            'suspicious': suspicious,
            'topology_path': topology_path or [],
            'direct_connection': target is not None and (device.id,target.id) in self.connections,
            'policy_thresholds': {'allow_threshold':self.allow_threshold,'step_up_threshold':self.step_up_threshold},
            'security_evidence': [
                {'name':'IDS','alarm':suspicious,'true_positive':.92,'false_positive':.04},
                {'name':'SIEM','alarm':suspicious,'true_positive':.88,'false_positive':.06},
                {'name':'Behavior Monitor','alarm':suspicious,'true_positive':.90,'false_positive':.05},
            ]
        }

    def evaluate_bayesian(self, device, target, zk_valid, suspicious):
        t0=time.perf_counter()
        trust, trace = evaluate_trust(device.trust, device.behaviors, self._evidences(suspicious))
        p = policy_decide(trust, zk_valid, device.posture, target.target_sensitivity if target else device.target_sensitivity, suspicious, self.allow_threshold, self.step_up_threshold)
        p['_latency_ms']=(time.perf_counter()-t0)*1000
        p['trust_trace']=trace
        p['trust_score']=trust
        return p

    def run(self, device_id, target_id=None, method='both', force_bad_zk=False, llm=None):
        device=next(d for d in self.devices if d.id==device_id)
        target=next((d for d in self.devices if d.id==target_id),None) if target_id else None
        nonce=f'{device.id}-{len(self.events)}-{time.time_ns()}'
        proof=prove(device.secret,device.id,nonce)
        zk_valid=verify(device.public_identity(),proof,device.id)
        if force_bad_zk:
            fake=dict(proof); fake['response']=(fake['response']+7)%113
            zk_valid=verify(device.public_identity(),fake,device.id)
        suspicious=device.role.lower() in {'attacker','suspicious'} or device.posture=='compromised' or any(b in SUSPICIOUS_BEHAVIORS for b in device.behaviors)
        bayes=self.evaluate_bayesian(device,target,zk_valid,suspicious) if method in ('both','bayesian') else None
        llm_result=None
        if method in ('both','llm'):
            if llm is None: raise ValueError('LLM evaluator is not configured')
            context=self.build_context(device,target,zk_valid,suspicious)
            llm_result=llm.evaluate(context)
            llm_result['trust_trace']=[]
            llm_result['allowed']=llm_result['action'] in {'ALLOW','STEP-UP'}
        result={'timestamp':time.strftime('%Y-%m-%d %H:%M:%S'),'device_id':device.id,'device':device.name,'target_id':target.id if target else '', 'target':target.name if target else '', 'role':device.role,'zk_valid':zk_valid,'behaviors':device.behaviors,'bayesian':bayes,'llm':llm_result}
        self.events.append(result)
        return result
