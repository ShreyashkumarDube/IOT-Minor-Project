"""Bayesian vs LLM comparison layer: reads both worlds (and ground truth), writes to neither.

Every parameter is computed with the SAME definition for both worlds, from each world's own
log. Ground truth is used only here, after the run. Definitions are written next to each
computation so they can be quoted in the report.

Key definitions
* Detected            = the world quarantined the attacker at or after its attack start.
* Time-to-STEP-UP/BLOCK = ticks from attack start to the first flag (challenge, Suspicious/Isolated
                        state or LLM STEP-UP/BLOCK verdict) / first quarantine, in that world.
* Per-request truth   = whether the SENDING DEVICE was an active attacker at that tick (blocking an
                        active attacker's cover traffic is correct; blocking honest traffic is not).
* Policy action       = the world's decision for a request before any step-up outcome
                        (ALLOW / STEP-UP / BLOCK; DROPPED by quarantine counts as BLOCK).
* Agreement           = both worlds took the same policy action on the same request id.
* Trust "legit" label = 1 for honest devices and for attackers before their attack starts, else 0.
"""
from __future__ import annotations

import csv
import io
import math
import os
from collections import Counter, defaultdict
from statistics import mean, median, pstdev
from typing import Callable, Dict, List, Optional, Sequence

from . import live_config as C
from .agents import HONEST_TYPES
from .metrics import compute_metrics, device_timeline, log_rows, to_csv

ENGINES = ('bayesian', 'llm')
LABEL = {'bayesian': 'Bayesian', 'llm': 'LLM'}

# Words in LLM factors/reasons -> the observed fact they refer to (heuristic, see factor_overlap()).
FACT_KEYWORDS = {
    'token': 'token_burst', 'flood': 'token_burst',
    'scan': 'scanning', 'port': 'scanning', 'destination': 'scanning',
    'replay': 'nonce_reuse', 'nonce': 'nonce_reuse',
    'zk': 'zk_failure', 'proof': 'zk_failure', 'authentication': 'zk_failure',
    'payload': 'large_payload', 'exfil': 'large_payload', 'bytes': 'large_payload',
    'zone': 'zone_change', 'lateral': 'zone_change',
    'link': 'unlinked', 'topology': 'unlinked', 'edge': 'unlinked',
    'sensitive': 'sensitive_target',
    'night': 'night', 'off-hours': 'night', 'off hours': 'night',
    'alarm': 'sensor_alarm', 'ids': 'sensor_alarm', 'siem': 'sensor_alarm', 'sensor': 'sensor_alarm',
    'rate': 'rate_spike', 'spike': 'rate_spike',
    'step-up': 'step_up', 'mfa': 'step_up',
}


# ----- small helpers -------------------------------------------------------------------------------------
def _mean(xs) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return mean(xs) if xs else None


def _pct(values: Sequence[float], q: float) -> Optional[float]:
    xs = sorted(values)
    if not xs:
        return None
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def _ratio(a, b) -> Optional[float]:
    return a / b if b else None


def is_active(world, device_id: str, tick: int) -> bool:
    gt = world.ground_truth[device_id]
    return gt['agent'] not in HONEST_TYPES and gt['attack_start'] is not None and tick >= gt['attack_start']


def request_records(world) -> List[dict]:
    return [r for r in world.records if r.get('kind') in ('request', 'dropped')]


def policy_of(r: dict) -> str:
    """Decision before any step-up outcome; a quarantine drop counts as BLOCK."""
    a = r.get('policy_action', r['action'])
    return 'BLOCK' if a == 'DROPPED' else a


# ----- one world --------------------------------------------------------------------------------------------
def world_stats(world) -> dict:
    """Every per-world number in the report, computed from this world's own log only."""
    m = compute_metrics(world)
    engine = world.engine
    honest_rows = [h for h in m['honest'] if h['truth'] == 'honest']
    noisy_rows = [h for h in m['honest'] if h['truth'] == 'noisy_honest']
    tp, fp, tn, fn = (m['confusion'][k] for k in ('TP', 'FP', 'TN', 'FN'))
    total = tp + fp + tn + fn

    # Wrongful quarantines (honest/noisy devices) and their duration, clipped to the run length.
    wrongful, wrongful_ticks = 0, 0
    for r in world.records:
        if r.get('quarantined_until') and world.ground_truth[r['src']]['agent'] in HONEST_TYPES:
            wrongful += 1
            wrongful_ticks += min(r['quarantined_until'], world.tick) - r['tick']

    # Trust behaviour from the per-tick history.
    hist, status = world.trust_history, world.status_history
    honest_vals, attacker_vals, brier_terms, pending = [], [], [], 0
    reliability = defaultdict(lambda: [0, 0.0, 0])   # bin -> [n, sum trust, n legit]
    volatility, flips, final = {}, {}, {}
    for d in world.devices:
        series = [(h['tick'], h.get(d)) for h in hist]
        diffs = [b - a for (_, a), (_, b) in zip(series, series[1:]) if a is not None and b is not None]
        volatility[d] = pstdev(diffs) if len(diffs) > 1 else 0.0
        states = [h.get(d) for h in status]
        flips[d] = sum(1 for a, b in zip(states, states[1:]) if a != b)
        final[d] = series[-1][1] if series else None
        for tick, trust in series:
            if trust is None:
                pending += 1
                continue
            legit = 0 if is_active(world, d, tick) else 1
            if world.ground_truth[d]['agent'] in HONEST_TYPES:
                honest_vals.append(trust)          # honest and noisy-honest devices, all ticks
            elif not legit:
                attacker_vals.append(trust)        # attackers only while their attack is active
            brier_terms.append((trust - legit) ** 2)
            b = min(int(trust * 5), 4)
            reliability[b][0] += 1
            reliability[b][1] += trust
            reliability[b][2] += legit
    honest_mean = _mean(honest_vals)
    attacker_mean = _mean(attacker_vals)

    # Recovery of noisy-honest devices: ticks from leaving Trusted until Trusted again.
    recoveries, unrecovered = [], 0
    for h in noisy_rows:
        d, start = h['device'], None
        for row in status:
            s = row.get(d)
            if start is None and s not in ('Trusted', 'Pending'):
                start = row['tick']
            elif start is not None and s == 'Trusted':
                recoveries.append(row['tick'] - start)
                start = None
        unrecovered += start is not None

    # Per-request decisions vs per-request truth.
    reqs = request_records(world)
    confusion = Counter((is_active(world, r['src'], r['tick']), policy_of(r)) for r in reqs)
    correct = sum(n for (mal, act), n in confusion.items() if (act != 'ALLOW') == mal)
    step_ups = [r for r in reqs if r.get('step_up')]

    # Cost: latency per evaluation.
    if engine == 'llm':
        evals = [r for r in world.records if r.get('kind') == 'llm_eval']
        latencies = [r['latency_ms'] for r in evals]
        ok = [r for r in evals if r['ok']]
        conf_correct = [r['confidence'] for r in ok if r['confidence'] is not None and
                        ((r['llm_action'] != 'ALLOW') == is_active(world, r['src'], r['tick']))]
        conf_wrong = [r['confidence'] for r in ok if r['confidence'] is not None and
                      ((r['llm_action'] != 'ALLOW') != is_active(world, r['src'], r['tick']))]
        errors = dict(world.controller.errors)
        clamped = world.controller.clamped
        confidence = _mean(r['confidence'] for r in ok)
    else:
        evals, ok = [], []
        latencies = list(world.eval_ms.values())
        conf_correct = conf_wrong = []
        errors, clamped, confidence = {}, 0, None

    return {
        'engine': engine, 'metrics': m, 'tp': tp, 'fp': fp, 'tn': tn, 'fn': fn,
        'accuracy': _ratio(tp + tn, total), 'specificity': _ratio(tn, tn + fp),
        'fpr_block_honest': _mean([h['quarantined'] for h in honest_rows]) if honest_rows else None,
        'fpr_block_noisy': _mean([h['quarantined'] for h in noisy_rows]) if noisy_rows else None,
        'fpr_stepup_honest': _mean([h['any_step_up'] for h in honest_rows]) if honest_rows else None,
        'fpr_stepup_noisy': _mean([h['any_step_up'] for h in noisy_rows]) if noisy_rows else None,
        'wrongful_quarantines': wrongful, 'wrongful_quarantine_ticks': wrongful_ticks,
        'final_trust': final, 'honest_mean_trust': honest_mean, 'attacker_mean_trust': attacker_mean,
        'separation_gap': None if honest_mean is None or attacker_mean is None else honest_mean - attacker_mean,
        'volatility': volatility, 'mean_volatility': _mean(volatility.values()),
        'flips': flips, 'mean_flips': _mean(flips.values()),
        'recovery_ticks': recoveries, 'mean_recovery': _mean(recoveries), 'unrecovered_episodes': unrecovered,
        'brier': _mean(brier_terms), 'pending_ticks': pending,
        'reliability': [{'trust bin': f'{b / 5:.1f}–{(b + 1) / 5:.1f}', 'n': n, 'mean trust': s / n, 'observed legit rate': legit / n}
                        for b, (n, s, legit) in sorted(reliability.items()) if n],
        'request_confusion': {f"{'attacker' if mal else 'legitimate'} request → {act}": n for (mal, act), n in sorted(confusion.items())},
        'request_accuracy': _ratio(correct, len(reqs)),
        'malicious_allowed': sum(n for (mal, act), n in confusion.items() if mal and act == 'ALLOW'),
        'benign_blocked': sum(n for (mal, act), n in confusion.items() if not mal and act == 'BLOCK'),
        'step_ups_issued': len(step_ups), 'step_ups_passed': sum(r['action'] == 'ALLOW' for r in step_ups),
        'step_ups_failed': sum(r['action'] == 'BLOCK' for r in step_ups),
        'evaluations': len(evals) if engine == 'llm' else len([r for r in reqs if r['kind'] == 'request']),
        'latencies': latencies, 'latency_mean': _mean(latencies), 'latency_median': median(latencies) if latencies else None,
        'latency_p95': _pct(latencies, 0.95), 'latency_max': max(latencies) if latencies else None,
        'errors': errors, 'clamped': clamped, 'confidence_mean': confidence,
        'confidence_correct': _mean(conf_correct), 'confidence_incorrect': _mean(conf_wrong),
        'throughput': world.throughput,
    }


# ----- across worlds ------------------------------------------------------------------------------------------
def agreement(b, l) -> dict:
    """Same request id, same policy action? Overall, per device, per tick, and the disagreement cases."""
    lb = {r['rid']: r for r in request_records(b)}
    ll = {r['rid']: r for r in request_records(l)}
    common = sorted(set(lb) & set(ll), key=lambda rid: (lb[rid]['tick'], rid))
    per_device, per_tick, cases = defaultdict(list), defaultdict(list), []
    for rid in common:
        rb, rl = lb[rid], ll[rid]
        same = policy_of(rb) == policy_of(rl)
        per_device[rb['src']].append(same)
        per_tick[rb['tick']].append(same)
        if not same:
            last_eval = next((e for e in reversed(l.records) if e.get('kind') == 'llm_eval' and e['src'] == rb['src']
                              and e['tick'] < rb['tick'] and e['ok']), None)
            cases.append({'tick': rb['tick'], 'device': rb['src'], 'request_id': rid,
                          'event': f"{rb['event_type']} → {rb['dst']} (ZK {rb.get('zk', '—')}, port {rb.get('port')}, {rb.get('bytes')} B)",
                          'bayes_decision': policy_of(rb), 'bayes_reason': rb['reason'],
                          'llm_decision': policy_of(rl), 'llm_reason': rl['reason'],
                          'llm_last_eval_reason': last_eval['reason'] if last_eval else 'PENDING (no evaluation yet)',
                          'sender_active_attacker': is_active(b, rb['src'], rb['tick']),
                          'request_generated_maliciously': b.request_truth[rid]})
    return {'rate': _mean([x for xs in per_device.values() for x in xs]),
            'per_device': {d: mean(v) for d, v in per_device.items()},
            'per_tick': [{'tick': t, 'agreement': mean(v)} for t, v in sorted(per_tick.items())],
            'cases': cases, 'requests_compared': len(common)}


def attackers_side_by_side(sb: dict, sl: dict) -> List[dict]:
    lm = {a['device']: a for a in sl['metrics']['attackers']}
    rows = []
    for a in sb['metrics']['attackers']:
        x = lm.get(a['device'], {})
        bt, lt = a['ttd_block'], x.get('ttd_block')
        if bt is None and lt is None:
            first, margin = 'neither', None
        elif lt is None or (bt is not None and bt < lt):
            first, margin = 'Bayesian', None if lt is None else lt - bt
        elif bt is None or lt < bt:
            first, margin = 'LLM', None if bt is None else bt - lt
        else:
            first, margin = 'tie', 0
        rows.append({'device': a['device'], 'attack_type': a['truth'], 'attack_start': a['attack_start'],
                     'bayes_detected': a['detected'], 'llm_detected': x.get('detected'),
                     'bayes_ticks_to_step_up': a['ttd_step_up'], 'llm_ticks_to_step_up': x.get('ttd_step_up'),
                     'bayes_ticks_to_block': bt, 'llm_ticks_to_block': lt,
                     'first_to_block': first, 'by_ticks': margin})
    return rows


def by_attack_type(attackers: List[dict]) -> List[dict]:
    out = []
    for t in sorted({a['attack_type'] for a in attackers}):
        rows = [a for a in attackers if a['attack_type'] == t]
        out.append({'attack_type': t, 'attackers': len(rows),
                    'bayes_detected': sum(bool(a['bayes_detected']) for a in rows),
                    'llm_detected': sum(bool(a['llm_detected']) for a in rows),
                    'bayes_mean_ticks_to_block': _mean(a['bayes_ticks_to_block'] for a in rows),
                    'llm_mean_ticks_to_block': _mean(a['llm_ticks_to_block'] for a in rows)})
    return out


def explanations(b, l, attackers: List[dict]) -> List[dict]:
    """For each detected attacker: the Bayesian terms behind its quarantine vs the LLM's factors and reason."""
    out = []
    for a in attackers:
        d, start = a['device'], a['attack_start']
        rb = next((r for r in b.records if r.get('kind') == 'request' and r['src'] == d and r['tick'] >= start and r.get('quarantined_until')), None)
        rl = next((r for r in l.records if r.get('kind') == 'llm_eval' and r['src'] == d and r['tick'] >= start and r.get('quarantined_until')), None)
        if rb is None and rl is None:
            continue
        bayes_terms = '; '.join(f'{lab} {val:+.2f}' for lab, val in (rb['terms'] if rb else []))
        if rb and rb.get('step_up') and rb['step_up'].get('term'):
            bayes_terms += f"; {rb['step_up']['term'][0]} {rb['step_up']['term'][1]:+.2f}"
        out.append({'device': d, 'attack_type': a['attack_type'],
                    'bayes_block_tick': rb['tick'] if rb else None, 'bayes_terms': bayes_terms or '—',
                    'bayes_reason': rb['reason'] if rb else 'not blocked',
                    'llm_block_tick': rl['tick'] if rl else None,
                    'llm_factors': '; '.join(rl['factors']) if rl else '—', 'llm_reason': rl['reason'] if rl else 'not blocked'})
    return out


def present_facts(ctx: dict) -> set:
    """Which fact concepts are really present in an LLM context window."""
    w = {'token_requests': 0, 'distinct_destinations': 0, 'distinct_ports': 0, 'nonce_reuse_detected': 0, 'zk_failures': 0,
         'zone_changing_requests': 0, 'requests_without_configured_link': 0, 'requests_to_sensitive_targets': 0,
         'night_requests': 0, 'requests_per_tick': 0, 'max_payload_bytes_by_kind': {}, 'baseline_mean_payload_bytes_by_kind': {},
         'sensor_alarm_counts': {}, 'step_up_challenges': {}, **ctx['window_facts']}   # omitted fields mean zero
    base = w.get('baseline_requests_per_tick') or 0
    big = any(v > C.PAYLOAD_FACTOR * w['baseline_mean_payload_bytes_by_kind'].get(k, float('inf'))
              for k, v in w['max_payload_bytes_by_kind'].items())
    facts = {
        'token_burst': w['token_requests'] > C.TOKEN_FREE_PER_WINDOW,
        'scanning': w['distinct_destinations'] > C.SCAN_DISTINCT_DSTS or w['distinct_ports'] > C.SCAN_DISTINCT_PORTS,
        'nonce_reuse': w['nonce_reuse_detected'] > 0, 'zk_failure': w['zk_failures'] > 0, 'large_payload': big,
        'zone_change': w['zone_changing_requests'] > 0, 'unlinked': w['requests_without_configured_link'] > 0,
        'sensitive_target': w['requests_to_sensitive_targets'] > 0, 'night': w['night_requests'] > 0,
        'sensor_alarm': sum(w['sensor_alarm_counts'].values()) > 0,
        'rate_spike': bool(base) and w['requests_per_tick'] > C.RATE_SPIKE_FACTOR * base,
        'step_up': sum(w['step_up_challenges'].values()) > 0,
    }
    return {k for k, v in facts.items() if v}


def factor_overlap(l) -> dict:
    """Do the LLM's factors/reason mention facts that really occurred in its window?
    Heuristic keyword matching; a negated mention ("no scanning") still counts as a mention."""
    supported, unsupported, examples = 0, 0, []
    for r in l.records:
        if r.get('kind') != 'llm_eval' or not r['ok']:
            continue
        text = (' '.join(r['factors']) + ' ' + r['reason']).lower()
        mentioned = {fact for word, fact in FACT_KEYWORDS.items() if word in text}
        present = present_facts(r['context'])
        supported += len(mentioned & present)
        missing = mentioned - present
        unsupported += len(missing)
        if missing and len(examples) < 20:
            examples.append({'tick': r['tick'], 'device': r['src'], 'mentioned_but_absent': ', '.join(sorted(missing)),
                             'factors': '; '.join(r['factors'])[:200]})
    total = supported + unsupported
    return {'supported': supported, 'unsupported': unsupported, 'support_rate': _ratio(supported, total), 'examples': examples}


def repeat_consistency(factory: Callable[[], object], repeats: int, ticks: int) -> Dict[str, dict]:
    """Run the same (scenario, seed) `repeats` times; how identical are each world's decisions and trust?"""
    runs = []
    for _ in range(repeats):
        twin = factory()
        twin.run(ticks)
        runs.append(twin)
        twin.shutdown()
    out = {}
    for engine in runs[0].worlds:
        decisions = [{r['rid']: policy_of(r) for r in request_records(t.worlds[engine])} for t in runs]
        common = set.intersection(*(set(d) for d in decisions))
        same = sum(len({d[rid] for d in decisions}) == 1 for rid in common)
        stds = []
        for i in range(min(len(t.worlds[engine].trust_history) for t in runs)):
            for dev in runs[0].worlds[engine].devices:
                vals = [t.worlds[engine].trust_history[i].get(dev) for t in runs]
                if all(v is not None for v in vals):
                    stds.append(pstdev(vals))
        out[engine] = {'repeats': repeats, 'decision_consistency': _ratio(same, len(common)), 'mean_trust_std': _mean(stds)}
    return out


# ----- the report ------------------------------------------------------------------------------------------
def _better(b, l, direction: Optional[str]) -> str:
    if direction is None or b is None or l is None or isinstance(b, str) or isinstance(l, str):
        return '—'
    if abs(b - l) < 1e-12:
        return 'tie'
    return LABEL['bayesian'] if (b > l) == (direction == 'higher') else LABEL['llm']


def compare(twin, consistency: Optional[Dict[str, dict]] = None) -> dict:
    """Full side-by-side comparison of a Both-mode run."""
    b, l = twin.bayes, twin.llm
    if b is None or l is None:
        raise ValueError('the comparison needs both worlds (mode "both")')
    sb, sl = world_stats(b), world_stats(l)
    att = attackers_side_by_side(sb, sl)
    agree = agreement(b, l)
    overlap = factor_overlap(l)
    rows: List[dict] = []

    def add(section, name, bv, lv, direction=None, fmt=None):
        diff = lv - bv if isinstance(bv, (int, float)) and isinstance(lv, (int, float)) and not isinstance(bv, bool) else None
        rows.append({'section': section, 'parameter': name, 'bayesian': bv, 'llm': lv, 'difference': diff,
                     'better': _better(bv, lv, direction), 'format': fmt})

    mb, ml = sb['metrics'], sl['metrics']
    n_att = len(mb['attackers'])
    add('Detection', 'Attackers detected (of ' + str(n_att) + ')', mb['confusion']['TP'], ml['confusion']['TP'], 'higher', 'int')
    add('Detection', 'Detection rate', mb['detection_rate'], ml['detection_rate'], 'higher', 'pct')
    add('Detection', 'Mean ticks to first STEP-UP (attackers)', _mean(a['bayes_ticks_to_step_up'] for a in att), _mean(a['llm_ticks_to_step_up'] for a in att), 'lower', 'f1')
    add('Detection', 'Mean ticks to first BLOCK (detected attackers)', _mean(a['bayes_ticks_to_block'] for a in att), _mean(a['llm_ticks_to_block'] for a in att), 'lower', 'f1')
    add('Detection', 'Attackers blocked first', sum(a['first_to_block'] == 'Bayesian' for a in att), sum(a['first_to_block'] == 'LLM' for a in att), 'higher', 'int')
    for key, name, direction in (('tp', 'True positives', 'higher'), ('fp', 'False positives', 'lower'),
                                 ('tn', 'True negatives', 'higher'), ('fn', 'False negatives', 'lower')):
        add('Detection', name, sb[key], sl[key], direction, 'int')
    add('Detection', 'Precision', mb['precision'], ml['precision'], 'higher', 'f2')
    add('Detection', 'Recall', mb['recall'], ml['recall'], 'higher', 'f2')
    add('Detection', 'F1', mb['f1'], ml['f1'], 'higher', 'f2')
    add('Detection', 'Accuracy (devices)', sb['accuracy'], sl['accuracy'], 'higher', 'f2')
    add('Detection', 'Specificity', sb['specificity'], sl['specificity'], 'higher', 'f2')
    add('Detection', 'FPR (quarantined) — honest devices', sb['fpr_block_honest'], sl['fpr_block_honest'], 'lower', 'pct')
    add('Detection', 'FPR (quarantined) — noisy-honest devices', sb['fpr_block_noisy'], sl['fpr_block_noisy'], 'lower', 'pct')
    add('Detection', 'FPR (any step-up) — honest devices', sb['fpr_stepup_honest'], sl['fpr_stepup_honest'], 'lower', 'pct')
    add('Detection', 'FPR (any step-up) — noisy-honest devices', sb['fpr_stepup_noisy'], sl['fpr_stepup_noisy'], 'lower', 'pct')
    add('Detection', 'False-negative rate (missed attackers)', mb['false_negative_rate'], ml['false_negative_rate'], 'lower', 'pct')
    slow = [a for a in att if a['attack_type'] == 'slow_low']
    if slow:
        add('Detection', 'Slow-and-low detected', 'yes' if all(a['bayes_detected'] for a in slow) else 'no',
            'yes' if all(a['llm_detected'] for a in slow) else 'no')
    add('Detection', 'Wrongful quarantines of honest devices', sb['wrongful_quarantines'], sl['wrongful_quarantines'], 'lower', 'int')
    add('Detection', 'Total wrongful quarantine duration (ticks)', sb['wrongful_quarantine_ticks'], sl['wrongful_quarantine_ticks'], 'lower', 'int')

    add('Trust', 'Mean trust — honest devices', sb['honest_mean_trust'], sl['honest_mean_trust'], 'higher', 'f3')
    add('Trust', 'Mean trust — attackers while active', sb['attacker_mean_trust'], sl['attacker_mean_trust'], 'lower', 'f3')
    add('Trust', 'Separation gap (honest − attacker)', sb['separation_gap'], sl['separation_gap'], 'higher', 'f3')
    add('Trust', 'Trust volatility (mean std of tick-to-tick change)', sb['mean_volatility'], sl['mean_volatility'], 'lower', 'f4')
    add('Trust', 'State changes per device (mean)', sb['mean_flips'], sl['mean_flips'], 'lower', 'f1')
    add('Trust', 'Noisy-honest recovery to ALLOW (mean ticks)', sb['mean_recovery'], sl['mean_recovery'], 'lower', 'f1')
    add('Trust', 'Noisy-honest episodes not recovered by run end', sb['unrecovered_episodes'], sl['unrecovered_episodes'], 'lower', 'int')
    add('Trust', 'Brier score (trust vs truth)', sb['brier'], sl['brier'], 'lower', 'f4')
    add('Trust', 'Device-ticks with no trust value (PENDING)', sb['pending_ticks'], sl['pending_ticks'], 'lower', 'int')

    add('Decisions', 'Per-request accuracy', sb['request_accuracy'], sl['request_accuracy'], 'higher', 'pct')
    add('Decisions', 'Requests from active attackers allowed', sb['malicious_allowed'], sl['malicious_allowed'], 'lower', 'int')
    add('Decisions', 'Legitimate requests blocked', sb['benign_blocked'], sl['benign_blocked'], 'lower', 'int')
    add('Decisions', 'Decision agreement rate (same request)', agree['rate'], agree['rate'], None, 'pct')
    add('Decisions', 'Step-up challenges issued', sb['step_ups_issued'], sl['step_ups_issued'], None, 'int')
    add('Decisions', 'Step-up challenges passed', sb['step_ups_passed'], sl['step_ups_passed'], None, 'int')
    add('Decisions', 'Step-up challenges failed', sb['step_ups_failed'], sl['step_ups_failed'], None, 'int')

    add('Cost', 'Evaluations performed', sb['evaluations'], sl['evaluations'], None, 'int')
    add('Cost', 'Latency per evaluation — mean (ms)', sb['latency_mean'], sl['latency_mean'], 'lower', 'f3')
    add('Cost', 'Latency per evaluation — median (ms)', sb['latency_median'], sl['latency_median'], 'lower', 'f3')
    add('Cost', 'Latency per evaluation — p95 (ms)', sb['latency_p95'], sl['latency_p95'], 'lower', 'f3')
    add('Cost', 'Latency per evaluation — max (ms)', sb['latency_max'], sl['latency_max'], 'lower', 'f3')
    for kind in ('timeout', 'unreachable', 'invalid_json', 'invalid_schema', 'error'):
        add('Cost', f'LLM errors — {kind}', 0, sl['errors'].get(kind, 0), 'lower', 'int')
    add('Cost', 'Out-of-range values clamped', 0, sl['clamped'], 'lower', 'int')
    add('Cost', 'Mean confidence', '—', sl['confidence_mean'], None, 'f2')
    add('Cost', 'Mean confidence on correct verdicts', '—', sl['confidence_correct'], None, 'f2')
    add('Cost', 'Mean confidence on incorrect verdicts', '—', sl['confidence_incorrect'], None, 'f2')
    add('Cost', 'Throughput (ticks per second)', sb['throughput'], sl['throughput'], 'higher', 'f1')

    if consistency:
        cb, cl = consistency.get('bayesian', {}), consistency.get('llm', {})
        add('Consistency', f"Decision consistency across {cb.get('repeats', '?')} repeats", cb.get('decision_consistency'), cl.get('decision_consistency'), 'higher', 'pct')
        add('Consistency', 'Mean trust std across repeats', cb.get('mean_trust_std'), cl.get('mean_trust_std'), 'lower', 'f4')

    add('Explainability', 'LLM factor mentions supported by its window', 'n/a (terms are computed)', overlap['supported'], None, 'int')
    add('Explainability', 'LLM factor mentions NOT in its window', 'n/a (terms are computed)', overlap['unsupported'], None, 'int')
    add('Explainability', 'LLM factor support rate', 'n/a', overlap['support_rate'], None, 'pct')

    per_device = []
    for d in sorted(b.devices):
        tb, tl = device_timeline(b, d), device_timeline(l, d)
        per_device.append({'device': d, 'name': b.devices[d].name, 'truth': b.ground_truth[d]['agent'],
                           'attack_start': b.ground_truth[d]['attack_start'],
                           'bayes_final_trust': sb['final_trust'][d], 'llm_final_trust': sl['final_trust'][d],
                           'bayes_volatility': sb['volatility'][d], 'llm_volatility': sl['volatility'][d],
                           'bayes_state_changes': sb['flips'][d], 'llm_state_changes': sl['flips'][d],
                           'bayes_first_flag': tb['first_flag'], 'llm_first_flag': tl['first_flag'],
                           'bayes_first_block': tb['first_block'], 'llm_first_block': tl['first_block'],
                           'agreement_rate': agree['per_device'].get(d)})
    return {'scenario': twin.scenario_label, 'seed': twin.seed, 'ticks': twin.tick,
            'llm_model': getattr(getattr(twin.llm, 'controller', None), 'model_name', None),   # provider + model ID
            'rows': rows, 'per_device': per_device, 'attackers': att, 'by_attack_type': by_attack_type(att),
            'agreement': agree, 'explanations': explanations(b, l, att), 'factor_overlap': overlap,
            'reliability': {'bayesian': sb['reliability'], 'llm': sl['reliability']},
            'request_confusion': {'bayesian': sb['request_confusion'], 'llm': sl['request_confusion']},
            'latencies': {'bayesian': sb['latencies'], 'llm': sl['latencies']}, 'consistency': consistency,
            'stats': {'bayesian': sb, 'llm': sl}}


# ----- formatting and export ---------------------------------------------------------------------------------
def fmt(value, kind: Optional[str]) -> str:
    if value is None:
        return '—'
    if isinstance(value, str):
        return value
    if isinstance(value, bool):
        return 'yes' if value else 'no'
    if kind == 'pct':
        return f'{value:.1%}'
    if kind == 'int':
        return str(int(value))
    if kind in ('f1', 'f2', 'f3', 'f4'):
        return f'{value:.{kind[1]}f}'
    return f'{value:.3f}' if isinstance(value, float) else str(value)


def summary_table(cmp: dict) -> List[dict]:
    """Rows ready for display/CSV: parameter, Bayesian, LLM, difference, better."""
    return [{'section': r['section'], 'parameter': r['parameter'], 'Bayesian': fmt(r['bayesian'], r['format']),
             'LLM': fmt(r['llm'], r['format']),
             'difference (LLM − Bayes)': '—' if r['difference'] is None else fmt(r['difference'], r['format'] if r['format'] != 'pct' else 'f3'),
             'better': r['better']} for r in cmp['rows']]


def _md_table(rows: List[dict]) -> str:
    if not rows:
        return '_none_\n'
    cols = list(rows[0])
    out = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    for r in rows:
        out.append('| ' + ' | '.join(str(r[c]).replace('|', '/').replace('\n', ' ') for c in cols) + ' |')
    return '\n'.join(out) + '\n'


def save_charts(cmp: dict, twin, outdir: str) -> List[str]:
    """PNG charts for the report (matplotlib)."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    os.makedirs(outdir, exist_ok=True)
    files = []
    b, l = twin.bayes, twin.llm
    devices = sorted(b.devices)
    cols = 3
    rows = (len(devices) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(13, 2.6 * rows), squeeze=False, sharex=True, sharey=True)
    for ax, d in zip(axes.flat, devices):
        for world, style, lab in ((b, '-', 'Bayes'), (l, '--', 'LLM')):
            ticks = [h['tick'] for h in world.trust_history]
            vals = [h.get(d) if h.get(d) is not None else float('nan') for h in world.trust_history]
            ax.plot(ticks, vals, style, lw=1.6, label=lab)
        gt = b.ground_truth[d]
        if gt['attack_start'] is not None:
            ax.axvline(gt['attack_start'], ls=':', color='#a855f7')
        ax.axhline(b.controller.allow_threshold, ls=':', color='#22c55e', lw=0.8)
        ax.axhline(b.controller.block_threshold, ls=':', color='#ef4444', lw=0.8)
        ax.set_title(f"{d} ({gt['agent']})", fontsize=9)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(devices):]:
        ax.axis('off')
    axes.flat[0].legend(fontsize=7)
    fig.suptitle('Trust over time — solid Bayes, dashed LLM (dotted purple = attack start)')
    fig.tight_layout()
    p = os.path.join(outdir, 'trust_curves.png'); fig.savefig(p, dpi=130); plt.close(fig); files.append(p)

    types = cmp['by_attack_type']
    fig, ax = plt.subplots(figsize=(8, 4))
    if types:
        x = range(len(types))
        nan = float('nan')
        ax.bar([i - 0.2 for i in x], [t['bayes_mean_ticks_to_block'] if t['bayes_mean_ticks_to_block'] is not None else nan for t in types], 0.4, label='Bayesian')
        ax.bar([i + 0.2 for i in x], [t['llm_mean_ticks_to_block'] if t['llm_mean_ticks_to_block'] is not None else nan for t in types], 0.4, label='LLM')
        ax.set_xticks(list(x)); ax.set_xticklabels([t['attack_type'] for t in types], rotation=15)
        for i, t in enumerate(types):
            ax.annotate(f"B {t['bayes_detected']}/{t['attackers']} · L {t['llm_detected']}/{t['attackers']}", (i, 0), ha='center', va='bottom', fontsize=8)
    ax.set_ylabel('mean ticks from attack start to quarantine'); ax.set_title('Time-to-detect per attack type')
    if types:
        ax.legend()
    fig.tight_layout(); p = os.path.join(outdir, 'time_to_detect.png'); fig.savefig(p, dpi=130); plt.close(fig); files.append(p)

    fig, ax = plt.subplots(figsize=(8, 4))
    labels = ['honest quarantined', 'noisy quarantined', 'honest step-up', 'noisy step-up']
    keys = ['fpr_block_honest', 'fpr_block_noisy', 'fpr_stepup_honest', 'fpr_stepup_noisy']
    sb, sl = cmp['stats']['bayesian'], cmp['stats']['llm']
    x = range(len(keys))
    ax.bar([i - 0.2 for i in x], [sb[k] or 0 for k in keys], 0.4, label='Bayesian')
    ax.bar([i + 0.2 for i in x], [sl[k] or 0 for k in keys], 0.4, label='LLM')
    ax.set_xticks(list(x)); ax.set_xticklabels(labels); ax.set_ylabel('fraction of devices'); ax.set_title('False positives'); ax.legend()
    fig.tight_layout(); p = os.path.join(outdir, 'false_positives.png'); fig.savefig(p, dpi=130); plt.close(fig); files.append(p)

    fig, ax = plt.subplots(figsize=(6, 4))
    data = [cmp['latencies']['bayesian'] or [float('nan')], cmp['latencies']['llm'] or [float('nan')]]
    ax.boxplot(data, showfliers=True)
    ax.set_xticks([1, 2]); ax.set_xticklabels(['Bayesian (per request)', 'LLM (per evaluation)'])
    ax.set_yscale('log'); ax.set_ylabel('ms (log scale)'); ax.set_title('Latency per evaluation')
    fig.tight_layout(); p = os.path.join(outdir, 'latency_box.png'); fig.savefig(p, dpi=130); plt.close(fig); files.append(p)

    fig, ax = plt.subplots(figsize=(9, 3.5))
    pts = cmp['agreement']['per_tick']
    ax.plot([p_['tick'] for p_ in pts], [p_['agreement'] for p_ in pts], lw=1.4)
    for d, gt in b.ground_truth.items():
        if gt['attack_start'] is not None:
            ax.axvline(gt['attack_start'], ls=':', color='#a855f7')
    ax.set_ylim(-0.02, 1.02); ax.set_xlabel('tick'); ax.set_ylabel('share of requests'); ax.set_title('Decision agreement over time')
    fig.tight_layout(); p = os.path.join(outdir, 'agreement_over_time.png'); fig.savefig(p, dpi=130); plt.close(fig); files.append(p)
    return files


def report_markdown(cmp: dict, chart_files: Sequence[str] = ()) -> str:
    """comparison_report.md: every table plus the charts, ready to paste into the project report."""
    parts = [f"# Bayesian vs LLM — isolated comparison\n\nScenario **{cmp['scenario']}**, seed {cmp['seed']}, {cmp['ticks']} ticks. "
             'Both engines ran in separate worlds with identical inputs; each enforced its own decisions. '
             'Ground truth was used only to compute these numbers after the run.\n'
             + (f"\nLLM: **{cmp['llm_model']}** (provider and model ID).\n" if cmp.get('llm_model') else '')]
    table = summary_table(cmp)
    for section in dict.fromkeys(r['section'] for r in table):
        parts.append(f'\n## {section}\n\n' + _md_table([{k: v for k, v in r.items() if k != 'section'} for r in table if r['section'] == section]))
    parts.append('\n## Attackers side by side\n\n' + _md_table(cmp['attackers']))
    parts.append('\n## Per attack type\n\n' + _md_table(cmp['by_attack_type']))
    parts.append('\n## Per-request decisions vs truth\n\n' + _md_table(
        [{'outcome': k, 'Bayesian': cmp['request_confusion']['bayesian'].get(k, 0), 'LLM': cmp['request_confusion']['llm'].get(k, 0)}
         for k in sorted(set(cmp['request_confusion']['bayesian']) | set(cmp['request_confusion']['llm']))]))
    parts.append('\n## Trust calibration (reliability)\n\n**Bayesian**\n\n' + _md_table(cmp['reliability']['bayesian']) +
                 '\n**LLM**\n\n' + _md_table(cmp['reliability']['llm']))
    parts.append('\n## Explainability — why each attacker was blocked\n\n' + _md_table(cmp['explanations']))
    fo = cmp['factor_overlap']
    parts.append(f"\n## LLM factor overlap\n\n{fo['supported']} factor mentions matched facts in the LLM's window; "
                 f"{fo['unsupported']} mentioned facts that were not present (keyword heuristic).\n\n" + _md_table(fo['examples'][:10]))
    parts.append(f"\n## Disagreements\n\n{len(cmp['agreement']['cases'])} of {cmp['agreement']['requests_compared']} requests "
                 'got different decisions (first 25 shown; all in disagreements.csv).\n\n' + _md_table(cmp['agreement']['cases'][:25]))
    if chart_files:
        parts.append('\n## Charts\n\n' + '\n'.join(f'![{os.path.basename(f)}]({os.path.basename(f)})' for f in chart_files) + '\n')
    return ''.join(parts)


def write_exports(cmp: dict, twin, outdir: str, charts: bool = True) -> List[str]:
    """Write every export file into `outdir` and return the paths."""
    os.makedirs(outdir, exist_ok=True)
    files = []

    def write(name, text):
        path = os.path.join(outdir, name)
        with open(path, 'w', encoding='utf-8', newline='') as fh:
            fh.write(text)
        files.append(path)

    write('comparison_summary.csv', to_csv(summary_table(cmp)))
    write('per_device_comparison.csv', to_csv(cmp['per_device']))
    write('disagreements.csv', to_csv(cmp['agreement']['cases']) or 'no disagreements\n')
    write('bayesian_world_log.csv', to_csv(log_rows(twin.bayes)))
    write('llm_world_log.csv', to_csv(log_rows(twin.llm)))
    chart_files = save_charts(cmp, twin, outdir) if charts else []
    files += chart_files
    write('comparison_report.md', report_markdown(cmp, chart_files))
    return files


def summary_csv_text(cmp: dict) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=list(summary_table(cmp)[0]))
    w.writeheader(); w.writerows(summary_table(cmp))
    return buf.getvalue()
