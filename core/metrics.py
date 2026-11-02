"""Detection metrics, computed from ground truth AFTER the decisions were made.

Nothing here is ever fed back to the controller or the LLM. Definitions (state them in the report):

* An attacker is DETECTED (Bayesian) when the controller quarantines it at or after its attack start.
* time-to-STEP-UP = first tick at/after attack start when the device left the Trusted band or was challenged.
* time-to-BLOCK   = first tick at/after attack start when the device was quarantined.
* For the LLM the same two times use its scheduled verdicts (first STEP-UP-or-BLOCK, first BLOCK).
* An honest or noisy-honest device is a FALSE POSITIVE (block) if it is ever quarantined, and a
  false positive (step-up) if it ever left Trusted or was challenged. Single requests blocked by the
  ZK hard gate (e.g. a reboot glitch) are counted separately and are not device-level false positives.
* Confusion matrix: predicted positive = quarantined at least once (attackers: after attack start).
"""
from __future__ import annotations

import csv
import io
import json
from statistics import mean, median
from typing import Dict, List, Optional

from .agents import HONEST_TYPES


def _first(ticks: List[int]) -> Optional[int]:
    return min(ticks) if ticks else None


def _flagged(r: dict) -> bool:
    """A request where the engine treated the device as suspicious (challenge, Suspicious/Isolated, quarantine)."""
    return bool(r['step_up']) or r['band_after'] in ('Suspicious', 'Isolated') or bool(r.get('quarantined_until'))


def device_timeline(world, device_id: str, since: int = 0) -> Dict[str, Optional[int]]:
    """First flag and first quarantine ticks for one device in ONE world, at or after `since`.
    Works for both engines because both worlds log the same record fields."""
    reqs = [r for r in world.records if r.get('kind') == 'request' and r['src'] == device_id and r['tick'] >= since]
    evals = [r for r in world.records if r.get('kind') == 'llm_eval' and r['src'] == device_id and r['tick'] >= since]
    flag_ticks = [r['tick'] for r in reqs if _flagged(r)] + [r['tick'] for r in evals if r.get('llm_action') in ('STEP-UP', 'BLOCK')]
    block_ticks = [r['tick'] for r in reqs + evals if r.get('quarantined_until')]
    return {
        'first_flag': _first(flag_ticks),
        'first_block': _first(block_ticks),
        'hard_gate_blocks': sum(1 for r in reqs if r['action'] == 'BLOCK' and r['zk'] == 'FAIL'),
        'requests': len(reqs),
    }


def _delay(tick: Optional[int], start: int) -> Optional[int]:
    return None if tick is None else tick - start


def compute_metrics(live) -> dict:
    """Detection metrics for one finished (or paused) world — Bayesian or LLM, same definitions."""
    attackers, honest = [], []
    for device_id, gt in sorted(live.ground_truth.items()):
        if gt['agent'] in HONEST_TYPES:
            t = device_timeline(live, device_id, 0)
            honest.append({'device': device_id, 'truth': gt['agent'], 'any_step_up': t['first_flag'] is not None,
                           'quarantined': t['first_block'] is not None, 'hard_gate_blocks': t['hard_gate_blocks']})
        elif gt['attack_start'] is not None and gt['attack_start'] <= live.tick:
            start = gt['attack_start']
            t = device_timeline(live, device_id, start)
            attackers.append({'device': device_id, 'truth': gt['agent'], 'attack_start': start,
                              'detected': t['first_block'] is not None,
                              'ttd_step_up': _delay(t['first_flag'], start), 'ttd_block': _delay(t['first_block'], start)})
    tp = sum(a['detected'] for a in attackers)
    fn = len(attackers) - tp
    fp = sum(h['quarantined'] for h in honest)
    tn = len(honest) - fp
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    if not attackers:
        f1 = None  # undefined without any positive cases (e.g. normal_day)
    elif precision and recall:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0
    evals = [r for r in live.records if r.get('kind') == 'llm_eval']
    ok = [r for r in evals if r['ok']]
    llm = {'evaluations': len(evals), 'errors': len(evals) - len(ok),
           'mean_latency_ms': mean(r['latency_ms'] for r in evals) if evals else None,
           'median_latency_ms': median(r['latency_ms'] for r in evals) if evals else None,
           'offline': bool(getattr(live.controller, 'offline', False)) if live.engine == 'llm' else None}
    return {
        'engine': live.engine,
        'scenario': getattr(live, 'scenario_label', live.scenario.get('name')), 'seed': live.seed, 'ticks': live.tick,
        'attackers': attackers, 'honest': honest,
        'detection_rate': tp / len(attackers) if attackers else None,
        'false_negative_rate': fn / len(attackers) if attackers else None,
        'fpr_block': fp / len(honest) if honest else None,
        'fpr_step_up': mean(h['any_step_up'] for h in honest) if honest else None,
        'confusion': {'TP': tp, 'FN': fn, 'FP': fp, 'TN': tn},
        'precision': precision, 'recall': recall, 'f1': f1,
        'llm': llm,
    }


# ----- exports -------------------------------------------------------------------------
def log_rows(live) -> List[dict]:
    """Flat rows of the terminal log (one per record)."""
    from .live_sim import render_record
    rows = []
    for r in live.records:
        rows.append({'tick': r['tick'], 'kind': r.get('kind'), 'src': r.get('src', ''), 'dst': r.get('dst', ''),
                     'event_type': r.get('event_type', ''), 'zk': r.get('zk', ''),
                     'alarms': ' '.join(k for k, v in r.get('alarms', {}).items() if v),
                     'flags': ' '.join(r.get('derived', [])), 'trust_before': r.get('trust_before', ''),
                     'engine': r.get('engine', ''), 'rid': r.get('rid', ''),
                     'trust_after': r.get('trust_after', r.get('llm_trust', '')), 'band_after': r.get('band_after', ''),
                     'action': r.get('action', r.get('llm_action', '')), 'reason': r.get('reason', r.get('text', '')),
                     'text': ' | '.join(line.strip() for line in render_record(r))})
    return rows


def to_csv(rows: List[dict]) -> str:
    if not rows:
        return ''
    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue()


def trust_csv(live) -> str:
    return to_csv(live.trust_history)


def metrics_json(metrics: dict) -> str:
    return json.dumps(metrics, indent=2, default=str)


def _fmt(x, pct=False):
    if x is None:
        return '—'
    return f'{x:.0%}' if pct else (f'{x:.2f}' if isinstance(x, float) else str(x))


def summary_markdown(metrics: dict) -> str:
    """Short human-readable run summary for the report."""
    m = metrics
    lines = [f"# Run summary — {m['scenario']} (seed {m['seed']}, {m['ticks']} ticks)", '',
             f"## {'Bayesian' if m['engine'] == 'bayesian' else 'LLM'} world",
             f"- Attackers detected (quarantined): {m['confusion']['TP']}/{m['confusion']['TP'] + m['confusion']['FN']} "
             f"(detection rate {_fmt(m['detection_rate'], True)}, false-negative rate {_fmt(m['false_negative_rate'], True)})",
             f"- Honest devices quarantined (false-positive rate): {_fmt(m['fpr_block'], True)}; any step-up: {_fmt(m['fpr_step_up'], True)}",
             f"- Precision {_fmt(m['precision'])}, recall {_fmt(m['recall'])}, F1 {_fmt(m['f1'])}", '']
    if m['attackers']:
        lines += ['| Attacker | Type | Attack start | Ticks to STEP-UP | Ticks to BLOCK |', '|---|---|---|---|---|']
        for a in m['attackers']:
            lines.append(f"| {a['device']} | {a['truth']} | {a['attack_start']} | {_fmt(a['ttd_step_up'])} | {_fmt(a['ttd_block'])} |")
        lines.append('')
    if m['engine'] == 'llm':
        l = m['llm']
        lines += [f"- LLM evaluations: {l['evaluations']}, errors: {l['errors']}, mean latency {_fmt(l['mean_latency_ms'])} ms", '']
    lines.append('_Metrics use ground truth only after the run; the controllers never see it._')
    return '\n'.join(lines) + '\n'
