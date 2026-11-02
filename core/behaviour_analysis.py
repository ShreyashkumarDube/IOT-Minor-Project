"""Examiner / display-side analysis: how each engine turned the observed behaviour into trust.

This module only READS finished world logs. It is never imported by a controller
(`core/controller.py`, `core/live_llm.py`); a test checks that. It has three parts:

1. Calculation view: the Bayesian per-request calculation, step by step and unrounded; the LLM
   evaluation that decided a request, with every window fact traced to the requests behind it; and
   one behaviour shown through both engines.
2. Attack kinds from behaviour: signatures over OBSERVED facts only (`infer_attack_kind`). The true
   agent type is used only afterwards, in `score_*` / `attack_cards(..., reveal=True)`.
3. The alarm system: how each IDS / SIEM / BM alarm was drawn (environment side, examiner only) and
   how each engine used it.

Every number comes from the world's records; nothing is recomputed into a decision.
"""
from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from itertools import product
from statistics import mean, median
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from . import live_config as C
from .comparison import FACT_KEYWORDS, present_facts
from .controller import clamp_trust, logit, raw_band, sigmoid
from .events import CONTROLLER_ID, REQUEST_KINDS, is_night, likelihood

SENSOR_NAMES = {'IDS': 'Intrusion Detection System', 'SIEM': 'Security Information & Event Management', 'BM': 'Behaviour Monitor'}

# ----- one vocabulary for "behaviour", seen through both engines -------------------------------------------
# key -> (label, Bayesian term labels, LLM window_facts fields, factor-keyword concept)
BEHAVIOURS: Dict[str, dict] = {
    'token_burst': {'label': 'Many token requests', 'bayes': ['token_request'], 'llm': ['token_requests'], 'concept': 'token_burst'},
    'nonce_reuse': {'label': 'Reused nonce (replayed proof)', 'bayes': ['token_replay'], 'llm': ['nonce_reuse_detected'], 'concept': 'nonce_reuse'},
    'zk_failure': {'label': 'Failed ZK proof', 'bayes': ['zk_auth_fail'], 'llm': ['zk_failures'], 'concept': 'zk_failure'},
    'unlinked': {'label': 'Request with no configured link', 'bayes': ['no_topology_edge'], 'llm': ['requests_without_configured_link'], 'concept': 'unlinked'},
    'zone_change': {'label': 'Reaching another zone', 'bayes': ['cross_zone_access'], 'llm': ['zone_changing_requests'], 'concept': 'zone_change'},
    'sensitive_target': {'label': 'Sensitive target', 'bayes': ['sensitive_target_access'], 'llm': ['requests_to_sensitive_targets'], 'concept': 'sensitive_target'},
    'large_payload': {'label': 'Payload far above baseline', 'bayes': ['unusual_payload'], 'llm': ['max_payload_bytes_by_kind', 'baseline_mean_payload_bytes_by_kind'], 'concept': 'large_payload'},
    'scanning': {'label': 'Many destinations / ports', 'bayes': ['device_scan', 'port_scan'], 'llm': ['distinct_destinations', 'distinct_ports'], 'concept': 'scanning'},
    'rate_spike': {'label': 'Request rate far above baseline', 'bayes': ['rate_spike'], 'llm': ['requests_per_tick', 'baseline_requests_per_tick'], 'concept': 'rate_spike'},
    'night': {'label': 'Activity at night', 'bayes': ['off_hours_access'], 'llm': ['night_requests'], 'concept': 'night'},
    'sensor_alarm': {'label': 'Sensor alarms (IDS / SIEM / BM)', 'bayes': ['IDS!', 'SIEM!', 'BM!'], 'llm': ['sensor_alarm_counts'], 'concept': 'sensor_alarm'},
    'step_up': {'label': 'Step-up challenge result', 'bayes': ['step_up_fail', 'step_up_pass'], 'llm': ['step_up_challenges'], 'concept': 'step_up'},
    'ordinary': {'label': 'Ordinary traffic (telemetry, service, reads)', 'bayes': ['normal_telemetry', 'authorized_service', 'read_sensor', 'firmware_update', 'reboot'],
                 'llm': ['request_kind_counts'], 'concept': None},
}
CONCEPT_TO_BEHAVIOUR = {v['concept']: k for k, v in BEHAVIOURS.items() if v['concept']}


def behaviour_of_term(label: str) -> Optional[str]:
    """Map a Bayesian term label ('token_request×5', 'IDS!', 'no_topology_edge', ...) to a behaviour key."""
    if label == 'decay' or label.endswith('·'):
        return None
    base = label.split('×')[0]
    for key, b in BEHAVIOURS.items():
        if base in b['bayes'] or label in b['bayes']:
            return key
    return None


def concepts_in_text(text: str) -> set:
    """Fact concepts an LLM factor/reason mentions (same keyword table as the comparison report)."""
    low = text.lower()
    return {fact for word, fact in FACT_KEYWORDS.items() if word in low}


# =========================================================================================================
# Part 1A (left): the Bayesian calculation of one request
# =========================================================================================================
def _term_kind(label: str) -> str:
    if label == 'decay':
        return 'decay'
    if label.endswith('!') or label.endswith('·'):
        return 'sensor'
    base = label.split('×')[0]
    return 'request kind' if base in {k.value for k in REQUEST_KINDS} else 'derived anomaly'


def _why(label: str, r: dict, ev: dict) -> str:
    """The raw observed fact that produced a term (from the record's display-only evidence block)."""
    base = label.split('×')[0]
    f = lambda x, d=1: '—' if x is None else f'{x:.{d}f}'  # noqa: E731
    if base == 'token_request':
        return f"{ev['token_count']} token requests from this device in the last {C.WINDOW_TICKS} ticks > free allowance {C.TOKEN_FREE_PER_WINDOW}"
    if base in {k.value for k in REQUEST_KINDS}:
        return f"the request itself is `{base}` (ordinary traffic is weak evidence)"
    if base == 'zk_auth_fail':
        return 'the Schnorr proof did not verify against the public key registered at enrolment'
    if base == 'token_replay':
        return f"nonce {r.get('nonce')} was already seen from {r['src']} (a replayed proof)"
    if base == 'rate_spike':
        exp = None if ev['baseline_rate'] is None else ev['baseline_rate'] * C.WINDOW_TICKS
        return (f"{ev['window_requests']} requests in the {C.WINDOW_TICKS}-tick window > threshold {f(ev['rate_threshold'])} "
                f"= max({C.RATE_SPIKE_FACTOR:g} × {f(exp)}, {f(exp)} + {C.RATE_MIN_EXCESS})")
    if base == 'unusual_payload':
        b = ev['baseline_bytes']
        return f"{ev['bytes']} B > {C.PAYLOAD_FACTOR:g} × baseline {f(b)} B = {f(None if b is None else C.PAYLOAD_FACTOR * b)} B"
    if base == 'device_scan':
        return f"{ev['distinct_destinations']} distinct destinations in the window > {C.SCAN_DISTINCT_DSTS}"
    if base == 'port_scan':
        return f"{ev['distinct_ports']} distinct ports in the window > {C.SCAN_DISTINCT_PORTS}"
    if base == 'no_topology_edge':
        return f"{r['src']} → {r['dst']}: no configured link"
    if base == 'cross_zone_access':
        return f"target zone '{ev['target_zone']}' is not one of this device's usual zones {ev['usual_zones']}"
    if base == 'sensitive_target_access':
        return f"target sensitivity {ev['target_sensitivity']} ≥ {C.SENSITIVE_TARGET} and no configured link"
    if base == 'off_hours_access':
        return f"tick {r['tick']} is night (ticks {C.NIGHT_START}–{C.TICKS_PER_DAY - 1} of every {C.TICKS_PER_DAY})"
    return ''


def _term_row(label: str, delta: float, r: dict, ev: dict) -> dict:
    kind = _term_kind(label)
    row = {'term': label, 'kind': kind, 'observed fact': '', 'P(e|legit)': None, 'P(e|attacker)': None,
           'ratio': None, 'ln(ratio)': None, 'weight': None, 'term value': delta, 'formula': ''}
    if kind == 'sensor':
        name, fired = label[:-1], label.endswith('!')
        tp, fp = C.SENSORS[name]
        p_legit, p_mal = (fp, tp) if fired else (1 - fp, 1 - tp)
        row.update({'observed fact': f"{name} {'fired' if fired else 'stayed quiet'}",
                    'formula': f"{C.SENSOR_WEIGHT:g} × ln({'FP/TP' if fired else '(1−FP)/(1−TP)'}) = {C.SENSOR_WEIGHT:g} × ln({p_legit:g}/{p_mal:g})",
                    'P(e|legit)': p_legit, 'P(e|attacker)': p_mal, 'weight': C.SENSOR_WEIGHT})
    else:
        p_legit, p_mal = likelihood(label.split('×')[0])
        row.update({'observed fact': _why(label, r, ev), 'P(e|legit)': p_legit, 'P(e|attacker)': p_mal, 'weight': C.EVENT_WEIGHT,
                    'formula': f"{C.EVENT_WEIGHT:g} × ln({p_legit:g}/{p_mal:g})"})
    row['ratio'] = row['P(e|legit)'] / row['P(e|attacker)']
    row['ln(ratio)'] = math.log(row['ratio'])
    return row


def previous_update(world, record: dict) -> Optional[dict]:
    """The same device's previous processed request in this world (where the decay started from)."""
    found = None
    for r in world.records:
        if r is record:
            return found
        if r.get('kind') == 'request' and r['src'] == record['src'] and 'calc' in r:
            found = r
    return found


def bayes_steps(record: dict, world=None) -> dict:
    """Every step of one Bayesian update with the real, unrounded numbers, ready to display and check."""
    calc = record.get('calc')
    if calc is None:
        raise ValueError('this record has no calculation (dropped request or not a Bayesian record)')
    ev = calc['evidence']
    decay = sum(d for lab, d in calc['terms'] if lab == 'decay')
    evidence = [_term_row(lab, d, record, ev) for lab, d in calc['terms'] if lab != 'decay']
    total = sum(r['term value'] for r in evidence)
    L1 = calc['L_start'] + total
    recomputed = clamp_trust(sigmoid(L1))
    out = {
        'rid': record.get('rid'), 'tick': record['tick'], 'src': record['src'], 'dst': record['dst'],
        'trust_before': calc['trust_before'], 'L0': calc['L_start'],
        'decay': decay, 'L_before_decay': calc['L_start'] - decay, 'trust_before_decay': sigmoid(calc['L_start'] - decay),
        'decay_lambda': C.DECAY_LAMBDA, 'prior': C.LIVE_PRIOR, 'logit_prior': logit(C.LIVE_PRIOR),
        'evidence': evidence, 'sum_terms': total, 'L1': L1, 'sigmoid': sigmoid(L1), 'trust_after': calc['trust_after'],
        'clamped': abs(sigmoid(L1) - calc['trust_after']) > 1e-12, 'recomputed': recomputed,
        'check_ok': abs(recomputed - calc['trust_after']) < 1e-9,
        'raw_band': calc['raw_band'], 'band_before': calc['band_before'], 'band_after': calc['band_after'],
        'hysteresis_held': calc['raw_band'] != calc['band_after'], 'thresholds': calc['thresholds'],
        'probation': calc['probation'], 'policy_action': record.get('policy_action', record['action']),
        'zk': record['zk'], 'replay': record['replay'], 'final_action': record['action'], 'reason': record['reason'],
        'quarantined_until': record.get('quarantined_until'), 'step_up': None, 'decay_ticks': None, 'decay_check': None,
    }
    out['check_text'] = (f"{calc['L_start']:+.4f} + ({total:+.4f}) = {L1:+.4f} → σ({L1:+.4f}) = {sigmoid(L1):.4f}"
                         + (f" → clamp to [{C.TRUST_MIN}, {C.TRUST_MAX}] = {recomputed:.4f}" if out['clamped'] else '')
                         + (' ✔ matches the stored trust' if out['check_ok'] else ' ✘ does NOT match the stored trust'))
    su = calc.get('step_up')
    if su:
        lab, d = su['term']
        p_legit, p_mal = likelihood(lab)
        rec = clamp_trust(sigmoid(su['L_start'] + d))
        out['step_up'] = {'zk': record['step_up']['zk'], 'mfa': record['step_up']['mfa'], 'term': lab, 'term value': d,
                          'P(e|legit)': p_legit, 'P(e|attacker)': p_mal, 'L_start': su['L_start'], 'L_end': su['L_start'] + d,
                          'trust_after': su['trust_after'], 'raw_band': su['raw_band'], 'band_before': su['band_before'],
                          'band_after': su['band_after'], 'check_ok': abs(rec - su['trust_after']) < 1e-9,
                          'check_text': f"{su['L_start']:+.4f} + ({d:+.4f}) = {su['L_start'] + d:+.4f} → σ = {sigmoid(su['L_start'] + d):.4f}"
                                        + (f" → clamp {rec:.4f}" if abs(sigmoid(su['L_start'] + d) - rec) > 1e-12 else '')
                                        + (' ✔' if abs(rec - su['trust_after']) < 1e-9 else ' ✘')}
    if world is not None and decay:
        prev = previous_update(world, record)
        if prev is not None:
            start = (prev['calc'].get('step_up') or prev['calc'])['trust_after']
            k = record['tick'] - prev['tick']
            L = logit(start)
            for _ in range(k):
                L = logit(clamp_trust(sigmoid(C.DECAY_LAMBDA * L + (1 - C.DECAY_LAMBDA) * logit(C.LIVE_PRIOR))))
            out['decay_ticks'] = k
            out['decay_from_trust'] = start
            out['decay_check'] = abs(L - calc['L_start']) < 1e-9   # False if a probation reset happened in between
    return out


def waterfall(steps: dict) -> List[Tuple[str, float]]:
    """(label, Δ) pairs from L0 to L1 (and the step-up term), for a waterfall chart."""
    items = [(r['term'], r['term value']) for r in steps['evidence']]
    if steps['step_up']:
        items.append((steps['step_up']['term'], steps['step_up']['term value']))
    return items


def all_bayes_rechecks(world) -> Tuple[int, int]:
    """(records checked, records whose re-added terms reproduce the stored trust to 1e-9)."""
    n = ok = 0
    for r in world.records:
        if r.get('kind') == 'request' and 'calc' in r:
            s = bayes_steps(r)
            n += 1
            ok += s['check_ok'] and (s['step_up'] is None or s['step_up']['check_ok'])
    return n, ok


# =========================================================================================================
# Part 1A (right): the LLM evaluation that decided a request, and its window traced to request ids
# =========================================================================================================
def llm_evaluations(world, device_id: str) -> List[dict]:
    return [r for r in world.records if r.get('kind') == 'llm_eval' and r['src'] == device_id]


def deciding_evaluation(world, record: dict) -> Tuple[Optional[dict], Optional[dict]]:
    """(evaluation whose decision applied to this request, latest evaluation attempt before it — possibly an LLM_ERROR)."""
    evals = [e for e in llm_evaluations(world, record['src']) if e['tick'] < record['tick']]
    tick = record.get('llm_decision_tick')
    deciding = next((e for e in reversed(evals) if e['ok'] and e['tick'] == tick), None) if tick is not None else None
    return deciding, (evals[-1] if evals else None)


def _llm_window_records(world, device_id: str, tick: int) -> List[dict]:
    """World L's own processed requests that were inside the device's window at the end of `tick`."""
    return [r for r in world.records if r.get('kind') == 'request' and r['src'] == device_id
            and tick - C.WINDOW_TICKS < r['tick'] <= tick]


def window_trace(world, evaluation: dict) -> List[dict]:
    """Every window fact in the context: value sent, value rebuilt from World L's records, how, and which requests."""
    d, t = evaluation['src'], evaluation['tick']
    ctl = world.controller
    led = ctl.ledgers[d]
    recs = _llm_window_records(world, d, t)
    sent_facts = evaluation['context']['window_facts']
    omitted = set()

    class _Facts(dict):   # a compact context omits zero / empty fields: read them back as 0 / {} and remember it
        def __missing__(self, key):
            omitted.add(key)
            return {} if key in ('request_kind_counts', 'max_payload_bytes_by_kind', 'baseline_mean_payload_bytes_by_kind',
                                 'sensor_alarm_counts', 'step_up_challenges') else (None if key == 'baseline_requests_per_tick' else 0)

    facts = _Facts(sent_facts)

    def linked(r):
        return r['dst'] == CONTROLLER_ID or (d, r['dst']) in ctl.edges

    def zone_changed(r):
        tgt = ctl.ledgers.get(r['dst'])
        return bool(tgt and tgt.zone != led.zone)

    def sens(r):
        tgt = ctl.ledgers.get(r['dst'])
        return tgt.sensitivity if tgt else 0.0

    rows = []

    def add(name, sent, rebuilt, how, rids=None, kind='count'):
        top = name.split('.')[0]
        was_omitted = top in omitted or ('.' in name and name.split('.', 1)[1] not in sent_facts.get(top, {}))
        rows.append({'fact': name, 'value sent to LLM': sent, 'rebuilt from records': rebuilt, 'matches': sent == rebuilt,
                     'omitted (= 0)': was_omitted,
                     'how it is computed': how, 'contributing requests': list(rids or []), 'n contributing': len(rids or []),
                     'type': kind})

    all_rids = [r['rid'] for r in recs]
    add('requests', facts['requests'], len(recs), f'processed requests from {d} with tick in ({t - C.WINDOW_TICKS}, {t}]', all_rids)
    span = max(1, min(C.WINDOW_TICKS, t - led.enrolled_tick))
    add('requests_per_tick', facts['requests_per_tick'], round(len(recs) / span, 2), f'requests ÷ {span} ticks', all_rids, 'derived')
    warm = [r for r in world.records if r.get('kind') == 'request' and r['src'] == d and r['tick'] <= led.enrolled_tick + C.WARMUP_TICKS]
    base_then = len(warm) / C.WARMUP_TICKS if t - led.enrolled_tick > C.WARMUP_TICKS else None   # frozen at the start of tick 21
    add('baseline_requests_per_tick', facts['baseline_requests_per_tick'], base_then,
        f'requests in the first {C.WARMUP_TICKS} ticks ÷ {C.WARMUP_TICKS} (frozen after warm-up)', [], 'baseline')
    kinds = Counter(r['event_type'] for r in recs)
    for k in sorted(set(kinds) | set(facts['request_kind_counts'])):
        add(f'request_kind_counts.{k}', facts['request_kind_counts'].get(k, 0), kinds.get(k, 0), f"requests of kind '{k}'",
            [r['rid'] for r in recs if r['event_type'] == k])
    add('token_requests', facts['token_requests'], kinds.get('token_request', 0), "requests of kind 'token_request'",
        [r['rid'] for r in recs if r['event_type'] == 'token_request'])
    dsts = sorted({r['dst'] for r in recs if r['dst'] != CONTROLLER_ID})
    add('distinct_destinations', facts['distinct_destinations'], len(dsts), f'distinct targets (controller excluded): {", ".join(dsts) or "none"}',
        [next(r['rid'] for r in recs if r['dst'] == x) for x in dsts], 'distinct')
    ports = sorted({r['port'] for r in recs})
    add('distinct_ports', facts['distinct_ports'], len(ports), f'distinct ports: {", ".join(map(str, ports)) or "none"}',
        [next(r['rid'] for r in recs if r['port'] == x) for x in ports], 'distinct')
    add('requests_without_configured_link', facts['requests_without_configured_link'], sum(not linked(r) for r in recs),
        'target is a device and (src, dst) is not a configured link', [r['rid'] for r in recs if not linked(r)])
    add('zone_changing_requests', facts['zone_changing_requests'], sum(zone_changed(r) for r in recs),
        f"target's zone differs from {d}'s own zone ({led.zone})", [r['rid'] for r in recs if zone_changed(r)])
    add('requests_to_sensitive_targets', facts['requests_to_sensitive_targets'], sum(sens(r) >= C.SENSITIVE_TARGET for r in recs),
        f'target sensitivity ≥ {C.SENSITIVE_TARGET}', [r['rid'] for r in recs if sens(r) >= C.SENSITIVE_TARGET])
    for k in sorted(facts['max_payload_bytes_by_kind']):
        of_kind = [r for r in recs if r['event_type'] == k]
        top = max(of_kind, key=lambda r: r['bytes']) if of_kind else None
        add(f'max_payload_bytes_by_kind.{k}', facts['max_payload_bytes_by_kind'][k], top['bytes'] if top else None,
            f"largest '{k}' payload in the window (baseline {facts['baseline_mean_payload_bytes_by_kind'].get(k, '—')} B)",
            [top['rid']] if top else [], 'max')
    add('zk_failures', facts['zk_failures'], sum(r['zk'] == 'FAIL' for r in recs), 'requests whose proof failed', [r['rid'] for r in recs if r['zk'] == 'FAIL'])
    add('nonce_reuse_detected', facts['nonce_reuse_detected'], sum(bool(r['replay']) for r in recs), 'requests whose nonce had been seen before',
        [r['rid'] for r in recs if r['replay']])
    add('night_requests', facts['night_requests'], sum(is_night(r['tick']) for r in recs),
        f'requests during ticks {C.NIGHT_START}–{C.TICKS_PER_DAY - 1} of each day', [r['rid'] for r in recs if is_night(r['tick'])])
    for s in C.SENSORS:
        add(f'sensor_alarm_counts.{s}', facts['sensor_alarm_counts'].get(s, 0), sum(bool(r['alarms'].get(s)) for r in recs),
            f'requests on which {s} fired', [r['rid'] for r in recs if r['alarms'].get(s)])
    for outcome, action in (('passed', 'ALLOW'), ('failed', 'BLOCK')):
        ids = [r['rid'] for r in recs if r.get('step_up') and r['action'] == action]
        add(f'step_up_challenges.{outcome}', facts['step_up_challenges'].get(outcome, 0), len(ids), f'step-up challenges {outcome}', ids)
    return rows


def factor_grounding(evaluation: dict) -> List[dict]:
    """Each factor the LLM named → the fact concepts it mentions → supported by its window (✔) or not (✘)."""
    present = present_facts(evaluation['context'])
    rows = []
    for text in list(evaluation.get('factors') or []) + ([f"(reason) {evaluation['reason']}"] if evaluation.get('reason') else []):
        found = sorted(concepts_in_text(text))
        rows.append({'factor': text, 'facts it refers to': ', '.join(found) or '— (no known fact named)',
                     'supported by the window': ' '.join(f"{c} {'✔' if c in present else '✘'}" for c in found) or '—',
                     'all supported': all(c in present for c in found) if found else None})
    return rows


def clamped_fields(evaluation: dict) -> List[str]:
    """Which numeric fields of the raw reply were outside [0, 1] (and therefore clamped)."""
    try:
        raw = json.loads(evaluation.get('raw_reply') or '')
    except (TypeError, ValueError):
        return []
    out = []
    for key in ('trust_score', 'risk_score', 'confidence'):
        try:
            v = float(raw.get(key)) if isinstance(raw, dict) and raw.get(key) is not None else None
        except (TypeError, ValueError):
            continue
        if v is not None and not 0.0 <= v <= 1.0:
            out.append(f'{key}={v:g}')
    return out


# =========================================================================================================
# Part 1B and 1C: same behaviour, two treatments; how the two calculations differ
# =========================================================================================================
def same_behaviour_table(device_id: str, tick: int, bayes_world=None, llm_world=None) -> List[dict]:
    """One row per behaviour seen on this device in the last WINDOW_TICKS ticks up to `tick`, through both engines."""
    lo = tick - C.WINDOW_TICKS
    bterms: Dict[str, List[Tuple[str, float]]] = defaultdict(list)
    observed: Dict[str, List[str]] = defaultdict(list)
    if bayes_world is not None:
        for r in bayes_world.records:
            if r.get('kind') == 'request' and r['src'] == device_id and lo < r['tick'] <= tick:
                labels = list(r['terms']) + ([tuple(r['step_up']['term'])] if r.get('step_up') and r['step_up'].get('term') else [])
                for lab, d in labels:
                    key = behaviour_of_term(lab)
                    if key:
                        bterms[key].append((lab, d))
                        observed[key].append(r['rid'])
    ctx_eval = None
    if llm_world is not None:
        evs = [e for e in llm_evaluations(llm_world, device_id) if e['tick'] <= tick]
        ctx_eval = evs[-1] if evs else None
    facts = ctx_eval['context']['window_facts'] if ctx_eval else None
    llm_present = present_facts(ctx_eval['context']) if ctx_eval else set()
    factor_text = ' '.join((ctx_eval or {}).get('factors') or []) + ' ' + ((ctx_eval or {}).get('reason') or '')
    mentioned = concepts_in_text(factor_text) if ctx_eval and ctx_eval.get('ok') else set()
    rows = []
    for key, b in BEHAVIOURS.items():
        in_llm = b['concept'] in llm_present if b['concept'] else bool(facts)
        if key not in bterms and not (in_llm and key != 'ordinary'):
            continue
        if bayes_world is None:
            bcell = '— (Bayesian engine not running)'
        elif key in bterms:
            by_label = defaultdict(list)
            for lab, d in bterms[key]:
                by_label[lab.split('×')[0] if not lab.endswith('!') else lab].append(d)
            bcell = '; '.join(f"{lab} ×{len(v)} → total {sum(v):+.3f}" for lab, v in by_label.items())
        else:
            bcell = 'no term (not treated as evidence here)'
        if llm_world is None:
            lcell = '— (LLM engine not running)'
        elif facts is None:
            lcell = 'PENDING — no evaluation yet'
        else:
            vals = []
            for field in b['llm']:
                v = facts.get(field, '0 (omitted)')
                vals.append(f'{field} = {json.dumps(v)}')
            lcell = f"t={ctx_eval['tick']}: " + '; '.join(vals)
        if llm_world is None or ctx_eval is None or not ctx_eval.get('ok'):
            fcell = '—'
        elif b['concept'] and b['concept'] in mentioned:
            hits = [f for f in ctx_eval['factors'] if b['concept'] in concepts_in_text(f)]
            fcell = '; '.join(hits) or '(mentioned in the reason)'
        else:
            fcell = 'not mentioned'
        rids = observed.get(key, [])
        rows.append({'behaviour': b['label'], 'how it was observed': f"{len(rids)} Bayesian term(s) in t∈({lo}, {tick}]" + (f", e.g. {rids[0]}" if rids else ''),
                     'Bayesian: event → term(s)': bcell, 'LLM: context field(s) → value': lcell, 'LLM factor that mentions it': fcell})
    return rows


def sensor_term(name: str, fired: bool) -> float:
    tp, fp = C.SENSORS[name]
    return C.SENSOR_WEIGHT * (math.log(fp / tp) if fired else math.log((1 - fp) / (1 - tp)))


def differences_table(every_n: int = C.LLM_EVERY_N, bayes_world=None, llm_world=None) -> List[dict]:
    """How the two calculations differ — every number taken from the code/config (and the worlds, if given)."""
    half_life = math.log(0.5) / math.log(C.DECAY_LAMBDA)
    ex = {k: C.EVENT_WEIGHT * math.log(likelihood(k)[0] / likelihood(k)[1]) for k in ('no_topology_edge', 'normal_telemetry')}
    s_fire = ', '.join(f'{n} {sensor_term(n, True):+.3f}' for n in C.SENSORS)
    s_quiet = ', '.join(f'{n} {sensor_term(n, False):+.3f}' for n in C.SENSORS)
    lat_b = lat_l = None
    if bayes_world is not None and bayes_world.eval_ms:
        lat_b = mean(bayes_world.eval_ms.values())
    if llm_world is not None:
        ls = [e['latency_ms'] for e in llm_world.records if e.get('kind') == 'llm_eval']
        lat_l = mean(ls) if ls else None
    model = getattr(getattr(llm_world, 'controller', None), 'model_name', None)
    rows = [
        ('When it updates', 'after every request (one update per request)',
         f'only when evaluated: every {every_n} of the device\'s requests, or at the end of the tick after a hard event '
         f'({", ".join(C.LLM_HARD_EVENTS)}); the decision is held in between'),
        ('Memory', f'one number in log-odds; every tick it is pulled back toward the prior '
                   f'(L ← {C.DECAY_LAMBDA}·L + {1 - C.DECAY_LAMBDA:.2f}·logit({C.LIVE_PRIOR})), so evidence fades with a half-life of ≈{half_life:.0f} ticks',
         f'no running number: a fresh summary of the last {C.WINDOW_TICKS} ticks of facts each time, plus its own previous assessment'),
        ('Type of maths', 'additive: L_new = L + Σ weight × ln(P(e|legit)/P(e|attacker)); trust = σ(L), clamped to '
                          f'[{C.TRUST_MIN}, {C.TRUST_MAX}]', 'judgment of a language model on a JSON summary; it returns a trust score directly'),
        ('What a behaviour becomes', f'a fixed term per event, e.g. no_topology_edge {ex["no_topology_edge"]:+.3f}, '
                                     f'normal_telemetry {ex["normal_telemetry"]:+.3f}',
         'a count, maximum or flag in a JSON field, e.g. requests_without_configured_link = 3; the model decides how much it matters'),
        ('Sensors', f'each sensor adds a weighted term on every request: fired {s_fire}; quiet {s_quiet}',
         'counts of alarms per sensor in the window, plus each sensor\'s reference TP/FP rates'),
        ('Cross-zone', 'target zone not among the device\'s usual zones (own zone + zones of its configured targets)',
         'zone_changing_requests: target zone ≠ the device\'s own zone (even along a configured link)'),
        ('Sensitive targets', f'only when sensitivity ≥ {C.SENSITIVE_TARGET} AND there is no configured link',
         f'requests_to_sensitive_targets: every request to sensitivity ≥ {C.SENSITIVE_TARGET}, linked or not'),
        ('Night', 'off_hours_access only for service/token requests at night', 'night_requests counts every request at night'),
        ('Starting point', f'every device starts at the prior {C.LIVE_PRIOR}', f'PENDING (no trust value) until the first evaluation; '
                                                                               f'default {C.PENDING_DEFAULT_ACTION} if the ZK check passes'),
        ('Determinism', 'fully deterministic: same inputs → same numbers', 'not guaranteed; measured with the Repeat-run check'
                                                                          + (f' (here: {model})' if model else '')),
        ('Latency', 'microseconds per request' + (f' (this run: mean {lat_b:.3f} ms)' if lat_b is not None else ''),
         'hundreds of ms to seconds per call with a real model' + (f' (this run: mean {lat_l:.1f} ms)' if lat_l is not None else '')),
        ('How it fails', 'wrong hand-set likelihoods; correlated evidence over-counted (tempered by the weights)',
         'timeouts, unreachable server, invalid JSON / schema (→ LLM_ERROR), invented facts, prompt sensitivity'),
        ('How to audit it', 'add the printed terms: every step is reproducible by hand',
         'read the exact context and raw reply; check each factor against the window facts (factor grounding)'),
    ]
    return [{'aspect': a, 'Bayesian engine': b, 'LLM engine': l} for a, b, l in rows]


def behaviour_first_ticks(world, device_id: str) -> Dict[str, int]:
    """First tick each behaviour produced a Bayesian term for this device (for the vertical lines in 1D)."""
    first: Dict[str, int] = {}
    for r in world.records:
        if r.get('kind') == 'request' and r['src'] == device_id and r['tick'] > C.WARMUP_TICKS:
            for lab, _ in r['terms']:
                key = behaviour_of_term(lab)
                if key and key not in ('ordinary', 'sensor_alarm') and key not in first:
                    first[key] = r['tick']
    return first


def llm_fact_first_ticks(world, device_id: str) -> Dict[str, int]:
    """First evaluation tick at which each fact concept was present in the LLM's window."""
    first: Dict[str, int] = {}
    for e in llm_evaluations(world, device_id):
        for c in present_facts(e['context']):
            key = CONCEPT_TO_BEHAVIOUR.get(c)
            if key and key not in ('sensor_alarm',) and key not in first:
                first[key] = e['tick']
    return first


# =========================================================================================================
# Part 2: attack kinds recognised from OBSERVED behaviour only
# =========================================================================================================
# Thresholds of the signatures (assumptions, chosen from the attack descriptions, NOT tuned to the scores).
SIG_TOKENS_PER_WINDOW = 15     # token_flood: ≥ 15 token requests in some 20-tick window (honest ≈ 1–2)
SIG_RATE_FACTOR = 3.0          # token_flood also needs the window rate ≥ 3 × the device's own warm-up rate
SIG_PORTS_PER_WINDOW = C.SCAN_DISTINCT_PORTS   # scanning: > 5 distinct ports in some window (honest devices use one port)
SIG_MOVES_PER_WINDOW = 8       # compromised_valid: ≥ 8 suspicious moves with a VALID proof in some window
SIG_SLOW_TOTAL = 4             # slow_low: ≥ 4 suspicious moves in total, but fewer than 8 in any window...
SIG_SLOW_STRONG = 1            # ...including at least 1 exfiltration-sized payload (> 4 × baseline); maintenance access never has one
SIG_ZK_FAIL = 2                # credential_replay: ≥ 2 ZK failures that a reboot does not explain
# A failed proof is 'explained by a reboot' if the device rebooted and has not yet sent a valid proof since
# on ordinary traffic (a rebooting device fails until it recovers). Version 1 used a fixed 3-tick grace instead; see docs/CALCULATION_VIEW.md.

KIND_LABEL = {'credential_replay': 'Credential / proof replay', 'token_flood': 'Token flood', 'scanning': 'Scanning',
              'compromised_valid': 'Compromised device with a valid key', 'slow_low': 'Slow-and-low',
              'no_attack': 'No attack pattern', 'noisy_benign': 'No attack (explainable noise: reboots / maintenance)'}
KIND_SIGNATURE = {
    'credential_replay': 'a reused nonce, or ≥ 2 ZK failures that no reboot explains (a reboot explains failures until the next valid proof on ordinary traffic)',
    'token_flood': f'≥ {SIG_TOKENS_PER_WINDOW} token requests in one {C.WINDOW_TICKS}-tick window and a window rate ≥ {SIG_RATE_FACTOR:g}× its own baseline',
    'scanning': f'> {SIG_PORTS_PER_WINDOW} distinct ports in one window (honest devices use a single port)',
    'compromised_valid': f'ZK passes, but ≥ {SIG_MOVES_PER_WINDOW} suspicious moves in one window (no link + other zone, unlinked sensitive target, or payload > {C.PAYLOAD_FACTOR:g}× baseline)',
    'slow_low': f'ZK passes; ≥ {SIG_SLOW_TOTAL} suspicious moves spread out (< {SIG_MOVES_PER_WINDOW} in any window), at least one with an exfiltration-sized payload',
    'noisy_benign': 'isolated ZK glitches right after a reboot and/or rare maintenance access, without a pattern',
    'no_attack': 'nothing above',
}
HONEST_EXPECTED = ('no_attack', 'noisy_benign')
ORDINARY_KINDS = ('normal_telemetry', 'authorized_service', 'read_sensor', 'token_request')


def wire_observations(world, device_id: str, upto_tick: Optional[int] = None) -> List[dict]:
    """What a network tap saw from this device (requests and quarantine-dropped requests) — no ground truth.
    Uses only public data: the records' observed fields, device zones/sensitivities and the configured links."""
    devs, links = world.devices, set(world.connections)
    own = devs[device_id]
    out = []
    for r in world.records:
        if r.get('kind') not in ('request', 'dropped') or r['src'] != device_id:
            continue
        if upto_tick is not None and r['tick'] > upto_tick:
            break
        tgt = devs.get(r['dst'])
        out.append({'rid': r['rid'], 'tick': r['tick'], 'dst': r['dst'], 'kind': r['event_type'], 'port': r['port'],
                    'bytes': r['bytes'], 'zk': r.get('zk', '—'), 'nonce': r.get('nonce'),
                    'linked': r['dst'] == CONTROLLER_ID or (device_id, r['dst']) in links,
                    'other_zone': bool(tgt and tgt.zone != own.zone),
                    'sensitivity': tgt.target_sensitivity if tgt else 0.0})
    return out


def _windows(obs: List[dict], window: int):
    """Yield (start_tick, observations) for every sliding window of `window` ticks that contains observations."""
    ticks = sorted({o['tick'] for o in obs})
    j = 0
    for t in ticks:
        while obs[j]['tick'] <= t - window:
            j += 1
        k = j
        sel = []
        while k < len(obs) and obs[k]['tick'] <= t:
            sel.append(obs[k])
            k += 1
        yield t, sel


def infer_attack_kind(observations: List[dict], window: int = C.WINDOW_TICKS) -> Tuple[str, List[dict], List[str]]:
    """Recognise an attack kind from observed behaviour only.
    Returns (kind, matched_rules, evidence_request_ids). Rules are checked in a fixed order; see KIND_SIGNATURE."""
    obs = sorted(observations, key=lambda o: o['tick'])
    warm = [o for o in obs if o['tick'] <= C.WARMUP_TICKS]
    base_rate = len(warm) / C.WARMUP_TICKS if warm else 0.0
    base_bytes: Dict[str, float] = {}
    for k in {o['kind'] for o in warm}:
        xs = [o['bytes'] for o in warm if o['kind'] == k]
        base_bytes[k] = sum(xs) / len(xs)

    seen, reused = set(), []
    for o in obs:
        if o['nonce'] is not None and o['nonce'] in seen:
            reused.append(o)
        if o['nonce'] is not None:
            seen.add(o['nonce'])
    zk_fail, unexplained, rebooting = [], [], False
    for o in obs:                       # observation order = wire order
        if o['kind'] == 'reboot':
            rebooting = True
        if o['zk'] == 'FAIL':
            zk_fail.append(o)
            if not rebooting:
                unexplained.append(o)
        elif o['zk'] == 'PASS' and o['kind'] in ORDINARY_KINDS:
            rebooting = False           # first valid proof on ordinary traffic after the reboot: recovered

    def big(o):
        b = base_bytes.get(o['kind'])
        return b is not None and o['bytes'] > C.PAYLOAD_FACTOR * b

    def move(o):   # a suspicious move made with a valid proof
        if o['zk'] == 'FAIL' or o['kind'] == 'token_request':
            return False
        lateral = not o['linked'] and o['other_zone']
        sensitive = not o['linked'] and o['sensitivity'] >= C.SENSITIVE_TARGET
        return lateral or sensitive or big(o)

    def strong(o):
        return big(o)

    after = [o for o in obs if o['tick'] > C.WARMUP_TICKS]
    max_tok = max_ports = max_moves = 0
    tok_rids, port_rids, move_rids = [], [], []
    tok_rate = 0.0
    for _, sel in _windows(after, window):
        toks = [o for o in sel if o['kind'] == 'token_request']
        if len(toks) > max_tok:
            max_tok, tok_rids, tok_rate = len(toks), [o['rid'] for o in toks], len(sel) / window
        ports = {o['port'] for o in sel}
        if len(ports) > max_ports:
            max_ports, port_rids = len(ports), [next(o['rid'] for o in sel if o['port'] == p) for p in sorted(ports)]
        mv = [o for o in sel if move(o)]
        if len(mv) > max_moves:
            max_moves, move_rids = len(mv), [o['rid'] for o in mv]
    all_moves = [o for o in after if move(o)]
    strong_moves = [o for o in all_moves if strong(o)]
    rate_ok = base_rate > 0 and tok_rate >= SIG_RATE_FACTOR * base_rate

    rules = [
        {'kind': 'credential_replay', 'rule': 'reused nonces', 'value': len(reused), 'threshold': '≥ 1', 'hit': len(reused) >= 1},
        {'kind': 'credential_replay', 'rule': 'ZK failures not explained by a reboot', 'value': len(unexplained),
         'threshold': f'≥ {SIG_ZK_FAIL}', 'hit': len(unexplained) >= SIG_ZK_FAIL},
        {'kind': 'token_flood', 'rule': f'max token requests in a {window}-tick window', 'value': max_tok,
         'threshold': f'≥ {SIG_TOKENS_PER_WINDOW}', 'hit': max_tok >= SIG_TOKENS_PER_WINDOW},
        {'kind': 'token_flood', 'rule': 'window rate ÷ own warm-up rate', 'value': round(tok_rate / base_rate, 1) if base_rate else None,
         'threshold': f'≥ {SIG_RATE_FACTOR:g}', 'hit': rate_ok},
        {'kind': 'scanning', 'rule': f'max distinct ports in a {window}-tick window', 'value': max_ports,
         'threshold': f'> {SIG_PORTS_PER_WINDOW}', 'hit': max_ports > SIG_PORTS_PER_WINDOW},
        {'kind': 'compromised_valid', 'rule': f'max suspicious valid-proof moves in a {window}-tick window', 'value': max_moves,
         'threshold': f'≥ {SIG_MOVES_PER_WINDOW}', 'hit': max_moves >= SIG_MOVES_PER_WINDOW},
        {'kind': 'slow_low', 'rule': 'suspicious valid-proof moves in total (spread out)', 'value': len(all_moves),
         'threshold': f'≥ {SIG_SLOW_TOTAL} and max per window < {SIG_MOVES_PER_WINDOW}',
         'hit': len(all_moves) >= SIG_SLOW_TOTAL and max_moves < SIG_MOVES_PER_WINDOW},
        {'kind': 'slow_low', 'rule': f'exfiltration-sized payloads (> {C.PAYLOAD_FACTOR:g}× baseline) among them', 'value': len(strong_moves),
         'threshold': f'≥ {SIG_SLOW_STRONG}', 'hit': len(strong_moves) >= SIG_SLOW_STRONG},
    ]
    hit = lambda k: all(r['hit'] for r in rules if r['kind'] == k)  # noqa: E731
    if rules[0]['hit'] or rules[1]['hit']:
        kind, rids = 'credential_replay', [o['rid'] for o in reused + unexplained]
    elif hit('token_flood'):
        kind, rids = 'token_flood', tok_rids
    elif hit('scanning'):
        kind, rids = 'scanning', port_rids
    elif hit('compromised_valid'):
        kind, rids = 'compromised_valid', move_rids
    elif hit('slow_low'):
        kind, rids = 'slow_low', [o['rid'] for o in all_moves]
    elif zk_fail or all_moves:
        kind, rids = 'noisy_benign', [o['rid'] for o in zk_fail + all_moves]
    else:
        kind, rids = 'no_attack', []
    return kind, rules, rids[:8]


def behaviour_first_seen(observations: List[dict]) -> Dict[str, int]:
    """First tick after warm-up at which each kind of unusual behaviour appeared on the wire."""
    first: Dict[str, int] = {}
    seen = set()
    for o in sorted(observations, key=lambda x: x['tick']):
        tags = []
        if o['nonce'] is not None and o['nonce'] in seen:
            tags.append('nonce reuse')
        if o['nonce'] is not None:
            seen.add(o['nonce'])
        if o['tick'] <= C.WARMUP_TICKS:
            continue
        if o['zk'] == 'FAIL':
            tags.append('ZK failure')
        if not o['linked']:
            tags.append('no configured link')
        if not o['linked'] and o['other_zone']:
            tags.append('other zone')
        if not o['linked'] and o['sensitivity'] >= C.SENSITIVE_TARGET:
            tags.append('sensitive target')
        for t in tags:
            first.setdefault(t, o['tick'])
    return first


def infer_world(world, upto_tick: Optional[int] = None) -> Dict[str, dict]:
    """infer_attack_kind for every device of a world (observed data only)."""
    out = {}
    for d in sorted(world.devices):
        obs = wire_observations(world, d, upto_tick)
        kind, rules, rids = infer_attack_kind(obs)
        out[d] = {'kind': kind, 'label': KIND_LABEL[kind], 'rules': rules, 'evidence': rids,
                  'first_seen': behaviour_first_seen(obs), 'n_observed': len(obs)}
    return out


def multi_attacker_note(inferred: Dict[str, dict]) -> Optional[str]:
    attackers = [d for d, v in inferred.items() if v['kind'] not in HONEST_EXPECTED]
    if len(attackers) > 1:
        return f"{len(attackers)} devices match an attack signature in this run ({', '.join(attackers)}) — a multi-attacker situation."
    return None


# ----- scoring (ground truth used ONLY here, after the run) ------------------------------------------------
def score_inference(world, inferred: Optional[Dict[str, dict]] = None) -> List[dict]:
    """Inferred kind vs true agent type for every device. Honest devices count as correct if no attack is inferred."""
    inferred = inferred or infer_world(world)
    rows = []
    for d, gt in sorted(world.ground_truth.items()):
        truth = gt['agent']
        kind = inferred[d]['kind']
        correct = kind in HONEST_EXPECTED if truth in ('honest', 'noisy_honest') else kind == truth
        rows.append({'device': d, 'truth': truth, 'inferred': kind, 'correct': correct})
    return rows


# =========================================================================================================
# Part 2B: per-attacker comparison cards
# =========================================================================================================
def _bayes_crossings(world, device_id: str, since: int) -> List[dict]:
    """Records where the Bayesian band got worse (Trusted→Suspicious/Isolated, Suspicious→Isolated)."""
    order = {'Isolated': 0, 'Suspicious': 1, 'Trusted': 2}
    out = []
    for r in world.records:
        if r.get('kind') == 'request' and r['src'] == device_id and r['tick'] >= since and 'calc' in r:
            calc = r['calc']
            chain = [(calc['band_before'], calc['band_after'], calc['terms'])]
            if calc.get('step_up'):
                su = calc['step_up']
                chain.append((su['band_before'], su['band_after'], [su['term']]))
            for before, after, terms in chain:
                if order.get(after, 2) < order.get(before, 2):
                    out.append({'tick': r['tick'], 'rid': r['rid'], 'from': before, 'to': after,
                                'terms': [(lab, d) for lab, d in terms if lab != 'decay']})
    return out


def _engine_card(world, device_id: str, since: int, truth_start: Optional[int]) -> dict:
    from .comparison import policy_of
    from .metrics import device_timeline
    tl = device_timeline(world, device_id, since)
    card = {'first_flag_tick': tl['first_flag'], 'first_block_tick': tl['first_block'], 'detected': tl['first_block'] is not None}
    if world.engine == 'bayesian':
        blk = next((r for r in world.records if r.get('kind') == 'request' and r['src'] == device_id and r['tick'] >= since
                    and r.get('quarantined_until')), None)
        card['trust_at_detection'] = None if blk is None else (blk['calc'].get('step_up') or blk['calc'])['trust_after']
        lo = tl['first_flag'] if tl['first_flag'] is not None else since
        hi = tl['first_block'] if tl['first_block'] is not None else world.tick
        neg = defaultdict(float)
        for r in world.records:
            if r.get('kind') == 'request' and r['src'] == device_id and lo - C.WINDOW_TICKS < r['tick'] <= hi and 'calc' in r:
                terms = list(r['calc']['terms']) + ([r['calc']['step_up']['term']] if r['calc'].get('step_up') else [])
                for lab, d in terms:
                    if lab != 'decay' and d < 0:
                        neg[lab.split('×')[0] if not lab.endswith('!') else lab] += d
        card['top_terms'] = [f'{lab} {v:+.2f}' for lab, v in sorted(neg.items(), key=lambda kv: kv[1])[:3]]
        crossings = _bayes_crossings(world, device_id, since)
        reacted = Counter()
        for c in crossings:   # count each behaviour once per band change
            keys = {behaviour_of_term(lab) for lab, d in c['terms'] if d < 0}
            reacted.update(BEHAVIOURS[k]['label'] for k in keys if k)
        card['reacted_to'] = [f'{k} (in {n} of {len(crossings)} band changes)' for k, n in reacted.most_common()]
        card['crossings'] = [f"t={c['tick']} {c['from']}→{c['to']}" for c in crossings[:4]]
        card['reason'] = blk['reason'] if blk else '—'
    else:
        evs = [e for e in llm_evaluations(world, device_id) if e['tick'] >= since and e['ok']]
        blk = next((e for e in evs if e.get('quarantined_until')), None)
        flag = next((e for e in evs if e['llm_action'] in ('STEP-UP', 'BLOCK')), None)
        card['trust_at_detection'] = blk['llm_trust'] if blk else None
        card['top_terms'] = []
        use = blk or flag
        card['reason'] = use['reason'] if use else '—'
        card['factors'] = use['factors'] if use else []
        present = present_facts(use['context']) if use else set()
        card['reacted_to'] = [f"{BEHAVIOURS[CONCEPT_TO_BEHAVIOUR[c]]['label']} {'✔' if c in present else '✘ (not in window)'}"
                              for c in sorted(concepts_in_text(' '.join(card['factors']) + ' ' + card['reason'])) if c in CONCEPT_TO_BEHAVIOUR]
        card['crossings'] = [f"t={e['tick']} {e['decision_before']}→{e['llm_action']}" for e in evs
                             if e['llm_action'] != e['decision_before']][:4]
    if truth_start is not None:
        reqs = [r for r in world.records if r.get('kind') in ('request', 'dropped') and r['src'] == device_id and r['tick'] >= truth_start]
        card['malicious_allowed'] = sum(policy_of(r) == 'ALLOW' for r in reqs)
        card['ticks_to_step_up'] = None if tl['first_flag'] is None else tl['first_flag'] - truth_start
        card['ticks_to_block'] = None if tl['first_block'] is None else tl['first_block'] - truth_start
        honest = [d for d, gt in world.ground_truth.items() if gt['agent'] in ('honest', 'noisy_honest')]
        card['honest_falsely_blocked'] = sum(device_timeline(world, d, 0)['first_block'] is not None for d in honest)
    return card


def attack_cards(worlds: Dict[str, object], reveal: bool) -> List[dict]:
    """One card per device that matched an attack signature or that either engine blocked.
    With reveal=True the true kind, attack start and truth-based numbers are added, and true attackers are always shown."""
    primary = worlds.get('bayesian') or worlds.get('llm')
    inferred = infer_world(primary)
    blocked = {d for w in worlds.values() for r in w.records if r.get('quarantined_until') for d in [r['src']]}
    cards = []
    for d in sorted(primary.devices):
        gt = primary.ground_truth[d]
        is_attacker = gt['agent'] not in ('honest', 'noisy_honest')
        inf = inferred[d]
        if not (inf['kind'] not in HONEST_EXPECTED or d in blocked or (reveal and is_attacker)):
            continue
        start = gt['attack_start'] if (reveal and is_attacker and gt['attack_start'] is not None and gt['attack_start'] <= primary.tick) else None
        since = start if start is not None else C.WARMUP_TICKS + 1
        card = {'device': d, 'name': primary.devices[d].name, 'inferred': inf['kind'], 'inferred_label': inf['label'],
                'signature': KIND_SIGNATURE[inf['kind']], 'rules': inf['rules'], 'evidence': inf['evidence'],
                'first_seen': inf['first_seen'], 'engines': {e: _engine_card(w, d, since, start) for e, w in worlds.items()}}
        if reveal:
            card['truth'] = gt['agent']
            card['attack_start'] = gt['attack_start']
            card['inferred_correct'] = (inf['kind'] in HONEST_EXPECTED) if not is_attacker else inf['kind'] == gt['agent']
        cards.append(card)
    return cards


def attack_kind_rows(twin) -> List[dict]:
    """Per true attacker in a Both-mode run: inferred kind and each engine's outcome (for the batch table)."""
    worlds = dict(twin.worlds)
    primary = worlds.get('bayesian') or worlds.get('llm')
    inferred = infer_world(primary)
    rows = []
    n_honest = sum(gt['agent'] in ('honest', 'noisy_honest') for gt in primary.ground_truth.values())
    for d, gt in sorted(primary.ground_truth.items()):
        if gt['agent'] in ('honest', 'noisy_honest') or gt['attack_start'] is None or gt['attack_start'] > primary.tick:
            continue
        row = {'device': d, 'attack_kind': gt['agent'], 'inferred': inferred[d]['kind'], 'inferred_correct': inferred[d]['kind'] == gt['agent'],
               'n_honest': n_honest}
        for e, w in worlds.items():
            c = _engine_card(w, d, gt['attack_start'], gt['attack_start'])
            row.update({f'{e}_detected': c['detected'], f'{e}_ticks_to_block': c['ticks_to_block'],
                        f'{e}_ticks_to_step_up': c['ticks_to_step_up'], f'{e}_malicious_allowed': c['malicious_allowed'],
                        f'{e}_honest_falsely_blocked': c['honest_falsely_blocked'],
                        f'{e}_triggers': [t.rsplit(' (', 1)[0].rsplit(' ✔', 1)[0].rsplit(' ✘', 1)[0] for t in c['reacted_to']]})
        rows.append(row)
    return rows


# =========================================================================================================
# Part 3: the alarm system
# =========================================================================================================
def request_alarm_rows(world, record: dict, llm_world=None, reveal: bool = False) -> List[dict]:
    """For one request: how each sensor's alarm was produced and how each engine used it.
    The examiner-only columns (truth, rate used, random draw) are added only when reveal=True."""
    rid = record['rid']
    draws = world.sensor_draws.get(rid, {})
    ctx = None
    if llm_world is not None:
        lr = next((r for r in llm_world.records if r.get('rid') == rid), None)
        if lr is not None:
            evs = [e for e in llm_evaluations(llm_world, record['src']) if e['tick'] >= record['tick']]
            ctx = evs[0] if evs else None
    rows = []
    for name, (tp, fp) in C.SENSORS.items():
        fired = record.get('alarms', {}).get(name)
        row = {'sensor': f'{name} ({SENSOR_NAMES[name]})', 'TP rate': tp, 'FP rate': fp,
               'fired?': '—' if fired is None else ('fired' if fired else 'quiet')}
        if reveal and name in draws:
            mal = world.request_truth.get(rid)
            row['request really malicious? (examiner)'] = 'yes' if mal else 'no'
            row['probability used p (examiner)'] = f"{draws[name]['p']:g} ({'TP' if mal else 'FP'} rate)"
            row['random draw u (examiner)'] = round(draws[name]['u'], 4)
            row['u < p ?'] = f"{draws[name]['u']:.4f} {'<' if draws[name]['u'] < draws[name]['p'] else '≥'} {draws[name]['p']:g} → {'fires' if draws[name]['u'] < draws[name]['p'] else 'quiet'}"
        if fired is None:
            row['Bayesian term'] = '— (request dropped, no update)'
        else:
            row['Bayesian term'] = (f"{C.SENSOR_WEIGHT:g} × ln({fp:g}/{tp:g}) = {sensor_term(name, True):+.4f}" if fired else
                                    f"{C.SENSOR_WEIGHT:g} × ln({1 - fp:g}/{1 - tp:g}) = {sensor_term(name, False):+.4f}")
        if llm_world is None:
            row['how the LLM sees it'] = '— (LLM engine not running)'
        elif ctx is None:
            row['how the LLM sees it'] = 'not evaluated yet after this request'
        else:
            w = ctx['context']['window_facts'].get('sensor_alarm_counts', {}).get(name, 0)
            ref = ctx['context']['sensor_reference'][name]
            row['how the LLM sees it'] = (f"in the evaluation at t={ctx['tick']}: sensor_alarm_counts.{name} = {w} "
                                          f"(reference TP {ref['true_positive_rate']}, FP {ref['false_positive_rate']}); "
                                          f"this request {'adds 1' if fired else 'adds 0'} if still in the window")
        rows.append(row)
    return rows


def alarm_combinations(world=None) -> List[dict]:
    """All 8 combinations of the three alarms: the combined Bayesian term (and how often each happened, if a world is given)."""
    names = list(C.SENSORS)
    counts = Counter()
    if world is not None:
        for r in world.records:
            if r.get('kind') == 'request':
                counts[tuple(bool(r['alarms'].get(n)) for n in names)] += 1
    rows = []
    for combo in sorted(product([True, False], repeat=len(names)), key=lambda c: -sum(c)):
        parts = [f"{C.SENSOR_WEIGHT:g}·ln({C.SENSORS[n][1]:g}/{C.SENSORS[n][0]:g})" if f else
                 f"{C.SENSOR_WEIGHT:g}·ln({1 - C.SENSORS[n][1]:g}/{1 - C.SENSORS[n][0]:g})" for n, f in zip(names, combo)]
        total = sum(sensor_term(n, f) for n, f in zip(names, combo))
        row = {'alarms fired': ' + '.join(n for n, f in zip(names, combo) if f) or 'none', 'how many fired': sum(combo),
               'combined Bayesian term': round(total, 4), 'calculation': ' + '.join(parts) + f' = {total:+.4f}'}
        if world is not None:
            row['requests with this pattern'] = counts.get(combo, 0)
        rows.append(row)
    return rows


def alarm_stats(world, llm_world=None) -> List[dict]:
    """Per sensor, after the run, from ground truth: observed TP/FP rates vs configured, precision,
    Bayesian log-odds contributed per device, and how often LLM factors mentioned alarms."""
    rows = []
    rids = list(world.sensor_draws)
    mal = [r for r in rids if world.request_truth.get(r)]
    ben = [r for r in rids if not world.request_truth.get(r)]
    per_dev = defaultdict(lambda: defaultdict(float))
    if world.engine == 'bayesian':
        for r in world.records:
            if r.get('kind') == 'request':
                for lab, d in r['terms']:
                    if lab[:-1] in C.SENSORS and lab[-1] in '!·':
                        per_dev[lab[:-1]][r['src']] += d
    evals = [e for e in (llm_world or world).records if e.get('kind') == 'llm_eval' and e['ok']] if (llm_world is not None or world.engine == 'llm') else []
    alarm_mentions = sum('sensor_alarm' in concepts_in_text(' '.join(e['factors']) + ' ' + e['reason']) for e in evals)
    for name, (tp, fp) in C.SENSORS.items():
        fired = {r for r in rids if world.sensor_draws[r][name]['u'] < world.sensor_draws[r][name]['p']}
        tp_n, fp_n = len(fired & set(mal)), len(fired & set(ben))
        dev = per_dev.get(name, {})
        rows.append({'sensor': name, 'requests (all, incl. dropped)': len(rids), 'malicious requests': len(mal),
                     'alarms on malicious': tp_n, 'observed TP rate': tp_n / len(mal) if mal else None, 'configured TP rate': tp,
                     'alarms on benign': fp_n, 'observed FP rate': fp_n / len(ben) if ben else None, 'configured FP rate': fp,
                     'alarm precision': tp_n / (tp_n + fp_n) if tp_n + fp_n else None,
                     'Bayesian log-odds per device': {d: round(v, 3) for d, v in sorted(dev.items())},
                     'LLM evaluations whose factors mention alarms': (f'{alarm_mentions}/{len(evals)}' if evals else '—')})
    return rows


def alarm_rows_for_csv(world, scenario: str, seed: int) -> List[dict]:
    """Flat per-sensor rows for results/alarm_stats.csv."""
    out = []
    for s in alarm_stats(world):
        out.append({'scenario': scenario, 'seed': seed, 'sensor': s['sensor'], 'requests': s['requests (all, incl. dropped)'],
                    'malicious_requests': s['malicious requests'], 'observed_tp_rate': s['observed TP rate'],
                    'configured_tp_rate': s['configured TP rate'], 'observed_fp_rate': s['observed FP rate'],
                    'configured_fp_rate': s['configured FP rate'], 'precision': s['alarm precision']})
    return out


def _ms(xs) -> Tuple[Optional[float], Optional[float]]:
    xs = [float(x) for x in xs if x is not None]
    if not xs:
        return None, None
    m = sum(xs) / len(xs)
    return m, math.sqrt(sum((x - m) ** 2 for x in xs) / len(xs))


def aggregate_attack_kinds(rows: List[dict], engines: Sequence[str] = ('bayesian', 'llm')) -> List[dict]:
    """One row per true attack kind over all (scenario, seed, attacker) rows: mean ± std, medians, top triggers."""
    out = []
    for kind in sorted({r['attack_kind'] for r in rows}):
        sel = [r for r in rows if r['attack_kind'] == kind]
        acc, acc_sd = _ms([float(r['inferred_correct']) for r in sel])
        row = {'attack_kind': kind, 'behaviour signature': KIND_SIGNATURE.get(kind, ''), 'attacker_runs': len(sel),
               'inferred_kind_accuracy': acc, 'inferred_kind_accuracy_std': acc_sd}
        for e in engines:
            if f'{e}_detected' not in sel[0]:
                continue
            det, det_sd = _ms([float(bool(r[f'{e}_detected'])) for r in sel])
            ttd = [r[f'{e}_ticks_to_block'] for r in sel if r[f'{e}_ticks_to_block'] is not None]
            allowed, allowed_sd = _ms([r[f'{e}_malicious_allowed'] for r in sel])
            fpr, fpr_sd = _ms([r[f'{e}_honest_falsely_blocked'] / r['n_honest'] for r in sel if r['n_honest']])
            trig = Counter(t for r in sel for t in r[f'{e}_triggers'])
            row.update({f'{e}_detection_rate': det, f'{e}_detection_rate_std': det_sd,
                        f'{e}_median_ticks_to_block': median(ttd) if ttd else None, f'{e}_max_ticks_to_block': max(ttd) if ttd else None,
                        f'{e}_malicious_allowed_mean': allowed, f'{e}_malicious_allowed_std': allowed_sd,
                        f'{e}_honest_fpr_mean': fpr, f'{e}_honest_fpr_std': fpr_sd,
                        f'{e}_top_triggers': '; '.join(f'{t} ({n})' for t, n in trig.most_common(3)) or '—'})
        out.append(row)
    return out
