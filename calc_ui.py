"""Trust calculation tab (Live mode): how each engine turned the observed behaviour into trust.

Three sections:
  1. Request calculation — the Bayesian per-request maths next to the LLM evaluation that decided the same request.
  2. Attacks by behaviour — attack kinds recognised from observed behaviour, and how each engine reacted.
  3. Alarm system — how IDS / SIEM / BM alarms are produced and how each engine uses them.
Everything is read from the worlds' own logs (core/behaviour_analysis.py). Examiner-only data (ground
truth, sensor random draws) appears only when "Reveal ground truth" is on.
"""
from __future__ import annotations

import json

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core import behaviour_analysis as BA
from core import live_config as C
from core.live_llm import LIVE_SYSTEM_PROMPT, PENDING
from ui import charts
from ui.theme import ACCENT, ENGINE_COLORS, STATUS_COLORS

ENGINE_TITLE = {'bayesian': 'Bayesian', 'llm': 'LLM'}


def _f(x, d=4):
    if x is None:
        return '—'
    if isinstance(x, bool):
        return 'yes' if x else 'no'
    if isinstance(x, float):
        return f'{x:.{d}f}'
    return str(x)


def llm_honesty_note(world) -> None:
    """Say clearly when the LLM numbers did not come from a real model."""
    if world is None:
        return
    name = world.controller.model_name
    if 'Fake' in name:
        st.warning(f'LLM numbers in this run come from **{name}** — a rule-based stand-in, NOT a real language model. '
                   'They show how the pipeline works, not how well an LLM performs.')
    else:
        st.caption(f'LLM: **{name}**. If this endpoint is a local stub or test server, these are not real-model results.')


def _df(rows, **kw):
    if not rows:
        return pd.DataFrame([{'info': 'nothing to show'}])
    return pd.DataFrame([{k: (json.dumps(v) if isinstance(v, (list, dict, tuple)) else v) for k, v in r.items()} for r in rows], **kw)


# ----- section 1 -----------------------------------------------------------------------------------------
def _request_options(world, device_id):
    recs = [r for r in world.records if r.get('kind') in ('request', 'dropped') and r['src'] == device_id]
    return list(reversed(recs))


def render_bayes_column(world, record) -> None:
    st.subheader('Bayesian: per request')
    if record is None:
        st.info('This request id does not exist in the Bayesian world.')
        return
    if record.get('kind') == 'dropped':
        st.warning(f"DROPPED in the Bayesian world — {record['reason']}. A quarantined device's links are cut, so no trust update happens.")
        return
    s = BA.bayes_steps(record, world)
    th = s['thresholds']
    lines = [
        f"1. **Trust before:** {s['trust_before']:.4f} → log-odds L₀ = ln(p/(1−p)) = ln({s['trust_before']:.4f}/{1 - s['trust_before']:.4f}) = **{s['L0']:+.4f}**",
    ]
    if s['decay']:
        extra = ''
        if s['decay_ticks'] is not None:
            extra = (f" over {s['decay_ticks']} tick(s) since the previous update (trust then {s['decay_from_trust']:.4f})"
                     + (' ✔ re-applied and matches' if s['decay_check'] else ' — a probation reset happened in between'))
        lines.append(f"2. **Decay (already applied at the start of the tick):** L ← {C.DECAY_LAMBDA}·L + {1 - C.DECAY_LAMBDA:.2f}·logit({C.LIVE_PRIOR}) "
                     f"(logit({C.LIVE_PRIOR}) = {s['logit_prior']:+.4f}); it moved L by **{s['decay']:+.4f}**, from {s['L_before_decay']:+.4f} to {s['L0']:+.4f}{extra}")
    else:
        lines.append('2. **Decay:** none since the previous update of this device (same tick, or a probation reset).')
    lines.append(f"3. **Observed behaviour → evidence:** {len([r for r in s['evidence'] if r['kind'] != 'sensor'])} term(s) — table below")
    lines.append(f"4. **Sensor terms:** {len([r for r in s['evidence'] if r['kind'] == 'sensor'])} terms (weight {C.SENSOR_WEIGHT}) — in the same table")
    lines.append(f"5. **Sum of terms** = {s['sum_terms']:+.4f} → L₁ = {s['L0']:+.4f} + ({s['sum_terms']:+.4f}) = **{s['L1']:+.4f}** → "
                 f"trust = 1/(1+e^−L₁) = {s['sigmoid']:.4f}" + (f" → clamped to **{s['trust_after']:.4f}**" if s['clamped'] else f" = **{s['trust_after']:.4f}**"))
    hyst = (f"held at **{s['band_after']}** by hysteresis (must pass the threshold by ±{th['hysteresis']})" if s['hysteresis_held']
            else f"→ **{s['band_after']}**")
    lines.append(f"6. **Band:** thresholds ALLOW ≥ {th['allow']:.2f}, BLOCK < {th['block']:.2f}. Raw band {s['raw_band']} {hyst} "
                 f"(was {s['band_before']}). Policy: ZK {s['zk']}{', nonce REPLAYED' if s['replay'] else ''}"
                 f"{', on probation' if s['probation'] else ''} → **{s['policy_action']}**")
    su = s['step_up']
    if su:
        lines.append(f"7. **STEP-UP:** fresh proof {su['zk']}, MFA {su['mfa']} → `{su['term']}` = {C.EVENT_WEIGHT}·ln({su['P(e|legit)']:g}/{su['P(e|attacker)']:g}) "
                     f"= {su['term value']:+.4f}; {su['check_text']} → band {su['band_after']} → **{s['final_action']}**")
    else:
        lines.append('7. **STEP-UP:** not needed for this request.')
    lines.append(f"8. **Quarantine / probation:** " + (f"quarantined until t={s['quarantined_until']} (links cut for {C.QUARANTINE_TICKS} ticks)"
                                                      if s['quarantined_until'] else ('on probation — every request needs a step-up' if s['probation'] else 'no change')))
    st.markdown('\n'.join(lines))
    ev = pd.DataFrame([{'term': r['term'], 'kind': r['kind'], 'observed fact': r['observed fact'], 'P(e|legit)': r['P(e|legit)'],
                        'P(e|attacker)': r['P(e|attacker)'], 'ratio': round(r['ratio'], 4), 'ln(ratio)': round(r['ln(ratio)'], 4),
                        'weight': r['weight'], 'term': r['term'], 'term value': round(r['term value'], 4)} for r in s['evidence']])
    st.dataframe(ev.loc[:, ~ev.columns.duplicated()], hide_index=True, use_container_width=True)
    (st.success if s['check_ok'] else st.error)('**Step 9 — check it yourself:** ' + s['check_text'])
    items = BA.waterfall(s)
    st.caption('Log-odds waterfall: L₀, then each term, then the final L')
    fig = charts.waterfall([i[0] for i in items], [i[1] for i in items], s['L0'], th['allow'], th['block'], BA.logit)
    st.plotly_chart(fig, use_container_width=True, key=f"wf_{record['rid']}")


def render_llm_column(world, record) -> None:
    st.subheader('LLM: per evaluation')
    if record is None:
        st.info('This request id does not exist in the LLM world.')
        return
    if record.get('kind') == 'dropped':
        st.warning(f"DROPPED in the LLM world — {record['reason']}.")
        return
    deciding, latest = BA.deciding_evaluation(world, record)
    if deciding is None:
        st.markdown(f"1. **Which evaluation applies:** none yet — the device is **{PENDING}**. Default rule: "
                    f"{C.PENDING_DEFAULT_ACTION} if the ZK check passes. This request: **{record['action']}** ({record['reason']}).")
        if latest is not None and not latest['ok']:
            st.error(f"Latest attempt at t={latest['tick']}: LLM_ERROR ({latest['error']}) — {latest.get('message', '')}")
        nxt = next((e for e in BA.llm_evaluations(world, record['src']) if e['tick'] >= record['tick']), None)
        if nxt is None:
            return
        st.caption(f"Showing the next evaluation instead (t={nxt['tick']}), which this request fed into.")
        deciding = nxt
    else:
        st.markdown(f"1. **Which evaluation applies:** the LLM evaluation at **t={deciding['tick']}** (trigger: *{deciding['trigger']}*). "
                    f"Its decision **{deciding['llm_action']}** is held until the next evaluation. This request: **{record['action']}** ({record['reason']}).")
        if latest is not None and latest is not deciding and not latest['ok']:
            st.error(f"A later attempt at t={latest['tick']} failed: LLM_ERROR ({latest['error']}) — the t={deciding['tick']} decision was kept.")
    trace = BA.window_trace(world, deciding)
    st.markdown(f"2. **Observed behaviour → facts** ({len(trace)} fields, rebuilt from World L's own records; ✔ = equals what was sent):")
    st.dataframe(pd.DataFrame([{'fact': t['fact'], 'value sent': json.dumps(t['value sent to LLM']), 'rebuilt ✔': '✔' if t['matches'] else '✘',
                                'how it is computed': t['how it is computed'], 'n requests': t['n contributing'],
                                'contributing request ids': ', '.join(t['contributing requests'][:12]) + (' …' if len(t['contributing requests']) > 12 else '')}
                               for t in trace]), hide_index=True, use_container_width=True, height=360)
    st.markdown('3. **What was sent, and what came back**')
    with st.expander('Exact context JSON sent to the LLM'):
        st.json(deciding['context'])
    with st.expander('System prompt'):
        st.code(LIVE_SYSTEM_PROMPT, language='text')
    with st.expander('Raw reply (exactly as received)', expanded=True):
        try:
            st.code(deciding.get('raw_reply') or '—', language='json', wrap_lines=True)
        except TypeError:   # older Streamlit without wrap_lines
            st.code(deciding.get('raw_reply') or '—', language='json')
    if deciding['ok']:
        clamped = BA.clamped_fields(deciding)
        st.markdown(f"Parsed after strict validation: trust **{deciding['llm_trust']:.4f}**, action **{deciding['llm_action']}**, "
                    f"confidence {_f(deciding['confidence'], 2)}" + (f" · clamped into [0, 1]: {', '.join(clamped)}" if clamped else ' · nothing clamped'))
    else:
        st.error(f"LLM_ERROR ({deciding['error']}): {deciding.get('message', '')}")
    st.markdown('4. **Factor grounding** — does each factor refer to a fact really present in the window?')
    st.dataframe(_df(BA.factor_grounding(deciding)), hide_index=True, use_container_width=True)
    st.caption(f"5. Latency {deciding['latency_ms']:.1f} ms · model {deciding.get('llm_model', '—')}")


def render_trust_markers(worlds, device_id, tick) -> None:
    """Bayesian trust (solid), LLM trust (dashed), LLM evaluations (diamonds) and the first tick of each new behaviour."""
    fig = go.Figure()
    b, l = worlds.get('bayesian'), worlds.get('llm')
    for world, dash, name in ((b, 'solid', 'Bayesian trust'), (l, 'dash', 'LLM trust')):
        if world is None:
            continue
        hist = pd.DataFrame(world.trust_history)
        if not hist.empty and device_id in hist:
            fig.add_trace(go.Scatter(x=hist['tick'], y=hist[device_id], mode='lines', name=name, connectgaps=False,
                                     line={'dash': dash, 'width': 1.8, 'color': ENGINE_COLORS['bayesian' if world is b else 'llm']}))
    if l is not None:
        evs = BA.llm_evaluations(l, device_id)
        ok = [e for e in evs if e['ok']]
        fig.add_trace(go.Scatter(x=[e['tick'] for e in ok], y=[e['llm_trust'] for e in ok], mode='markers', name='LLM evaluation',
                                 marker={'symbol': 'diamond', 'size': 7, 'color': ENGINE_COLORS['llm']},
                                 text=[f"{e['trigger']}: {e['llm_action']}" for e in ok], hovertemplate='t = %{x} %{text}<extra></extra>'))
        bad = [e for e in evs if not e['ok']]
        if bad:
            fig.add_trace(go.Scatter(x=[e['tick'] for e in bad], y=[0.02] * len(bad), mode='markers', name='LLM_ERROR',
                                     marker={'symbol': 'x', 'color': STATUS_COLORS['Isolated']}))
    firsts = BA.behaviour_first_ticks(b, device_id) if b is not None else BA.llm_fact_first_ticks(l, device_id)
    by_tick = {}
    for key, t in firsts.items():
        by_tick.setdefault(t, []).append(BA.BEHAVIOURS[key]['label'])
    for i, (t, labels) in enumerate(sorted(by_tick.items())):
        fig.add_vline(x=t, line_dash='dot', line_width=1, line_color=STATUS_COLORS['Suspicious'])
        fig.add_annotation(x=t, y=0.97 - 0.07 * (i % 4), text=f't = {t}: ' + ', '.join(labels), showarrow=False, xanchor='left',
                           font={'size': 10, 'color': STATUS_COLORS['Suspicious']})
    fig.add_vline(x=tick, line_color=ACCENT, line_width=1.5, annotation_text='selected request', annotation_font={'size': 10})
    ctl = next(iter(worlds.values())).controller
    charts.add_thresholds(fig, ctl.allow_threshold, ctl.block_threshold)
    fig.update_yaxes(range=[0, 1])
    st.caption('Dotted amber lines: first tick each new behaviour appeared.')
    st.plotly_chart(charts.style(fig, 360, ytitle='trust', xtitle='tick'), use_container_width=True, key='calc_markers')


def render_section_request(twin, device_id, rid) -> None:
    worlds = twin.worlds
    rec = {e: next((r for r in w.records if r.get('rid') == rid), None) for e, w in worlds.items()}
    st.caption('The same request id exists in every world (identical traffic). Left: how the Bayesian controller updated trust for '
               'THIS request. Right: the LLM evaluation whose decision applied to it — the LLM does not calculate per request.')
    cols = st.columns(len(worlds))
    for col, (engine, world) in zip(cols, worlds.items()):
        with col:
            (render_bayes_column if engine == 'bayesian' else render_llm_column)(world, rec[engine])
    tick = int(rid.split('-')[0])
    st.subheader('Same behaviour, two treatments')
    st.caption(f'Every behaviour seen on {device_id} in the {C.WINDOW_TICKS} ticks up to t={tick}: the Bayesian terms it produced, '
               'the LLM context fields it appeared in, and whether the LLM named it.')
    st.dataframe(_df(BA.same_behaviour_table(device_id, tick, worlds.get('bayesian'), worlds.get('llm'))), hide_index=True, use_container_width=True)
    st.subheader('How the two calculations differ')
    st.dataframe(_df(BA.differences_table(getattr(worlds.get('llm'), 'controller', None).every_n if worlds.get('llm') else C.LLM_EVERY_N,
                                          worlds.get('bayesian'), worlds.get('llm'))), hide_index=True, use_container_width=True)
    st.subheader('Trust over time')
    render_trust_markers(worlds, device_id, tick)


# ----- section 2 -----------------------------------------------------------------------------------------
def render_section_attacks(twin, reveal: bool) -> None:
    worlds = dict(twin.worlds)
    st.caption('Attack kinds are recognised from OBSERVED behaviour only (core/behaviour_analysis.py → infer_attack_kind): request kinds, '
               'destinations, ports, payload sizes, ZK results, nonces and the public topology. The true agent type is never used for this; '
               'with Reveal on it is shown next to the guess, for scoring only.')
    llm_honesty_note(worlds.get('llm'))
    with st.expander('The behaviour signatures (rules, checked in this order)'):
        st.dataframe(pd.DataFrame([{'attack kind': BA.KIND_LABEL[k], 'signature (observed facts only)': v} for k, v in BA.KIND_SIGNATURE.items()]),
                     hide_index=True, use_container_width=True)
    cards = BA.attack_cards(worlds, reveal)
    note = BA.multi_attacker_note(BA.infer_world(next(iter(worlds.values()))))
    if note:
        st.info(note)
    if not cards:
        st.info('No device matches an attack signature and no device was blocked so far.' + ('' if reveal else ' (Reveal is off.)'))
        return
    summary = []
    for c in cards:
        row = {'device': c['device'], 'recognised as': c['inferred_label']}
        if reveal:
            row.update({'true kind (examiner)': c['truth'], 'recognised correctly': '✔' if c['inferred_correct'] else '✘'})
        for e, ec in c['engines'].items():
            row[f'{ENGINE_TITLE[e]} detected'] = 'yes' if ec['detected'] else 'no'
            if reveal:
                row[f'{ENGINE_TITLE[e]} ticks to BLOCK'] = _f(ec.get('ticks_to_block'))
        summary.append(row)
    st.markdown('**This run, attack by attack**')
    st.dataframe(pd.DataFrame(summary), hide_index=True, use_container_width=True)
    for c in cards:
        with st.container(border=True):
            head = f"**{c['device']} — {c['name']}** · recognised as **{c['inferred_label']}**"
            if reveal:
                head += f" · true kind (examiner): `{c['truth']}`" + (f" from t={c['attack_start']}" if c.get('attack_start') is not None else '') \
                        + (' ✔' if c['inferred_correct'] else ' ✘')
            st.markdown(head)
            st.caption('Signature: ' + c['signature'])
            a, b = st.columns([3, 2])
            a.dataframe(pd.DataFrame([{'rule': r['rule'], 'value': _f(r['value']), 'needs': r['threshold'], 'met': '✔' if r['hit'] else '·',
                                       'for': BA.KIND_LABEL[r['kind']]} for r in c['rules']]), hide_index=True, use_container_width=True)
            b.markdown('**Behaviour seen** (first tick after warm-up)')
            b.write(', '.join(f'{k} t={t}' for k, t in sorted(c['first_seen'].items(), key=lambda kv: kv[1])) or 'nothing unusual')
            b.markdown('**Example requests**')
            b.write(', '.join(c['evidence']) or '—')
            cols = st.columns(len(c['engines']))
            for col, (e, ec) in zip(cols, c['engines'].items()):
                with col:
                    st.markdown(f'**{ENGINE_TITLE[e]} world**')
                    rows = [('detected (quarantined)', 'yes' if ec['detected'] else 'no'),
                            ('first STEP-UP / BLOCK tick', f"{_f(ec['first_flag_tick'])} / {_f(ec['first_block_tick'])}"),
                            ('trust at detection', _f(ec['trust_at_detection']))]
                    if reveal and 'ticks_to_block' in ec:
                        rows += [('ticks from attack start to STEP-UP / BLOCK', f"{_f(ec['ticks_to_step_up'])} / {_f(ec['ticks_to_block'])}"),
                                 ('requests allowed while malicious', _f(ec['malicious_allowed'])),
                                 ('honest devices falsely blocked (this run)', _f(ec['honest_falsely_blocked']))]
                    if e == 'bayesian':
                        rows += [('top 3 terms behind the drop', '; '.join(ec['top_terms']) or '—'),
                                 ('band changes', '; '.join(ec['crossings']) or '—'), ('reason', ec['reason'])]
                    else:
                        rows += [('LLM reason', ec['reason']), ('LLM factors', '; '.join(ec.get('factors') or []) or '—'),
                                 ('decision changes', '; '.join(ec['crossings']) or '—')]
                    rows.append(('reacted to', '; '.join(ec['reacted_to']) or '—'))
                    st.dataframe(pd.DataFrame(rows, columns=['', 'value']), hide_index=True, use_container_width=True)


# ----- section 3 -----------------------------------------------------------------------------------------
def render_section_alarms(twin, rid, reveal: bool) -> None:
    worlds = twin.worlds
    env = twin.primary
    rec = next((r for r in env.records if r.get('rid') == rid), None)
    st.subheader(f'How the alarms for request {rid} were produced')
    tp, fp = C.SENSORS['IDS']
    st.info(f'Each sensor fires if a random number **u** (between 0 and 1) is below its rate **p**. For a malicious request p is the '
            f'true-positive rate (IDS {tp}); otherwise it is the false-positive rate (IDS {fp}). The controllers only receive '
            'fired / quiet — never u, p or whether the request was malicious.')
    if rec is not None:
        brec = next((r for r in worlds['bayesian'].records if r.get('rid') == rid), None) if 'bayesian' in worlds else None
        rows = BA.request_alarm_rows(env, brec or rec, worlds.get('llm'), reveal)
        st.dataframe(_df(rows), hide_index=True, use_container_width=True)
        if not reveal:
            st.caption('Turn on **Reveal ground truth (examiner view)** to see whether the request was malicious, the rate used and the random draw u.')
    st.subheader('Every combination of alarms and its combined Bayesian term')
    st.dataframe(_df(BA.alarm_combinations(worlds.get('bayesian'))), hide_index=True, use_container_width=True)
    st.caption('A quiet sensor adds a small positive term; a fired sensor a negative one. The LLM gets alarm counts and the reference rates '
               'instead, and decides itself how much they matter.')
    st.subheader('Whole-run alarm statistics (examiner view)')
    if not reveal:
        st.caption('These numbers need ground truth (which requests were malicious) — turn on Reveal to compute them.')
    else:
        stats = BA.alarm_stats(worlds['bayesian'] if 'bayesian' in worlds else env, worlds.get('llm'))
        st.dataframe(pd.DataFrame([{'sensor': s['sensor'], 'requests': s['requests (all, incl. dropped)'], 'malicious': s['malicious requests'],
                                    'observed TP rate': _f(s['observed TP rate'], 3), 'configured TP': s['configured TP rate'],
                                    'observed FP rate': _f(s['observed FP rate'], 3), 'configured FP': s['configured FP rate'],
                                    'precision': _f(s['alarm precision'], 3),
                                    'net Bayesian log-odds per device (Σ of its terms; quiet = +, fired = −)':
                                        ', '.join(f'{d} {v:+.2f}' for d, v in s['Bayesian log-odds per device'].items()) or '—',
                                    'LLM evals mentioning alarms': s['LLM evaluations whose factors mention alarms']} for s in stats]),
                     hide_index=True, use_container_width=True)
    st.warning('**Limitation.** The sensors are driven by ground truth (they fire with a TP rate on malicious requests and an FP rate on '
               'benign ones) and are treated as independent. Real sensors react to signatures and are correlated. To see how much the '
               'Bayesian results depend on them, run the ablation `python experiments/run_batch.py --scenario all --seeds 20 --sensor-weight 0` '
               '(guide §9.2): every attacker is still caught from behaviour alone, but false quarantines of honest devices rise.')


# ----- the tab -----------------------------------------------------------------------------------------
def render_calculation_tab(twin, reveal: bool) -> None:
    if st.session_state.get('live_running'):
        st.info('Pause the run to inspect the calculations.')
        return
    if twin.tick == 0:
        st.info('Run (or Step) the simulation first.')
        return
    primary = twin.primary
    c1, c2 = st.columns([1, 3])
    device_id = c1.selectbox('Device', sorted(primary.devices), key='calc_device',
                             format_func=lambda d: f'{d} — {primary.devices[d].name}')
    options = _request_options(primary, device_id)
    if not options:
        st.info('This device has not sent any request yet.')
        return
    labels = {r['rid']: f"{r['rid']} · {r['event_type']} → {r['dst']} · {r['action']}" for r in options}
    if st.session_state.get('calc_rid') not in labels:
        st.session_state.pop('calc_rid', None)
    default = next((i for i, r in enumerate(options) if r.get('kind') == 'request'), 0)   # newest request that was processed
    rid = c2.selectbox('Request (newest first; DROPPED = links cut by quarantine)', list(labels), index=default,
                       format_func=labels.get, key='calc_rid')
    t1, t2, t3 = st.tabs(['1 · Request calculation', '2 · Attacks by behaviour', '3 · Alarm system'])
    with t1:
        render_section_request(twin, device_id, rid)
    with t2:
        render_section_attacks(twin, reveal)
    with t3:
        render_section_alarms(twin, rid, reveal)
