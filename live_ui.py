"""Streamlit UI for Live Network mode (kept separate so app.py's Classic code stays untouched).

Run modes: Bayesian only (World B), LLM only (World L), Both — isolated comparison (B and L side by side).
The TwinSimulation lives in st.session_state; a st.fragment re-runs on a timer while running.
Every panel reads ONE world's own data; the LLM columns never show Bayesian values.
"""
from __future__ import annotations

import html
import os
import re
from typing import List

import pandas as pd
import streamlit as st

from core import live_config as C
from core.agents import ATTACKER_TYPES
from core.controller import BAND_ACTION
from core.gemini_engine import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL, load_dotenv, mask_key
from core.live_llm import LIVE_SYSTEM_PROMPT, TEST_CONTEXT, LLMController, make_live_evaluator, parse_llm_output
from core.live_sim import LiveSimulation, list_scenarios, load_scenario, render_record
from core.metrics import compute_metrics, log_rows, metrics_json, summary_markdown, to_csv, trust_csv
from core.comparison import compare, repeat_consistency, summary_table, write_exports
from core.twin import MODE_LABELS, TwinSimulation
from calc_ui import render_calculation_tab
from ui import charts
from ui.theme import (ENGINE_COLORS, STATUS_COLORS, fmt_ms, fmt_pct, fmt_tick, fmt_trust, kpi_row, status_badge, status_color,
                      status_line, step_row_html)

CURRENT_NETWORK = 'Current network (from the builder)'
TERMINAL_LINES = 300
WORLD_TITLE = {'bayesian': 'Bayesian world', 'llm': 'LLM world'}
FLOW_STAGES = ['Request packet', 'Topology gate', 'Zero-Knowledge', 'Behavior inspection',
               'Trust calculation', 'Policy engine', 'Final decision']
MAX_TRUST_LINES = 6


# ----- state -----------------------------------------------------------------------
def scenario_options() -> List[str]:
    """Scenario choices shown in the sidebar: the presets, then the network from the builder."""
    return list_scenarios() + [CURRENT_NETWORK]


PROVIDERS = ['Ollama (local)', 'Google Gemini (API)', 'OpenAI-compatible API']
PROVIDER_IDS = {'Ollama (local)': 'ollama', 'Google Gemini (API)': 'gemini', 'OpenAI-compatible API': 'openai_compatible'}
GEMINI_NOTE = "Simulated device data is sent to Google's API. No real data is used."


def provider_id(label: str) -> str:
    return PROVIDER_IDS.get(label, 'openai_compatible')


def llm_settings() -> dict:
    """Current LLM settings from the sidebar widgets (with defaults before they are drawn)."""
    g = st.session_state.get
    provider = g('live_llm_provider', 'Ollama (local)')
    pid = provider_id(provider)
    if pid == 'ollama':
        model = g('live_llm_model', os.getenv('OLLAMA_MODEL', 'llama3.2:3b'))
        endpoint = g('live_llm_endpoint', os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api/chat'))
        key = ''
    elif pid == 'gemini':
        load_dotenv()
        model = g('live_gemini_model', os.getenv('GEMINI_MODEL', GEMINI_DEFAULT_MODEL))
        endpoint, key = 'Google Gemini API', g('live_gemini_key', '') or os.getenv('GEMINI_API_KEY', '')
    else:
        model = g('live_llm_model2', os.getenv('LLM_MODEL', ''))
        endpoint = g('live_llm_endpoint2', os.getenv('LLM_ENDPOINT', 'https://api.openai.com/v1/chat/completions'))
        key = g('live_llm_key', os.getenv('LLM_API_KEY', ''))
    return {'provider': provider, 'model': model, 'endpoint': endpoint, 'api_key': key,
            'every_n': int(g('live_llm_n', C.LLM_EVERY_N)), 'timeout': int(g('live_llm_timeout', C.LLM_TIMEOUT_S)),
            'min_ticks': int(g('live_llm_min_ticks', C.LLM_MIN_TICKS_BETWEEN_EVALS)),
            'parallel': int(g('live_llm_parallel', C.LLM_MAX_WORKERS)),
            'max_tokens': int(g('live_llm_max_tokens', C.LLM_MAX_OUTPUT_TOKENS)),
            'keep_alive': str(g('live_llm_keep_alive', C.LLM_KEEP_ALIVE)),
            'num_ctx': int(g('live_llm_num_ctx', C.LLM_NUM_CTX)),
            'think_off': bool(g('live_llm_think_off', C.LLM_THINK is False)),
            'rpm': int(g('live_gemini_rpm', C.GEMINI_MAX_RPM))}


def settings_fingerprint(settings: dict) -> tuple:
    """Hashable view of the settings used to detect changes; the API key is masked, never stored in clear."""
    return tuple(sorted({**settings, 'api_key': mask_key(settings.get('api_key'))}.items()))


def live_evaluator(settings: dict):
    """The Live-mode LLM client for these settings (compact prompt, schema output, capped reply, model kept loaded)."""
    return make_live_evaluator(provider_id(settings['provider']), settings['model'],
                               settings['endpoint'], settings['api_key'], settings['timeout'], max_tokens=settings['max_tokens'],
                               keep_alive=settings['keep_alive'] or None, num_ctx=settings['num_ctx'],
                               think=False if settings['think_off'] else None, rpm=settings.get('rpm', C.GEMINI_MAX_RPM))


def llm_factory(settings: dict):
    """A function that builds a fresh LLMController for World L."""
    def make():
        return LLMController(live_evaluator(settings), every_n=settings['every_n'], timeout_s=settings['timeout'],
                             max_workers=settings['parallel'], min_ticks_between=settings['min_ticks'])
    return make


def build_twin(sim, scenario_name: str, seed: int, mode: str, settings: dict) -> TwinSimulation:
    """Create fresh world(s) for the chosen scenario and run mode."""
    factory = llm_factory(settings) if mode != 'bayesian' else None
    if scenario_name == CURRENT_NETWORK:
        twin = TwinSimulation(sim.devices, sim.connections, None, seed, mode, llm_controller_factory=factory)
    else:
        twin = TwinSimulation.from_scenario(scenario_name, seed=seed, mode=mode, llm_controller_factory=factory)
    for world in twin.worlds.values():
        world.scenario_label = scenario_name
    twin.scenario_label = scenario_name
    twin.applied = (scenario_name, seed, mode, settings_fingerprint(settings) if mode != 'bayesian' else None)
    return twin


def _scenario_changed() -> None:
    """When a preset is picked, load its seed into the Seed box (the user can still change it)."""
    name = st.session_state.live_scenario
    if name != CURRENT_NETWORK:
        st.session_state.live_seed = int(load_scenario(name).get('seed', 0))


def _reset(sim) -> None:
    old = st.session_state.get('live_twin')
    if old is not None:
        old.shutdown()
    twin = build_twin(sim, st.session_state.live_scenario, int(st.session_state.live_seed),
                      st.session_state.live_run_mode, llm_settings())
    st.session_state.live_twin = twin
    st.session_state.live_sim = twin.primary   # alias: the first world (Bayesian if present)
    st.session_state.live_running = False


def _init_state(sim) -> None:
    st.session_state.setdefault('live_scenario', 'normal_day' if 'normal_day' in scenario_options() else scenario_options()[0])
    st.session_state.setdefault('live_seed', 42)
    st.session_state.setdefault('live_running', False)
    st.session_state.setdefault('live_run_mode', 'bayesian')
    if 'live_twin' not in st.session_state:
        _reset(sim)


# ----- rendering helpers ---------------------------------------------------------------
_HEAD = re.compile(r'^\[t=(\d+)\] ?')
_WORDS = re.compile(r'\b(STEP-UP|ALLOW|BLOCK|DROPPED|QUARANTINED)\b')
_WORD_CLASS = {'ALLOW': 'w-allow', 'STEP-UP': 'w-step', 'BLOCK': 'w-block', 'DROPPED': 'w-drop', 'QUARANTINED': 'w-drop'}


def _is_detail(line: str) -> bool:
    """The long calculation lines (log-odds terms, LLM state, factors) — shown smaller, and hideable."""
    body = line.strip()
    return body.startswith(('Bayes trust', 'LLM trust', 'factors:'))


def _terminal_line(line: str) -> str:
    m = _HEAD.match(line)
    prefix = f'<span class="t">t={m.group(1)}</span> ' if m else ''
    text = html.escape(line[m.end():] if m else line.strip())
    text = _WORDS.sub(lambda w: f'<b class="{_WORD_CLASS[w.group(1)]}">{w.group(1)}</b>', text)
    return f'<div class="l{" calc" if _is_detail(line) else ""}">{prefix}{text}</div>'


def terminal_markup(lines: List[str], show_details: bool = True) -> str:
    """Dark, monospace, scrolls from the bottom (newest line visible). Display only: the text is render_record's."""
    shown = [l for l in lines[-TERMINAL_LINES:] if show_details or not _is_detail(l)]
    body = ''.join(_terminal_line(l) for l in shown) or '<div class="l">Waiting for traffic. Press Start or Step.</div>'
    return f'<div class="zt-term"><div>{body}</div></div>'


def current_bayes_action(s: dict) -> str:
    """What the Bayesian world would do with the device's next request."""
    if s['status'] == 'Quarantined':
        return 'BLOCK (quarantined)'
    if s['status'] == 'Probation':
        return 'STEP-UP (probation)'
    return BAND_ACTION[s['band']]


def current_llm_action(s: dict) -> str:
    """What the LLM world would do with the device's next request (its OWN decision)."""
    if s['status'] == 'Quarantined':
        return 'BLOCK (quarantined)'
    if s['status'] == 'Probation':
        return 'STEP-UP (probation)'
    return s['decision']


def _status_text(s: dict, long: bool = True) -> str:
    text = s['status']
    if long and s.get('quarantined_until'):
        text += f" to t = {s['quarantined_until']}"
    return status_badge(text)


def status_table(twin: TwinSimulation, reveal: bool) -> pd.DataFrame:
    """One row per device. Single world: Trust / Status / Action. Both: a Bayes group and an LLM group,
    each reading only its own world."""
    b, l = twin.bayes, twin.llm
    sb = b.controller.snapshot() if b else {}
    sl = l.controller.snapshot() if l else {}
    both = bool(b and l)
    primary = twin.primary
    last_event = {}
    for r in reversed(primary.records):
        if r.get('kind') == 'request' and r['src'] not in last_event:
            last_event[r['src']] = f"{fmt_tick(r['tick'])} {r['event_type']}"
    rows = []
    for device_id, dev in sorted(primary.devices.items()):
        row = {'ID': device_id, 'Name': dev.name} if both else {'ID': device_id, 'Name': dev.name, 'Zone': dev.zone}
        if b:
            s = sb[device_id]
            row.update({'Bayes trust' if both else 'Trust': fmt_trust(s['trust']),
                        'Bayes status' if both else 'Status': _status_text(s, not both),
                        'Bayes action' if both else 'Action': current_bayes_action(s)})
        if l:
            s = sl[device_id]
            trust_cell = ('LLM_ERROR' if s['error'] else 'PENDING') if s['trust'] is None else fmt_trust(s['trust'])
            action = current_llm_action(s) + (f" · LLM_ERROR ({s['error']})" if s['error'] else '')
            row.update({'LLM trust' if both else 'Trust': trust_cell, 'LLM status' if both else 'Status': _status_text(s, not both),
                        'LLM action' if both else 'Action': action,
                        'LLM confidence': 'PENDING' if s['last_eval_tick'] is None or s['confidence'] is None else f"{s['confidence']:.2f}",
                        'LLM last eval': 'PENDING' if s['last_eval_tick'] is None else fmt_tick(s['last_eval_tick'])})
        if both:
            ba, la = current_bayes_action(sb[device_id]).split()[0], current_llm_action(sl[device_id]).split()[0]
            row['Agree'] = '—' if la == 'PENDING' else ('yes' if ba == la else 'no')
        if not both:
            row['Last event'] = last_event.get(device_id, '—')
        if reveal:
            gt = primary.ground_truth[device_id]
            row['Ground truth'] = gt['agent'] + (f" (attack {fmt_tick(gt['attack_start'])}+)" if gt['attack_start'] is not None else '')
        rows.append(row)
    df = pd.DataFrame(rows)
    if both:   # agreement right after the name; the LLM detail columns last (they may scroll on narrow screens)
        first = ['ID', 'Name', 'Agree']
        df = df[first + [c for c in df.columns if c not in first]]
    return df


def compress_lines(records: List[dict], hide_allow: bool) -> List[str]:
    """Terminal lines; consecutive DROPPED requests from one device in one tick become a single line."""
    lines: List[str] = []
    i = 0
    while i < len(records):
        r = records[i]
        if r.get('kind') == 'dropped':
            j = i
            while j + 1 < len(records) and records[j + 1].get('kind') == 'dropped' and records[j + 1]['src'] == r['src'] and records[j + 1]['tick'] == r['tick']:
                j += 1
            n = j - i + 1
            lines.append(f"[t={r['tick']:03d}] {r['src']} → … {n} request{'s' if n > 1 else ''} DROPPED — {r['reason']}")
            i = j + 1
            continue
        routine = (r.get('kind') == 'request' and r['action'] == 'ALLOW' and not r['derived'] and not r['step_up']
                   and (r['trust_after'] is None or r['trust_before'] is None or r['trust_after'] >= r['trust_before']))
        if not (hide_allow and routine):
            lines.extend(render_record(r))
        i += 1
    return lines


def topology_figure(live: LiveSimulation):
    """Fixed zone-column layout; node colour = status; quarantined links dashed red."""
    snap = live.controller.snapshot()
    hover = {d: f"{dev.name} · trust {fmt_trust(snap[d]['trust'])}" for d, dev in live.devices.items()}
    return charts.topology_figure(live.devices, live.connections, {d: s['status'] for d, s in snap.items()}, hover,
                                  'ZT controller' if live.engine == 'bayesian' else 'LLM controller')


def default_trust_devices(worlds) -> List[str]:
    """The MAX_TRUST_LINES devices with the lowest current trust in any world (the ones worth watching)."""
    lows = {}
    for world in worlds.values():
        for d, s in world.controller.snapshot().items():
            t = 1.0 if s['trust'] is None else s['trust']
            lows[d] = min(lows.get(d, 1.0), t)
    return [d for d, _ in sorted(lows.items(), key=lambda kv: (kv[1], kv[0]))[:MAX_TRUST_LINES]]


def trust_figure(twin, reveal: bool, devices=None):
    """Trust over time: solid = Bayesian world, dashed = LLM world."""
    worlds = twin.worlds if isinstance(twin, TwinSimulation) else {twin.engine: twin}
    primary = next(iter(worlds.values()))
    devices = list(devices or default_trust_devices(worlds))
    starts = {d: gt['attack_start'] for d, gt in primary.ground_truth.items()
              if reveal and gt['attack_start'] is not None and gt['attack_start'] <= primary.tick and d in devices}
    return charts.trust_figure(worlds, devices, starts)


def latest_record(live: LiveSimulation, device_id: str):
    """Most recent processed request (not dropped) from this device."""
    return next((r for r in reversed(live.records) if r.get('kind') == 'request' and r['src'] == device_id), None)


def live_flow_html(r: dict) -> str:
    """Seven-stage pipeline for one live request, as a static row of numbered steps."""
    terms = ', '.join(f'{l} {d:+.2f}' for l, d in r['terms']) or (
        f"LLM decision from {fmt_tick(r.get('llm_decision_tick'))}" if r.get('llm_decision_tick') is not None else 'LLM PENDING (default policy)')
    stages = [
        ('Request packet', f"{fmt_tick(r['tick'])} {r['src']} → {r['dst']}", f"{r['event_type']} · port {r['port']} · {r['bytes']} B"),
        ('Topology gate', 'Configured link?', ', '.join(f for f in r['derived'] if f in ('no_topology_edge', 'cross_zone_access', 'sensitive_target_access')) or 'authorised path'),
        ('Zero-Knowledge', f"ZK {r['zk']}", 'nonce REPLAYED' if r['replay'] else f"nonce {r['nonce']}"),
        ('Behavior inspection', 'Derived flags', ', '.join(r['derived']) or 'none'),
        ('Trust calculation', f"{fmt_trust(r['trust_before'])} → {fmt_trust(r['trust_after'])}", terms),
        ('Policy engine', f"{r['band_before']} → {r['band_after']}", (f"step-up ZK {r['step_up']['zk']} · MFA {r['step_up']['mfa']}" if r['step_up'] else r['reason'])),
        ('Final decision', r['action'], r['reason'] + (f" · quarantined until {fmt_tick(r['quarantined_until'])}" if r.get('quarantined_until') else '')),
    ]
    return step_row_html(stages, r['action'])


def flow_deep_dive(r: dict, stage: str, live: LiveSimulation) -> None:
    """Detail panel for one stage of the selected device's latest request."""
    if stage == 'Request packet':
        st.dataframe(pd.DataFrame([{'Field': k, 'Value': str(r[k])} for k in ('tick', 'src', 'dst', 'event_type', 'port', 'bytes', 'nonce')]),
                     hide_index=True, use_container_width=True)
        st.caption('This is everything the controller learns about the request itself — no role or label.')
    elif stage == 'Topology gate':
        linked = (r['src'], r['dst']) in live.controller.edges
        st.write(f"Destination: **{r['dst']}** · configured link {r['src']} → {r['dst']}: **{'yes' if linked or r['dst'] == 'CONTROLLER' else 'no'}**")
        st.write('Topology flags raised:', [f for f in r['derived'] if f in ('no_topology_edge', 'cross_zone_access', 'sensitive_target_access')] or 'none')
        st.caption('Requests to the controller are always on an authorised path. Cross-zone and sensitive-target flags only fire when no configured link authorises them.')
    elif stage == 'Zero-Knowledge':
        st.code(f"nonce      = {r['nonce']!r}\nproof      = {r['proof']}\nverify(...) = {r['zk'] == 'PASS'}\nnonce seen before = {r['replay']}", language='python')
        st.caption('The controller verifies the Schnorr proof against the public key registered at enrollment and remembers every nonce, so a replayed proof is caught even though the maths still checks out.')
    elif r.get('engine') == 'llm' and stage in ('Behavior inspection', 'Trust calculation', 'Policy engine'):
        llm_deep_dive(r, stage, live)
    elif stage == 'Behavior inspection':
        from core.events import likelihood
        rows = [{'Derived event': f, 'P(e | legit)': likelihood(f)[0], 'P(e | malicious)': likelihood(f)[1]} for f in r['derived']]
        st.write(f"Token requests in the last {C.WINDOW_TICKS} ticks: **{r['token_count']}** (free allowance {C.TOKEN_FREE_PER_WINDOW})")
        st.dataframe(pd.DataFrame(rows) if rows else pd.DataFrame([{'Derived event': 'none'}]), hide_index=True, use_container_width=True)
    elif stage == 'Trust calculation':
        st.latex(r"L \leftarrow L + \sum_e w_e \log\frac{P(e\mid legit)}{P(e\mid mal)} + \sum_s w_s \log\frac{P(alarm_s\mid legit)}{P(alarm_s\mid mal)},\quad trust=\sigma(L)")
        rows, running = [], r['trust_before']
        from core.controller import clamp_trust, logit, sigmoid
        for label, delta in r['terms']:
            if label == 'decay':
                rows.append({'Term': 'decay (already applied at tick start)', 'Δ log-odds': delta, 'Trust after': running})
                continue
            running = clamp_trust(sigmoid(logit(running) + delta))
            rows.append({'Term': label, 'Δ log-odds': delta, 'Trust after': round(running, 4)})
        if r['step_up']:
            rows.append({'Term': r['step_up']['term'][0], 'Δ log-odds': r['step_up']['term'][1], 'Trust after': r['step_up']['trust_after']})
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
        st.caption(f'Weights: events {C.EVENT_WEIGHT}, sensors {C.SENSOR_WEIGHT} (tempering for correlated evidence). Trust is clamped to [{C.TRUST_MIN}, {C.TRUST_MAX}].')
    elif stage == 'Policy engine':
        ctl = live.controller
        st.code(f"if ZK fails or nonce replayed: BLOCK\nelif on probation:              STEP-UP\nelif trust >= {ctl.allow_threshold:.2f}:  ALLOW   (Trusted)\n"
                f"elif trust >= {ctl.block_threshold:.2f}:  STEP-UP (Suspicious: fresh ZK proof + MFA)\nelse:                          BLOCK + quarantine {C.QUARANTINE_TICKS} ticks\n"
                f"hysteresis ±{C.HYSTERESIS}", language='python')
        st.write(f"Band {r['band_before']} → {r['band_after']} · action **{r['action']}** · {r['reason']}")
        if r['step_up']:
            st.json(r['step_up'])
    else:
        st.metric('Decision', r['action'])
        st.write(r['reason'])
        if r.get('quarantined_until'):
            st.error(f"Quarantined until t={r['quarantined_until']}; then probation at trust {C.PROBATION_TRUST}.")


def llm_deep_dive(r: dict, stage: str, live: LiveSimulation) -> None:
    """LLM-world detail: the facts World L observed, the LLM's own last evaluation, and its policy."""
    led = live.controller.ledgers[r['src']]
    last = next((e for e in reversed(live.records) if e.get('kind') == 'llm_eval' and e['src'] == r['src'] and e['tick'] <= r['tick']), None)
    if stage == 'Behavior inspection':
        st.write('Facts World L recorded for this request (no anomaly labels are computed for the LLM):')
        st.write({'ZK': r['zk'], 'nonce reused': r['replay'], 'configured link': 'no_topology_edge' not in r['derived'],
                  'sensor alarms': [k for k, v in r['alarms'].items() if v]})
        st.caption(f'The LLM sees a {C.WINDOW_TICKS}-tick summary of such facts plus the last {C.LLM_RECENT_REQUESTS} requests.')
    elif stage == 'Trust calculation':
        if last is None:
            st.info('PENDING — the LLM has not evaluated this device yet; the default policy applies.')
            return
        st.write(f"Evaluation at t={last['tick']} (trigger: {last['trigger']}, {last['latency_ms']:.0f} ms)")
        if last['ok']:
            confidence = '—' if last['confidence'] is None else f"{last['confidence']:.2f}"
            st.write(f"LLM trust **{last['llm_trust']:.2f}** · action **{last['llm_action']}** · confidence {confidence}")
            st.write('Reason:', last['reason'] or '—')
            st.write('Factors:', last['factors'] or '—')
        else:
            st.error(f"LLM_ERROR ({last['error']}): {last.get('message', '')} — decision kept: {last['decision_before']}")
        with st.expander('Exact context sent to the LLM (World L facts only)'):
            st.json(last['context'])
    else:
        st.code(f"if ZK fails or nonce replayed: BLOCK   (hard gate, same in both worlds)\nelif on probation:              STEP-UP\n"
                f"elif LLM decision exists:       apply it (ALLOW / STEP-UP / BLOCK → quarantine {C.QUARANTINE_TICKS} ticks)\n"
                f"else (PENDING):                 {C.PENDING_DEFAULT_ACTION} if ZK passes\n"
                f"LLM re-evaluates every {live.controller.every_n} requests or right after {', '.join(C.LLM_HARD_EVENTS)}", language='python')
        st.write(f"Current LLM decision: **{led.decision}** (t={led.last_eval_tick}) · this request: **{r['action']}** · {r['reason']}")


_WIDTHS_ONE = {'ID': 72, 'Name': 170, 'Zone': 100, 'Trust': 64, 'Status': 190, 'Action': 170, 'LLM confidence': 96,
               'LLM last eval': 96, 'Last event': 210, 'Ground truth': 200}
_WIDTHS_BOTH = {'ID': 60, 'Name': 104, 'Bayes trust': 64, 'Bayes status': 100, 'Bayes action': 140, 'LLM trust': 64,
                'LLM status': 100, 'LLM action': 84, 'LLM confidence': 48, 'LLM last eval': 60, 'Agree': 48, 'Ground truth': 180}
_LABELS = {'Bayes trust': 'Trust (B)', 'Bayes status': 'Status (B)', 'Bayes action': 'Action (B)', 'LLM trust': 'Trust (L)',
           'LLM status': 'Status (L)', 'LLM action': 'Action (L)', 'LLM confidence': 'Conf.', 'LLM last eval': 'Last eval'}


def _status_columns(df: pd.DataFrame) -> dict:
    """Fixed pixel widths so the table fits the page. Both mode leaves out Zone (see the map) and Last event (see the terminals)."""
    widths = _WIDTHS_BOTH if 'Bayes trust' in df.columns else _WIDTHS_ONE
    return {c: st.column_config.Column(_LABELS.get(c, c), width=w) for c, w in widths.items() if c in df.columns}


def _style_rows(df: pd.DataFrame):
    """Status columns: coloured dot + text (no cell backgrounds). Trust columns right-aligned."""
    def colour(col):
        if col.name.endswith(('status', 'Status')):
            return [f'color: {status_color(str(v).lstrip("● "))}' for v in col]
        if col.name.endswith(('trust', 'Trust')) or col.name == 'LLM confidence':
            return ['text-align: right'] * len(col)
        if col.name in ('LLM action', 'LLM trust') :
            return ['color: ' + STATUS_COLORS['Isolated'] if 'LLM_ERROR' in str(v) else '' for v in col]
        return [''] * len(col)
    return df.style.apply(colour, axis=0)


def render_results(live: LiveSimulation) -> None:
    """Results: detection metrics computed from ground truth after the decisions, plus exports."""
    if live.tick == 0:
        st.info('Run the simulation first.')
        return
    m = compute_metrics(live)
    model = getattr(getattr(live, 'controller', None), 'model_name', None)
    if model and getattr(live, 'engine', None) == 'llm':
        st.caption(f'LLM: {model}')
    kpi_row([('Attackers detected', f"{m['confusion']['TP']}/{m['confusion']['TP'] + m['confusion']['FN']}"),
             ('False-negative rate', fmt_pct(m['false_negative_rate'])), ('Honest quarantined (FPR)', fmt_pct(m['fpr_block'])),
             ('Honest stepped-up', fmt_pct(m['fpr_step_up'])), ('F1', '—' if m['f1'] is None else f"{m['f1']:.2f}")],
            [None, None, 'Examiner view: computed from ground truth after the run; never fed back to a controller.', None, None])
    left, right = st.columns([3, 2], gap='large')
    with left:
        st.subheader('Attackers')
        st.caption('Ticks are measured from each attack start.')
        st.dataframe(pd.DataFrame(m['attackers']) if m['attackers'] else pd.DataFrame([{'info': 'no attack has started'}]),
                     hide_index=True, use_container_width=True)
        st.subheader('Honest and noisy-honest devices')
        st.dataframe(pd.DataFrame(m['honest']), hide_index=True, use_container_width=True)
    with right:
        st.subheader('Confusion matrix')
        cm = m['confusion']
        st.table(pd.DataFrame({'Predicted attacker': [cm['TP'], cm['FP']], 'Predicted honest': [cm['FN'], cm['TN']]},
                              index=['Truly attacker', 'Truly honest']))
        f2 = lambda x: '—' if x is None else f'{x:.2f}'  # noqa: E731
        st.caption(f"Positive = quarantined. Precision {f2(m['precision'])} · recall {f2(m['recall'])}")
    st.subheader('Trust over time')
    st.plotly_chart(trust_figure(live, True), use_container_width=True, key='results_trust')
    name = f"{getattr(live, 'scenario_label', 'run')}_seed{live.seed}_t{live.tick}".replace(' ', '_').replace('(', '').replace(')', '')
    d = st.columns(4)
    d[0].download_button('Terminal log (CSV)', to_csv(log_rows(live)), f'{name}_log.csv', 'text/csv', key='dl_log')
    d[1].download_button('Trust per tick (CSV)', trust_csv(live), f'{name}_trust.csv', 'text/csv', key='dl_trust')
    d[2].download_button('Metrics (JSON)', metrics_json(m), f'{name}_metrics.json', 'application/json', key='dl_metrics')
    d[3].download_button('Run summary (Markdown)', summary_markdown(m), f'{name}_summary.md', 'text/markdown', key='dl_summary')


# ----- main entry ----------------------------------------------------------------------------
def _llm_settings_controls() -> None:
    """LLM settings (inside the Advanced group)."""
    st.markdown('**LLM settings**')
    provider = st.selectbox('Provider', PROVIDERS, key='live_llm_provider',
                            help='OpenAI-compatible also covers llama.cpp llama-server (see the README).')
    if provider.startswith('Ollama'):
        st.text_input('Model', os.getenv('OLLAMA_MODEL', 'llama3.2:3b'), key='live_llm_model')
        st.text_input('Endpoint', os.getenv('OLLAMA_ENDPOINT', 'http://localhost:11434/api/chat'), key='live_llm_endpoint')
    elif provider_id(provider) == 'gemini':
        _gemini_controls()
    else:
        st.text_input('Model', os.getenv('LLM_MODEL', ''), key='live_llm_model2')
        st.text_input('Chat endpoint', os.getenv('LLM_ENDPOINT', 'https://api.openai.com/v1/chat/completions'), key='live_llm_endpoint2')
        st.text_input('API key', os.getenv('LLM_API_KEY', ''), type='password', key='live_llm_key', help='Leave empty for a local server.')
    st.slider('Evaluate a device every N requests', 1, 50, C.LLM_EVERY_N, key='live_llm_n',
              help='Plus an immediate evaluation after a failed proof, a replayed nonce or a failed step-up.')
    a, b = st.columns(2)
    a.number_input('Min ticks between', 0, 50, C.LLM_MIN_TICKS_BETWEEN_EVALS, key='live_llm_min_ticks',
                   help='No repeat evaluation of the same device within this many ticks, unless a hard event occurs.')
    b.number_input('Timeout (s)', 1, 180, C.LLM_TIMEOUT_S, key='live_llm_timeout')
    a.number_input('Parallel calls', 1, 16, C.LLM_MAX_WORKERS, key='live_llm_parallel',
                   help="For parallel calls set the environment variable OLLAMA_NUM_PARALLEL to the same number as "
                        "'parallel calls' (default 4) before starting Ollama.")
    b.number_input('Max output tokens', 32, 2048, C.LLM_MAX_OUTPUT_TOKENS, key='live_llm_max_tokens',
                   help='Ollama num_predict / OpenAI max_tokens.')
    if provider.startswith('Ollama'):
        a.text_input('keep_alive', C.LLM_KEEP_ALIVE, key='live_llm_keep_alive', help='Keep the model loaded between calls.')
        b.number_input('num_ctx', 1024, 32768, C.LLM_NUM_CTX, step=1024, key='live_llm_num_ctx',
                       help='Context window, set explicitly so the prompt is never truncated.')
        st.checkbox('Thinking off', value=C.LLM_THINK is False, key='live_llm_think_off',
                    help='Sends "think": false for models with a thinking mode (e.g. qwen3); dropped if the model does not support it.')
    if provider.startswith('Ollama'):
        st.caption('Set OLLAMA_NUM_PARALLEL to the number of parallel calls before starting Ollama.')
    t, w = st.columns(2)
    if t.button('Test connection', key='live_llm_test', use_container_width=True):
        try:
            raw = live_evaluator(llm_settings()).evaluate_raw(TEST_CONTEXT)
            res, _ = parse_llm_output(raw)
            st.success(f"Reachable: {res['action']}, trust {res['trust_score']:.2f}, {fmt_ms(raw.get('_latency_ms', 0))}")
        except Exception as exc:  # noqa: BLE001
            st.error(mask_text(str(exc))[:300])
    if w.button('Warm up model', key='live_llm_warm', use_container_width=True,
                help='One tiny request so the first real tick does not pay the model-load time.'):
        try:
            st.success(f'Model loaded ({fmt_ms(live_evaluator(llm_settings()).warm_up())}).')
        except Exception as exc:  # noqa: BLE001
            st.error(mask_text(str(exc))[:300])


def _gemini_controls() -> None:
    """Gemini model ID, key (password field or GEMINI_API_KEY / .env), rate limit and the model list."""
    load_dotenv()
    st.text_input('Model ID', os.getenv('GEMINI_MODEL', GEMINI_DEFAULT_MODEL), key='live_gemini_model',
                  help='A Flash-Lite or Flash model is recommended (fast, cheap); Pro is not needed for this task. '
                       'Use "List models" to see the IDs your key can use.')
    env_key = os.getenv('GEMINI_API_KEY', '')
    st.text_input('API key', '', type='password', key='live_gemini_key',
                  placeholder=f'from GEMINI_API_KEY ({mask_key(env_key)})' if env_key else 'paste a Google AI Studio key',
                  help='Or set GEMINI_API_KEY in the environment or in a .env file. The key is never logged or exported.')
    st.number_input('Max requests per minute', 1, 1000, C.GEMINI_MAX_RPM, key='live_gemini_rpm',
                    help='Client-side limit; calls wait for a free slot. 429 replies are retried with backoff (1 s, 2 s, 4 s).')
    st.caption(GEMINI_NOTE)
    if st.button('List models', key='live_gemini_list', use_container_width=True):
        try:
            names = live_evaluator(llm_settings()).list_models()
            st.code('\n'.join(names) or '(none)', language=None)
        except Exception as exc:  # noqa: BLE001
            st.error(mask_text(str(exc))[:300])


def mask_text(text: str) -> str:
    """Remove any API key that might appear in an error message."""
    for k in {os.getenv('GEMINI_API_KEY', ''), st.session_state.get('live_gemini_key', ''), st.session_state.get('live_llm_key', '')}:
        if k:
            text = text.replace(k, mask_key(k))
    return text


def _sidebar(sim, twin: TwinSimulation):
    """All Live-mode controls in three groups. Returns (twin, speed, reveal, hide_allow)."""
    with st.sidebar:
        with st.expander('Simulation', expanded=True):
            desc = '' if st.session_state.live_scenario == CURRENT_NETWORK else load_scenario(st.session_state.live_scenario).get('description', '')
            st.selectbox('Scenario', scenario_options(), key='live_scenario', on_change=_scenario_changed, help=desc or None)
            st.number_input('Seed', 0, 10_000, key='live_seed', help='Same scenario + seed = identical run.')
            st.radio('Engines', list(MODE_LABELS), format_func=MODE_LABELS.get, key='live_run_mode',
                     help='Both = two isolated worlds with identical inputs; the other world is not created in single-engine modes.')
            c = st.columns(4, gap='small')
            if c[0].button('Start', key='live_start', use_container_width=True, disabled=st.session_state.live_running):
                st.session_state.live_running = True
            if c[1].button('Pause', key='live_pause', use_container_width=True, disabled=not st.session_state.live_running):
                st.session_state.live_running = False
            if c[2].button('Step', key='live_step', use_container_width=True):
                st.session_state.live_running = False
                twin.step()
            if c[3].button('Reset', key='live_reset', use_container_width=True):
                _reset(sim)
                twin = st.session_state.live_twin
            speed = st.slider('Speed (ticks per second)', 1, 10, 2, key='live_speed')
        with st.expander('Policy', expanded=False):
            ctl = twin.primary.controller
            allow = st.slider('ALLOW if trust ≥', 0.50, 0.95, float(ctl.allow_threshold), 0.01, key='live_allow',
                              help='Bayesian world: enforced. LLM world: guidance in its context.')
            block = st.slider('BLOCK if trust <', 0.10, 0.70, float(ctl.block_threshold), 0.01, key='live_block')
            if block < allow:
                for world in twin.worlds.values():   # Bayesian: enforced thresholds; LLM: guidance in its context
                    world.controller.allow_threshold, world.controller.block_threshold = allow, block
            else:
                st.error('The BLOCK threshold must be below the ALLOW threshold.')
        with st.expander('Advanced', expanded=False):
            if st.session_state.live_run_mode != 'bayesian':
                _llm_settings_controls()
                st.divider()
            st.markdown('**Trigger attack**')
            t1, t2 = st.columns(2)
            attack_device = t1.selectbox('Device', sorted(twin.primary.devices), key='live_attack_device')
            attack_type = t2.selectbox('Type', ATTACKER_TYPES, key='live_attack_type')
            if st.button('Trigger attack', key='live_trigger', use_container_width=True,
                         help='The device turns attacker from the next tick, in every world. The controllers are not told.'):
                start = twin.trigger_attack(attack_device, attack_type)
                st.toast(f'{attack_device} turns {attack_type} from t = {start} in every world. The controllers are not told.')
            reveal = st.toggle('Reveal ground truth', value=False, key='live_reveal',
                               help='Examiner view: shows which devices are attackers. No controller ever sees this.')
            hide_allow = st.checkbox('Hide routine traffic', value=False, key='live_hide_allow',
                                     help='Hide ALLOW lines without flags from the terminals.')
    return twin, speed, reveal, hide_allow


def _world_kpis(engine: str, world) -> None:
    snap = world.controller.snapshot()
    n = lambda *keys: sum(s['status'] in keys for s in snap.values())  # noqa: E731
    pending = n('Pending')
    st.markdown(f'**{WORLD_TITLE[engine]}**')
    kpi_row([('Tick', world.tick), ('Trusted', n('Trusted')), ('Suspicious', n('Suspicious', 'Probation', 'Isolated')),
             ('Quarantined', n('Quarantined'))])
    if pending:
        st.caption(f'{pending} pending (no LLM evaluation yet)')


def render_live_mode(sim) -> None:
    """Render the whole Live Network mode."""
    _init_state(sim)
    twin, speed, reveal, hide_allow = _sidebar(sim, st.session_state.live_twin)

    wanted = (st.session_state.live_scenario, int(st.session_state.live_seed), st.session_state.live_run_mode,
              settings_fingerprint(llm_settings()) if st.session_state.live_run_mode != 'bayesian' else None)
    if wanted != twin.applied:
        st.info('Scenario, seed, engines or LLM settings changed. Press Reset to apply.')
    status_line([f'Scenario {twin.scenario_label}', f'seed {twin.seed}', MODE_LABELS[twin.mode],
                 'running' if st.session_state.live_running else 'paused', fmt_tick(twin.tick)])

    interval = max(0.2, 1.0 / speed) if st.session_state.live_running else None
    ticks_per_refresh = max(1, round(speed * interval)) if interval else 0

    live_tab, compare_tab, results_tab, calc_tab = st.tabs(['Live', 'Bayesian vs LLM', 'Results', 'Trust calculation'])
    with compare_tab:
        render_comparison_tab(twin, sim)
    with calc_tab:
        render_calculation_tab(twin, reveal)
    with results_tab:
        engine = st.radio('World', list(twin.worlds), format_func=WORLD_TITLE.get, horizontal=True, key='results_world') \
            if len(twin.worlds) > 1 else next(iter(twin.worlds))
        render_results(twin.worlds[engine])

    @st.fragment(run_every=interval)
    def live_panel() -> None:
        tw: TwinSimulation = st.session_state.live_twin
        status = st.empty()
        if st.session_state.live_running:
            for _ in range(ticks_per_refresh):
                tw.step(progress=lambda n: status.info(f'LLM evaluating {n} device{"s" if n != 1 else ""}; tick {tw.tick + 1} waits for the answers.'))
            status.empty()
        if tw.llm is not None:
            ctl = tw.llm.controller
            if ctl.quota_exhausted:
                st.error('Gemini daily quota exhausted. The LLM world is offline and keeps its last decisions; the run continues.')
            elif ctl.offline:
                st.error('LLM offline. The LLM world keeps its last decisions and retries at each scheduled point.')
            if ctl.errors:
                st.caption('LLM errors so far: ' + ', '.join(f'{k} ×{v}' for k, v in sorted(ctl.errors.items())))
            counters = getattr(ctl.evaluator, 'counters', None)
            if callable(counters):
                st.caption('Gemini: ' + ' · '.join(f'{k} {v}' for k, v in counters().items()))

        kpi_cols = st.columns(len(tw.worlds), gap='large')
        for col, (engine, world) in zip(kpi_cols, tw.worlds.items()):
            with col:
                _world_kpis(engine, world)

        st.subheader('Device status')
        if len(tw.worlds) > 1:
            st.caption('B = Bayesian world, L = LLM world. Each group reads only its own world.')
        table = status_table(tw, reveal)
        st.dataframe(_style_rows(table), use_container_width=True, hide_index=True, row_height=36,
                     height=40 + 36 * len(table), column_config=_status_columns(table))

        head, toggle = st.columns([4, 1])
        head.subheader('Terminal')
        show_calc = toggle.toggle('Show calculation details', value=True, key='live_show_calc')
        term_cols = st.columns(len(tw.worlds), gap='medium')
        for col, (engine, world) in zip(term_cols, tw.worlds.items()):
            with col:
                if len(tw.worlds) > 1:
                    st.caption(WORLD_TITLE[engine])
                st.markdown(terminal_markup(compress_lines(world.records[-TERMINAL_LINES:], hide_allow), show_calc), unsafe_allow_html=True)

        g1, g2 = st.columns(2, gap='large')
        with g1:
            st.subheader('Topology')
            map_engine = st.radio('Colour the map by', list(tw.worlds), format_func=WORLD_TITLE.get, horizontal=True,
                                  key='live_map_world', label_visibility='collapsed') if len(tw.worlds) > 1 else next(iter(tw.worlds))
            st.plotly_chart(topology_figure(tw.worlds[map_engine]), use_container_width=True, key='live_topology')
        with g2:
            st.subheader('Trust over time')
            view = st.radio('Lines', ['Lowest trust', 'Choose devices'], horizontal=True, key='live_trust_view',
                            label_visibility='collapsed', help=f'Lowest trust shows the {MAX_TRUST_LINES} devices with the lowest trust. '
                                                               'Solid = Bayesian, dashed = LLM.')
            chosen = None
            if view == 'Choose devices':
                chosen = st.multiselect('Devices', sorted(tw.primary.devices), default=default_trust_devices(tw.worlds),
                                        key='live_trust_devices', label_visibility='collapsed')
            st.plotly_chart(trust_figure(tw, reveal, devices=chosen), use_container_width=True, key='live_trust')

        with st.expander('Explain a decision', expanded=False):
            f0, f1 = st.columns([1, 1])
            flow_engine = f0.radio('World', list(tw.worlds), format_func=WORLD_TITLE.get, key='live_flow_world', horizontal=True) \
                if len(tw.worlds) > 1 else next(iter(tw.worlds))
            flow_device = f1.selectbox('Device', sorted(tw.primary.devices), key='live_flow_device')
            stage = st.radio('Stage', FLOW_STAGES, horizontal=True, key='live_flow_stage', index=4)
            world = tw.worlds[flow_engine]
            record = latest_record(world, flow_device)
            if record is None:
                st.info('No processed request from this device yet (it may be quarantined or idle).')
            else:
                st.markdown(live_flow_html(record), unsafe_allow_html=True)
                flow_deep_dive(record, stage, world)

    with live_tab:
        live_panel()


def _ui_value(row: dict, value) -> str:
    """Display formatting for the comparison table (the CSV export keeps its own formatting)."""
    from core.comparison import fmt
    if value is None or isinstance(value, (str, bool)):
        return fmt(value, row['format'])
    name = row['parameter'].lower()
    if 'latency' in name:
        return fmt_ms(value)
    if row['format'] == 'pct':
        return fmt_pct(value)
    if 'trust' in name and row['format'] in ('f2', 'f3', 'f4'):
        return f'{value:.2f}'
    return fmt(value, row['format'])


def comparison_display_table(cmp: dict) -> pd.DataFrame:
    rows = []
    for r in cmp['rows']:
        diff = r['difference']
        rows.append({'Section': r['section'], 'Parameter': r['parameter'], 'Bayesian': _ui_value(r, r['bayesian']),
                     'LLM': _ui_value(r, r['llm']),
                     'Difference (LLM − Bayes)': '—' if diff is None else _ui_value({**r, 'format': 'f3' if r['format'] == 'pct' else r['format']}, diff),
                     'Better': r['better']})
    return pd.DataFrame(rows)


def render_comparison_tab(twin: TwinSimulation, sim) -> None:
    """Bayesian vs LLM: every parameter side by side, computed from both worlds' logs + ground truth."""
    if twin.mode != 'both':
        st.info('Choose Engines → Both — isolated comparison in the sidebar and press Reset to compare the two engines.')
        return
    if st.session_state.get('live_running'):
        st.info('Pause the run to compute the comparison.')
        return
    if twin.tick == 0:
        st.info('Run (or Step) the simulation first.')
        return
    key = (twin.applied, twin.tick)
    consistency = st.session_state.get('cmp_consistency', {}).get(key)
    cmp = compare(twin, consistency)
    if cmp.get('llm_model'):
        st.caption(f"LLM: {cmp['llm_model']}")
    sb, sl = cmp['stats']['bayesian'], cmp['stats']['llm']
    mb, ml = sb['metrics'], sl['metrics']
    n = len(mb['attackers'])
    kpi_row([('Attackers caught (B / L)', f"{mb['confusion']['TP']}/{n} · {ml['confusion']['TP']}/{n}"),
             ('Honest quarantined (B / L)', f"{fmt_pct(mb['fpr_block'])} · {fmt_pct(ml['fpr_block'])}"),
             ('Decision agreement', fmt_pct(cmp['agreement']['rate'])),
             ('Latency mean (B / L)', f"{fmt_ms(sb['latency_mean'])} · {fmt_ms(sl['latency_mean'])}")],
            ['Examiner view: computed from ground truth after the run.', None,
             'Share of requests on which both worlds took the same decision.',
             f"Bayesian per request, LLM per evaluation ({sl['evaluations']} evaluations; {sum(sl['errors'].values())} LLM errors)."])

    st.subheader('All parameters')
    table = comparison_display_table(cmp)

    def tint(col):
        if col.name != 'Better':
            return [''] * len(col)
        return [f'color:{ENGINE_COLORS["bayesian"]}' if v == 'Bayesian' else (f'color:{ENGINE_COLORS["llm"]}' if v == 'LLM' else '') for v in col]
    st.dataframe(table.style.apply(tint, axis=0), use_container_width=True, hide_index=True, height=560)

    st.subheader('Attackers side by side')
    st.dataframe(pd.DataFrame(cmp['attackers']) if cmp['attackers'] else pd.DataFrame([{'info': 'no attack started'}]),
                 use_container_width=True, hide_index=True)
    with st.expander('Per device'):
        st.dataframe(pd.DataFrame(cmp['per_device']), use_container_width=True, hide_index=True)

    st.subheader('Trust over time')
    attackers = [d for d, gt in twin.bayes.ground_truth.items() if gt['agent'] not in ('honest', 'noisy_honest')]
    chosen = st.multiselect('Devices', sorted(twin.bayes.devices), default=attackers or sorted(twin.bayes.devices)[:2], key='cmp_devices',
                            label_visibility='collapsed')
    st.plotly_chart(trust_figure(twin, True, devices=chosen or None), use_container_width=True, key='cmp_trust')

    types = cmp['by_attack_type']
    keys = ['fpr_block_honest', 'fpr_block_noisy', 'fpr_stepup_honest', 'fpr_stepup_noisy']
    pts = pd.DataFrame(cmp['agreement']['per_tick'])
    r1 = st.columns(2, gap='large')
    with r1[0]:
        st.subheader('Time to detect')
        st.plotly_chart(charts.bar_pair([t['attack_type'] for t in types], [t['bayes_mean_ticks_to_block'] for t in types],
                                        [t['llm_mean_ticks_to_block'] for t in types], 'ticks to quarantine'),
                        use_container_width=True, key='cmp_ttd')
    with r1[1]:
        st.subheader('False positives')
        st.plotly_chart(charts.bar_pair(['honest<br>quarantined', 'noisy<br>quarantined', 'honest<br>step-up', 'noisy<br>step-up'],
                                        [sb[k] or 0 for k in keys], [sl[k] or 0 for k in keys], 'share of devices'),
                        use_container_width=True, key='cmp_fpr')
    r2 = st.columns(2, gap='large')
    with r2[0]:
        st.subheader('Latency per evaluation')
        st.plotly_chart(charts.latency_box(cmp['latencies']['bayesian'], cmp['latencies']['llm']), use_container_width=True, key='cmp_latency')
    with r2[1]:
        st.subheader('Decision agreement over time')
        st.plotly_chart(charts.line(pts['tick'] if not pts.empty else [], pts['agreement'] if not pts.empty else [], 'share of requests',
                                    yrange=[0, 1.02]), use_container_width=True, key='cmp_agree')

    with st.expander('Per-request decisions vs truth, and trust calibration'):
        conf = cmp['request_confusion']
        st.dataframe(pd.DataFrame([{'outcome': k, 'Bayesian': conf['bayesian'].get(k, 0), 'LLM': conf['llm'].get(k, 0)}
                                   for k in sorted(set(conf['bayesian']) | set(conf['llm']))]), hide_index=True, use_container_width=True)
        c1, c2 = st.columns(2)
        c1.caption('Bayesian reliability'); c1.dataframe(pd.DataFrame(cmp['reliability']['bayesian']), hide_index=True)
        c2.caption('LLM reliability'); c2.dataframe(pd.DataFrame(cmp['reliability']['llm']), hide_index=True)
    with st.expander('Explainability: why each attacker was blocked, and LLM factor overlap'):
        st.dataframe(pd.DataFrame(cmp['explanations']) if cmp['explanations'] else pd.DataFrame([{'info': 'no attacker blocked yet'}]),
                     hide_index=True, use_container_width=True)
        fo = cmp['factor_overlap']
        st.caption(f"LLM factor mentions matching facts in its window: {fo['supported']} · mentioning facts not in its window: "
                   f"{fo['unsupported']} (keyword heuristic)")
        if fo['examples']:
            st.dataframe(pd.DataFrame(fo['examples']), hide_index=True, use_container_width=True)
    cases = cmp['agreement']['cases']
    with st.expander(f"Disagreements: {len(cases)} of {cmp['agreement']['requests_compared']} requests"):
        st.dataframe(pd.DataFrame(cases[:300]) if cases else pd.DataFrame([{'info': 'no disagreements'}]), hide_index=True, use_container_width=True)

    st.subheader('Consistency and export')
    k1, k2, k3, k4 = st.columns(4, vertical_alignment='bottom')
    repeats = k1.number_input('Repeats', 2, 10, C.REPEAT_RUNS_DEFAULT, key='cmp_repeats')
    if k2.button('Repeat run', key='cmp_repeat', use_container_width=True,
                 help='Re-run this scenario + seed for the same number of ticks and measure how identical each engine is.'):
        scenario, seed, mode, _ = twin.applied
        with st.spinner(f'Running {repeats} repeats of {twin.tick} ticks (slow with a real LLM)…'):
            result = repeat_consistency(lambda: build_twin(sim, scenario, seed, mode, llm_settings()), int(repeats), twin.tick)
        st.session_state.setdefault('cmp_consistency', {})[key] = result
        st.rerun()
    if k3.button('Prepare comparison export', key='cmp_export', use_container_width=True, help='CSV + Markdown + charts in one zip.'):
        import io, tempfile, zipfile
        with tempfile.TemporaryDirectory() as tmp, st.spinner('Writing files and charts…'):
            files = write_exports(cmp, twin, tmp)
            buf = io.BytesIO()
            with zipfile.ZipFile(buf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for f in files:
                    zf.write(f, os.path.basename(f))
        st.session_state.cmp_zip = (key, buf.getvalue())
    if st.session_state.get('cmp_zip', (None,))[0] == key:
        k4.download_button('Download export (zip)', st.session_state.cmp_zip[1], key='cmp_download', use_container_width=True,
                           file_name=f"comparison_{twin.scenario_label}_seed{twin.seed}_t{twin.tick}.zip".replace(' ', '_'), mime='application/zip')
    if consistency:
        st.caption(f"Decision consistency: Bayesian {fmt_pct(consistency['bayesian']['decision_consistency'])}, "
                   f"LLM {fmt_pct(consistency['llm']['decision_consistency'])} over {consistency['llm']['repeats']} repeats")
