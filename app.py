import os
import streamlit as st
import pandas as pd
from core.simulator import ZTIoTSimulator, Device
from core.llm_engine import SYSTEM_PROMPT as CLASSIC_SYSTEM_PROMPT, LLMConfig, LLMEvaluator, LLMUnavailable
from core.gemini_engine import DEFAULT_MODEL as GEMINI_DEFAULT_MODEL, GeminiEvaluator, load_dotenv, mask_key, scrub
from core.device_catalog import (CATALOG, CUSTOM_OPTION, PROTOCOLS, POWER_OPTIONS, POSTURES, SENSITIVITY_LEVELS,
                                 option_label, device_label, derive_secret, next_device_id, catalog_defaults,
                                 validate_device_fields, build_device, validate_editor_fields)
from live_ui import render_live_mode
from ui import charts
from ui.theme import fmt_ms, fmt_pct, fmt_trust, kpi_row, load_css, page_header, step_row_html

st.set_page_config(page_title='ZT-IoT Sentinel', layout='wide')
load_css()

if 'sim' not in st.session_state: st.session_state.sim = ZTIoTSimulator()
if 'last' not in st.session_state: st.session_state.last = None
if 'pending_run' not in st.session_state: st.session_state.pending_run = None
if 'selected_flow_step' not in st.session_state: st.session_state.selected_flow_step = 'Trust calculation'
if 'catalog_nonce' not in st.session_state: st.session_state.catalog_nonce = 0  # bumps to give the add-device form fresh defaults
sim = st.session_state.sim

# Behavior likelihoods and the suspicious-behavior set come from core (single source of truth).
from core.bayesian_engine import BEHAVIOR_LIKELIHOODS, SUSPICIOUS_BEHAVIORS

def safe_float(value, fallback=0.0):
    """Convert optional/malformed model values to a finite float for UI rendering."""
    try:
        number = float(value)
        return number if pd.notna(number) else fallback
    except (TypeError, ValueError):
        return fallback

page_header('ZT-IoT Sentinel', 'Zero-trust IoT simulator · Bayesian vs LLM trust evaluation')

FLOW_STEPS = ['Request packet', 'Topology gate', 'Zero-Knowledge', 'Behavior inspection', 'Trust calculation', 'Policy engine', 'Final decision']


def classic_stages(device, target, result=None):
    """The 7 observable stages of one Classic evaluation (values from the latest matching result, if any)."""
    target_name = target.name if target else 'Policy / local resource'
    direct = bool(target and (device.id, target.id) in sim.connections)
    sensitivity = target.target_sensitivity if target else device.target_sensitivity
    bres = (result or {}).get('bayesian') or {}
    lres = (result or {}).get('llm') or {}
    zk_text = ('PASS — proof verified' if result.get('zk_valid') else 'FAIL — proof rejected') if result else 'not run yet'
    behavior_text = ', '.join(device.behaviors) or 'no behaviors'
    if result:
        behavior_text += ' · suspicious=' + str(any(b in SUSPICIOUS_BEHAVIORS for b in device.behaviors) or device.role in {'attacker', 'suspicious'} or device.posture == 'compromised')
    trust_parts = []
    if bres:
        trust_parts.append('Bayes: ' + ' → '.join(fmt_trust(float(x.get('trust', 0))) for x in bres.get('trust_trace', [])))
    if lres:
        trust_parts.append(f"LLM: {fmt_trust(safe_float(lres.get('trust_score'), 0))}")
    trust_detail = ' · '.join(trust_parts) if trust_parts else f'prior {fmt_trust(device.trust)}; IDS / SIEM / Behavior Monitor; sensitivity {sensitivity:.2f}'
    decision = 'awaiting evaluation' if not result else ' · '.join(f"{name}: {rr.get('action', 'N/A')}" for name, rr in [('Bayesian', bres), ('LLM', lres)] if rr)
    final = (bres or lres).get('action') if result else None
    stages = [
        ('Request packet', f'{device.id} → {target_name}', f'role {device.role} · zone {device.zone} · {device.protocol}'),
        ('Topology gate', 'direct edge' if direct else 'no direct edge', f'{len(sim.connections)} configured links'),
        ('Zero-Knowledge', zk_text, 'the secret is never displayed'),
        ('Behavior inspection', f'posture {device.posture}', behavior_text),
        ('Trust calculation', 'model outputs' if result else 'not run yet', trust_detail),
        ('Policy engine', 'hard gates + thresholds', f"allow ≥ {sim.allow_threshold:.2f} · step-up ≥ {sim.step_up_threshold:.2f} · sensitivity {sensitivity:.2f}"),
        ('Final decision', decision, (bres or lres).get('reason', '') if result else ''),
    ]
    return stages, final


def explain_bayesian_trace(result, device):
    rows = []
    for item in result.get('trust_trace', []):
        stage = item.get('stage', 'stage')
        trust = safe_float(item.get('trust', 0), 0.0)
        if stage == 'initial':
            explanation = f'Starting prior trust from device configuration: {device.trust:.1%}.'
        elif stage.startswith('behavior:'):
            behavior = stage.split(':', 1)[1]
            score = BEHAVIOR_LIKELIHOODS.get(behavior,.5)
            explanation = f'Likelihood table value {score:.0%}; ' + ('benign pattern supports trust' if score >= .75 else 'risky/unfamiliar pattern lowers trust' if score < .75 else 'neutral/unknown signal') + '; posterior trust updated.'
        elif stage.startswith('evidence:'):
            explanation = 'Security sensor evidence combined with current trust using Bayes’ rule.'
        else:
            explanation = 'Trust update.'
        rows.append({'Stage':stage,'Posterior trust':trust,'Why this changed':explanation})
    return rows

# ---------------- Mode switch: Live Network (default) or Classic evaluator ----------------
MODES = ['Live network', 'Classic evaluator']
st.session_state.setdefault('app_mode', MODES[0])
app_mode = st.segmented_control('Mode', MODES, key='app_mode', label_visibility='collapsed',
                                help='Live: continuous event-driven network; the controller infers attackers from behaviour only. '
                                     'Classic: the original one-shot evaluator.') or MODES[0]
if app_mode == MODES[0]:
    render_live_mode(sim)
    st.stop()
st.caption('Classic mode uses labelled roles (demonstration only); Live mode infers from behaviour.')

# ---------------- Sidebar: LLM + execution ----------------
with st.sidebar:
  eval_box = st.expander('Evaluation', expanded=True)
  llm_box = st.expander('LLM', expanded=False)
  with llm_box:
    provider = st.selectbox('Provider', ['Ollama (local)', 'Google Gemini (API)', 'OpenAI-compatible API'])
    if provider.startswith('Ollama'):
        model = st.text_input('Ollama model', os.getenv('OLLAMA_MODEL','llama3.2:3b'))
        endpoint = st.text_input('Endpoint', os.getenv('OLLAMA_ENDPOINT','http://localhost:11434/api/chat'))
        api_key=''
    elif provider.startswith('Google'):
        load_dotenv()
        model = st.text_input('Gemini model ID', os.getenv('GEMINI_MODEL', GEMINI_DEFAULT_MODEL), key='classic_gemini_model',
                              help='Flash-Lite or Flash is recommended; Pro is not needed.')
        env_key = os.getenv('GEMINI_API_KEY', '')
        api_key = st.text_input('API key', '', type='password', key='classic_gemini_key',
                                placeholder=f'from GEMINI_API_KEY ({mask_key(env_key)})' if env_key else 'paste a Google AI Studio key') or env_key
        endpoint = 'Google Gemini API'
        st.caption("Simulated device data is sent to Google's API. No real data is used.")
    else:
        model = st.text_input('Model', os.getenv('LLM_MODEL',''))
        endpoint = st.text_input('Chat endpoint', os.getenv('LLM_ENDPOINT','https://api.openai.com/v1/chat/completions'))
        api_key = st.text_input('API key', os.getenv('LLM_API_KEY',''), type='password')
    timeout=st.number_input('Timeout (seconds)',10,180,45)
    temperature=st.slider('Temperature',0.0,1.0,0.0,0.05)
    if provider.startswith('Google'):   # temperature is not sent to Gemini (deprecated from Gemini 3 on)
        llm=GeminiEvaluator(model, api_key or None, CLASSIC_SYSTEM_PROMPT, None, max_output_tokens=None, timeout=timeout)
    else:
        cfg=LLMConfig('ollama' if provider.startswith('Ollama') else 'openai_compatible',model,endpoint,api_key,timeout,temperature)
        llm=LLMEvaluator(cfg)
    tc, wc = st.columns(2)
    if wc.button('Warm up model', key='classic_llm_warm', use_container_width=True,
                 help='Send one tiny request so the model is loaded before the first real evaluation.'):
        try:
            st.success(f'Model loaded and answering ({llm.warm_up():.0f} ms).')
        except Exception as e: st.error(scrub(str(e), api_key))
    if tc.button('Test connection', key='classic_llm_test', use_container_width=True):
        try:
            test_ctx={'device':{'id':'TEST','name':'Test Sensor','role':'normal','trust':.9,'posture':'healthy','behaviors':['normal_telemetry']},'target':{'name':'Test Gateway','target_sensitivity':.2},'zk_valid':True,'suspicious':False,'topology_path':['TEST','GATEWAY'],'policy_thresholds':{'allow_threshold':.72,'step_up_threshold':.88},'security_evidence':[]}
            r=llm.evaluate(test_ctx); st.success(f'LLM reachable: {r["action"]}, trust {r["trust_score"]:.2f}, {fmt_ms(r["_latency_ms"])}')
        except Exception as e: st.error(scrub(str(e), api_key))
  with eval_box:
    device_ids=[d.id for d in sim.devices]
    device_labels={d.id: device_label(d) for d in sim.devices}  # values stay plain ids; only the display changes
    source_id=st.selectbox('Source device',device_ids,format_func=lambda i: device_labels.get(i, i))
    target_options=['— No target —']+[d.id for d in sim.devices if d.id!=source_id]
    target_choice=st.selectbox('Target device/resource',target_options,format_func=lambda i: device_labels.get(i, i))
    method=st.radio('Run', ['Both — Bayesian + LLM','Bayesian only','LLM only'])
    force_bad_zk=st.checkbox('Simulate invalid ZK proof')
    if st.button('Run evaluation', key='classic_run', use_container_width=True, type='primary'):
        try:
            tid=None if target_choice.startswith('—') else target_choice
            m={'Both — Bayesian + LLM':'both','Bayesian only':'bayesian','LLM only':'llm'}[method]
            st.session_state.pending_run={'source_id':source_id,'target_id':tid,'method':m,'force_bad_zk':force_bad_zk}
            st.rerun()
        except Exception as e: st.error(str(e))
    if st.button('Reset simulation', key='classic_reset', use_container_width=True):
        st.session_state.sim=ZTIoTSimulator(); st.session_state.last=None; st.session_state.catalog_nonce+=1; st.rerun()

# ---------------- Live run: render the moving pipeline before the evaluator blocks ----------------
if st.session_state.get('pending_run'):
    job = st.session_state.pending_run
    source = next((d for d in sim.devices if d.id == job['source_id']), None)
    target = next((d for d in sim.devices if d.id == job['target_id']), None) if job['target_id'] else None
    if source:
        st.subheader('Evaluating')
        st.markdown(step_row_html(classic_stages(source, target)[0]), unsafe_allow_html=True)
        with st.spinner('Running the ZK proof, the trust evaluation and the policy…'):
            try:
                st.session_state.last = sim.run(job['source_id'], job['target_id'], job['method'], job['force_bad_zk'], llm)
            except Exception as e:
                st.error(f'Evaluation failed: {e}')
        st.session_state.pending_run = None
        st.rerun()

# ---------------- Runtime device editor ----------------
with st.expander('Network builder: devices, properties and topology', expanded=False):
    left,right=st.columns([1,1])
    with left:
        st.subheader('Add / edit device')
        edit_id=st.selectbox('Edit existing device', ['+ New device']+[d.id for d in sim.devices])
        existing=next((d for d in sim.devices if d.id==edit_id),None)
        def val(attr, default=''):
            return getattr(existing,attr,default) if existing else default
        idv=st.text_input('Device ID',val('id','DEV-NEW'))
        name=st.text_input('Name',val('name','New IoT Device'))
        dtype=st.text_input('Type',val('type','Sensor'))
        zone=st.text_input('Zone',val('zone','Public IoT'))
        ip=st.text_input('IP',val('ip','10.0.4.10'))
        firmware=st.text_input('Firmware',val('firmware','v1.0'))
        protocol=st.selectbox('Protocol',PROTOCOLS,index=PROTOCOLS.index(val('protocol','MQTT')) if val('protocol','MQTT') in PROTOCOLS else 0)
        power_opts=['Low','Medium','High']+POWER_OPTIONS  # original draw levels + catalog power sources
        power=st.selectbox('Power',power_opts,index=power_opts.index(val('power','Medium')) if val('power','Medium') in power_opts else 1)
        status=st.selectbox('Status',['Online','Offline','Suspicious','Quarantined'],index=['Online','Offline','Suspicious','Quarantined'].index(val('status','Online')))
        role=st.selectbox('Security role',['normal','attacker','suspicious'],index=['normal','attacker','suspicious'].index(val('role','normal')))
        posture=st.selectbox('Posture',['healthy','unknown','compromised'],index=['healthy','unknown','compromised'].index(val('posture','healthy')))
        trust=st.slider('Initial trust',0.0,1.0,float(val('trust',.8)),.01)
        sensitivity=st.slider('Target sensitivity',0.0,1.0,float(val('target_sensitivity',.5)),.01)
        all_beh=['normal_telemetry','authorized_service','read_sensor','firmware_update','cross_zone_access','rapid_scanning','credential_reuse','critical_target_probe','unexpected_port','data_exfiltration']
        behaviors=st.multiselect('Observed behavior profile',all_beh,default=val('behaviors',['normal_telemetry']))
        c1,c2=st.columns(2)
        if c1.button('Save device', key='classic_save'):
            secret=existing.secret if existing else derive_secret(idv)  # stable; hash() changed every run
            nd=Device(idv,name,dtype,zone,ip,firmware,protocol,power,status,secret,trust,posture,sensitivity,role,behaviors or ['normal_telemetry'])
            editor_errors=validate_editor_fields(idv,name,ip,sim.devices,existing.id if existing else None)
            if editor_errors:
                for e in editor_errors: st.error(e)
            else:
                try:
                    if existing:
                        idx=sim.devices.index(existing); sim.devices[idx]=nd
                    else: sim.add_device(nd)
                    st.success('Saved.'); st.rerun()
                except Exception as e: st.error(str(e))
        if existing and c2.button('Delete device', key='classic_delete'):
            sim.remove_device(existing.id); st.rerun()
    with right:
        st.subheader('Topology connections')
        ids=[d.id for d in sim.devices]
        for src in ids:
            targets=[d for d in ids if d!=src]
            selected=[t for t in targets if (src,t) in sim.connections]
            newsel=st.multiselect(f'{src} →',targets,default=selected,key=f'top_{src}')
            for t in targets: sim.set_connection(src,t,t in newsel)
        st.caption('Connections are directed. The selected source/target is evaluated in the current runtime topology.')

def render_catalog_adder() -> None:
    """'Add device from catalog' control: pick a catalog type (form auto-fills) or a custom device."""
    flash = st.session_state.pop('catalog_flash', None)
    if flash: st.success(flash)
    choice = st.selectbox('Device type', list(CATALOG) + [CUSTOM_OPTION], format_func=option_label, key='catalog_choice')
    defaults = catalog_defaults(choice, sim.devices)
    if choice != CUSTOM_OPTION:
        level = CATALOG[choice]['sensitivity']
        st.caption(f'Defaults are modelling assumptions (sensitivity level: **{level}** = {SENSITIVITY_LEVELS[level]:.2f}). Edit any field before adding.')
    st.caption(f'The device will be created as **{next_device_id(sim.devices)}** with role `normal`.')
    k = f'cat_{choice}_{st.session_state.catalog_nonce}_'  # new keys per type/after each add -> fresh auto-filled values
    all_beh = list(BEHAVIOR_LIKELIHOODS)
    def opt_index(options, value):
        return options.index(value) if value in options else None
    with st.form(key=f'{k}form'):
        a, b = st.columns(2)
        fields = {
            'name': a.text_input('Name *', defaults['name'], key=k+'name'),
            'type': b.text_input('Type *', defaults['type'], key=k+'type'),
            'zone': a.text_input('Zone *', defaults['zone'], key=k+'zone'),
            'ip': b.text_input('IP address *', defaults['ip'], key=k+'ip', placeholder='e.g. 10.0.4.30'),
            'firmware': a.text_input('Firmware *', defaults['firmware'], key=k+'fw', placeholder='e.g. v1.0.0'),
            'protocol': b.selectbox('Protocol *', PROTOCOLS, index=opt_index(PROTOCOLS, defaults['protocol']), key=k+'proto'),
            'power': a.selectbox('Power source *', POWER_OPTIONS, index=opt_index(POWER_OPTIONS, defaults['power']), key=k+'power'),
            'posture': b.selectbox('Posture *', POSTURES, index=opt_index(POSTURES, defaults['posture']), key=k+'posture'),
            'target_sensitivity': a.slider('Target sensitivity', 0.0, 1.0, float(defaults['target_sensitivity']), .01, key=k+'sens'),
            'trust': b.slider('Initial trust', 0.0, 1.0, float(defaults['trust']), .01, key=k+'trust'),
            'behaviors': st.multiselect('Observed behavior profile *', all_beh, default=defaults['behaviors'], key=k+'beh'),
        }
        submitted = st.form_submit_button('Add device', type='primary')
    if submitted:
        errors = validate_device_fields(fields, sim.devices)
        if errors:
            for e in errors: st.error(e)
            return
        nd = build_device(fields, sim.devices)
        sim.add_device(nd)
        st.session_state.catalog_nonce += 1
        st.session_state.catalog_flash = f'Added {nd.id} — {nd.name}. It now appears in Source/Target and the topology editor; remove it via "Edit existing device" → Delete.'
        st.rerun()

with st.expander('Add device from catalog', expanded=False):
    render_catalog_adder()

# ---------------- Decision flow ----------------
st.subheader('Decision flow')
flow_source = next((d for d in sim.devices if d.id == source_id), sim.devices[0] if sim.devices else None)
flow_target_id = None if str(target_choice).startswith('—') else target_choice
flow_target = next((d for d in sim.devices if d.id == flow_target_id), None)
last = st.session_state.last
flow_result = last if last and flow_source and last.get('device_id') == flow_source.id and (last.get('target_id') or '') == (flow_target.id if flow_target else '') else None
if flow_source:
    stages, final = classic_stages(flow_source, flow_target, flow_result)
    st.markdown(step_row_html(stages, final), unsafe_allow_html=True)
    flow_steps = FLOW_STEPS
    chosen_step = st.radio('Inspect stage', flow_steps, index=flow_steps.index(st.session_state.selected_flow_step), horizontal=True,
                           key='classic_flow_stage', label_visibility='collapsed')
    st.session_state.selected_flow_step = chosen_step
    chosen = st.session_state.selected_flow_step
    stage_box = st.expander(f'Stage details: {chosen}', expanded=True)
    with stage_box:
        d = flow_source
        t = flow_target
        direct = bool(t and (d.id, t.id) in sim.connections)
        sensitivity = t.target_sensitivity if t else d.target_sensitivity
        r = flow_result or {}
        bres = (r.get('bayesian') or {}) if r else {}
        lres = (r.get('llm') or {}) if r else {}
        if chosen == 'Request packet':
            st.markdown('**Purpose:** construct the access request context. These values are read from the device currently selected in the sidebar and runtime builder.')
            st.dataframe(pd.DataFrame([{'Field':k,'Actual value':v,'Why it matters':why} for k,v,why in [
                ('Device ID',d.id,'Stable identity label used to resolve the requesting device.'),('Name / type',f'{d.name} / {d.type}','Human-readable identity and device class.'),('Role',d.role,'attacker/suspicious roles contribute to the suspicious flag.'),('Zone',d.zone,'Network segment context for cross-zone behavior.'),('IP / protocol',f'{d.ip} / {d.protocol}','Request metadata supplied to the LLM context.'),('Firmware / status',f'{d.firmware} / {d.status}','Posture clues; these are context unless a hard rule explicitly checks them.'),('Initial trust',f'{d.trust:.4f}','Bayesian prior probability before behavior/evidence updates.'),('Target',f'{t.id} — {t.name}' if t else 'No target selected','Resource whose sensitivity influences policy/context.'),('Target sensitivity',f'{sensitivity:.4f}','Higher values indicate a more sensitive resource.')]]),use_container_width=True,hide_index=True)
        elif chosen == 'Topology gate':
            st.markdown('**Purpose:** determine whether a directed source → target edge is configured. The current prototype passes this as context to the LLM; absence of an edge is not itself a hard Bayesian block.')
            st.code(f"source_id = {d.id!r}\ntarget_id = {t.id if t else None!r}\nconfigured_connections = {len(sim.connections)}\ndirect_connection = {direct!r}\nmatching_edge = {(d.id, t.id) if t else None!r}", language='python')
            st.write('Configured outgoing edges:', [f'{a} → {b}' for a,b in sim.connections if a == d.id] or ['No outgoing edges configured.'])
            st.write('Observed result:', '**DIRECT EDGE FOUND**' if direct else '**NO DIRECT EDGE**')
        elif chosen == 'Zero-Knowledge':
            st.markdown('**Purpose:** prove knowledge of the registered secret without displaying the secret itself. This educational simulator uses tiny demonstration parameters and is not production cryptography.')
            st.markdown('1. The simulator creates a fresh nonce for this run.  2. `prove(secret, device_id, nonce)` constructs a proof.  3. `verify(public_identity, proof, device_id)` checks it.  4. The optional invalid-proof switch mutates the response to demonstrate rejection.')
            st.code("nonce = f'{device.id}-{len(events)}-{time.time_ns()}'\nproof = prove(device.secret, device.id, nonce)\nzk_valid = verify(device.public_identity(), proof, device.id)\nif force_bad_zk:\n    proof['response'] = (proof['response'] + 7) % 113\n    zk_valid = verify(device.public_identity(), proof, device.id)", language='python')
            if r: st.metric('Actual proof result', 'PASS' if r.get('zk_valid') else 'FAIL')
            else: st.info('No completed run for this selected request yet; run an evaluation to see the actual PASS/FAIL result.')
            st.warning('A valid proof establishes the simulated authentication check only; it does not automatically make the device trustworthy.')
        elif chosen == 'Behavior inspection':
            suspicious_names = SUSPICIOUS_BEHAVIORS
            raw_behaviors = getattr(d, 'behaviors', [])
            if callable(raw_behaviors):
                raw_behaviors = raw_behaviors()
            if raw_behaviors is None:
                raw_behaviors = []
            elif isinstance(raw_behaviors, str):
                raw_behaviors = [raw_behaviors]
            elif not isinstance(raw_behaviors, (list, tuple, set)):
                try:
                    raw_behaviors = list(raw_behaviors)
                except TypeError:
                    raw_behaviors = []
            behaviors = [str(item) for item in raw_behaviors if item is not None]
            suspicious = str(d.role).lower() in {'attacker','suspicious'} or str(d.posture).lower() == 'compromised' or any(x in suspicious_names for x in behaviors)
            rows=[]
            likelihoods = BEHAVIOR_LIKELIHOODS
            for behavior in behaviors:
                score=BEHAVIOR_LIKELIHOODS.get(behavior,.5)
                rows.append({'Behavior':behavior,'Bayesian likelihood P(legitimate | behavior proxy)':score,'Interpretation':'Benign-like evidence' if score>=.75 else 'Risky / unusual evidence' if score<.75 else 'Unknown / neutral evidence','Suspicious flag contribution':behavior in suspicious_names})
            st.write(f'Role = **{d.role}** · posture = **{d.posture}** · status = **{d.status}** · combined suspicious flag = **{suspicious}**')
            # Explicit branch avoids ambiguous rendering and gives clearer failures.
            if rows:
                behavior_df = pd.DataFrame.from_records(rows)
                st.dataframe(behavior_df, use_container_width=True, hide_index=True)
            else:
                st.info('No behaviors configured for this device.')
            st.markdown('The suspicious flag becomes true when role is attacker/suspicious, posture is compromised, or at least one configured behavior belongs to the suspicious behavior set. The Bayesian engine then updates trust using each behavior likelihood and three security sensors (IDS, SIEM, Behavior Monitor).')
        elif chosen == 'Trust calculation':
            st.markdown('**Bayesian engine — actual equations**')
            st.markdown("For each behavior, the code uses a likelihood proxy *s*. It sets P(E | legitimate) = s and P(E | attacker) = 1 − s, then updates trust with Bayes’ rule:")
            st.latex(r"P(L\mid E)=\frac{P(E\mid L)P(L)}{P(E\mid L)P(L)+P(E\mid A)P(A)}")
            st.markdown('For each sensor evidence item, if an alarm is raised, the implementation uses its false-positive rate for the legitimate hypothesis and true-positive rate for the attacker hypothesis; for no alarm, it uses their complements.')
            if bres:
                trace=bres.get('trust_trace',[])
                trows=[]
                current=float(d.trust)
                for item in trace:
                    stage=item.get('stage','')
                    after=safe_float(item.get('trust',0), current)
                    behavior=stage.split(':',1)[1] if stage.startswith('behavior:') else None
                    score=BEHAVIOR_LIKELIHOODS.get(behavior,.5) if behavior else None
                    trows.append({'Update':stage,'Prior trust':current,'Likelihood proxy s':score if score is not None else 'sensor / initial','Posterior trust':after,'Change':after-current,'Meaning':'Prior' if stage=='initial' else ('Raises trust' if after>current else 'Lowers trust' if after<current else 'No change')})
                    current=after
                st.markdown('**Actual trace from the latest matching run**')
                st.dataframe(pd.DataFrame(trows),use_container_width=True,hide_index=True)
                st.metric('Final Bayesian trust',fmt_trust(bres.get('trust_score',0)))
            else:
                st.info('Run Bayesian or Both mode for actual intermediate posterior values. The formula and configured behavior likelihoods above are available before running.')
            st.markdown('**Configured behavior likelihood table**')
            st.dataframe(pd.DataFrame([{'Behavior':k,'Likelihood proxy s':v,'Direction in this implementation':'Higher s generally supports legitimacy; lower s tends to reduce trust.'} for k,v in BEHAVIOR_LIKELIHOODS.items()]),use_container_width=True,hide_index=True)
            st.caption('These are hand-configured model likelihood proxies, not learned probabilities from a training dataset.')
            st.markdown('**LLM evaluator — observable processing**')
            st.write('The app serializes device properties, target, ZK result, suspicious flag, topology and policy thresholds as JSON; sends them with a system instruction requesting structured JSON; parses the returned JSON; clamps trust/risk/confidence to [0, 1]; and replaces invalid actions with BLOCK. The model’s private internal reasoning is not available to the app.')
            if lres:
                st.markdown('Returned structured output')
                st.json({k:v for k,v in lres.items() if not k.startswith('_')})
            else:
                st.info('No LLM result for the selected request yet. Configure a reachable endpoint and run LLM or Both mode.')
        elif chosen == 'Policy engine':
            st.markdown('**Bayesian policy rule order matters.** The first matching hard gate wins: compromised posture → failed ZK proof → suspicious/low-trust handling → general trust threshold.')
            st.code("if posture == 'compromised': BLOCK\nelif not zk_valid: BLOCK\nelif suspicious and trust < step_up_threshold:\n    if trust >= allow_threshold and sensitivity < 0.80: STEP-UP\n    else: BLOCK\nelif trust >= allow_threshold: ALLOW\nelse: BLOCK",language='python')
            st.dataframe(pd.DataFrame([{'Policy parameter':'Allow threshold','Configured value':sim.allow_threshold,'Meaning':'Minimum trust for normal ALLOW.'},{'Policy parameter':'Step-up threshold','Configured value':sim.step_up_threshold,'Meaning':'Suspicious devices below this level require stronger validation.'},{'Policy parameter':'Target sensitivity','Configured value':sensitivity,'Meaning':'STEP-UP path only permits step-up if sensitivity is below 0.80 in the current implementation.'},{'Policy parameter':'Posture','Configured value':d.posture,'Meaning':'compromised is an unconditional Bayesian BLOCK.'},{'Policy parameter':'ZK proof','Configured value':('PASS' if r.get('zk_valid') else 'FAIL') if r else 'Not run','Meaning':'failed proof is an unconditional Bayesian BLOCK.'}]),use_container_width=True,hide_index=True)
            if bres: st.success(f"Bayesian: {bres.get('action')} — {bres.get('reason')}")
            if lres: st.info(f"LLM advisory decision: {lres.get('action')} — {lres.get('reason')}")
            st.caption('The LLM action is generated by the model from the supplied policy context; unlike Bayesian policy_decide, it is not produced by the same deterministic if/else rule sequence.')
        elif chosen == 'Final decision':
            if not r:
                st.info('No completed result matching the currently selected source and target. Run an evaluation to populate the final decision audit.')
            else:
                st.dataframe(pd.DataFrame([{'Evaluator':'Bayesian','Trust':bres.get('trust_score'),'Risk':bres.get('risk'),'Action':bres.get('action'),'Reason':bres.get('reason')} if bres else {'Evaluator':'Bayesian','Trust':None,'Risk':None,'Action':'Not run','Reason':''},{'Evaluator':'LLM','Trust':lres.get('trust_score'),'Risk':lres.get('risk_score'),'Action':lres.get('action'),'Reason':lres.get('reason')} if lres else {'Evaluator':'LLM','Trust':None,'Risk':None,'Action':'Not run','Reason':''}]),use_container_width=True,hide_index=True)
                st.markdown(f"**Authentication:** {'PASS' if r.get('zk_valid') else 'FAIL'} · **Behaviors:** {', '.join(r.get('behaviors',[]))} · **Source role:** {r.get('role')}")
                st.markdown('Compare the explicit reason with the trace above: the reason should be tied to proof validity, posture, suspicious behavior, trust thresholds, or the model-returned factors.')

# ---------------- Dashboard ----------------
tabs=st.tabs(['Overview','Devices','Topology','Comparison','Audit'])
last=st.session_state.last

with tabs[0]:
    kpi_row([('Devices', len(sim.devices)), ('Connections', len(sim.connections)), ('Runs', len(sim.events)),
             ('Bayesian', 'Deterministic'), ('LLM', 'JSON constrained')])
    st.markdown('**Pipeline:** Device → ZK proof → Bayesian/LLM Trust Evaluation → Policy Engine → ALLOW / STEP-UP / BLOCK')
    st.info('The LLM is not treated as a cryptographic authenticator. ZK verifies secret knowledge; the LLM interprets supplied security context for trust and policy. The response is schema-validated before display.')
    if last:
        a,b=st.columns(2)
        for col,key,title in [(a,'bayesian','Bayesian'),(b,'llm','LLM')]:
            with col:
                st.subheader(title)
                r=last.get(key)
                if r: kpi_row([('Trust', fmt_trust(r['trust_score'])), ('Decision', r['action'])]); st.write(r['reason'])
                else: st.warning('Not run')

with tabs[1]:
    rows=[]
    for d in sim.devices:
        rows.append({'ID':d.id,'Name':d.name,'Type':d.type,'Zone':d.zone,'Role':d.role,'Posture':d.posture,'Trust':d.trust,'Sensitivity':d.target_sensitivity,'Protocol':d.protocol,'Behaviors':', '.join(d.behaviors)})
    st.dataframe(pd.DataFrame(rows),use_container_width=True,hide_index=True)

with tabs[2]:
    devs = {d.id: d for d in sim.devices}
    st.plotly_chart(charts.topology_figure(devs, sim.connections, {d: 'Pending' for d in devs},
                                           {d.id: f'{d.name} · role {d.role} · trust {fmt_trust(d.trust)}' for d in sim.devices},
                                           'Controller', height=460), use_container_width=True, key='classic_topology')
    st.caption('Connections: ' + (', '.join(f'{a} → {b}' for a, b in sim.connections) if sim.connections else 'none'))

with tabs[3]:
    if not last:
        st.warning('Run an evaluation first.')
    else:
        st.subheader(f'Live comparison: {last["device"]} → {last["target"] or "unspecified target"}')
        b=last.get('bayesian'); l=last.get('llm')
        if b and l:
            rows=[
                ['Trust score',b['trust_score'],l['trust_score'],abs(b['trust_score']-l['trust_score'])],
                ['Risk score',b['risk'],l['risk_score'],abs(b['risk']-l['risk_score'])],
                ['Decision',b['action'],l['action'], 'MATCH' if b['action']==l['action'] else 'MISMATCH'],
                ['Latency (ms)',b['_latency_ms'],l['_latency_ms'],l['_latency_ms']-b['_latency_ms']],
                ['Confidence','—',l['confidence'],'—'],
                ['ZK valid','PASS' if last['zk_valid'] else 'FAIL','PASS' if last['zk_valid'] else 'FAIL','—'],
            ]
            st.dataframe(pd.DataFrame(rows,columns=['Parameter','Bayesian','LLM','Difference / agreement']),use_container_width=True,hide_index=True)
            kpi_row([('Trust delta', f'{abs(b["trust_score"]-l["trust_score"]):.2f}'), ('Decision agreement', 'yes' if b['action']==l['action'] else 'no'),
                     ('Bayesian latency', fmt_ms(b['_latency_ms'])), ('LLM latency', fmt_ms(l['_latency_ms']))])
            st.subheader('Reasoning and factors')
            x,y=st.columns(2)
            with x:
                st.markdown('**Bayesian trace**')
                st.dataframe(pd.DataFrame(b['trust_trace']),use_container_width=True,hide_index=True)
                st.write(b['reason'])
            with y:
                st.markdown('**LLM factors**')
                st.write(l.get('factors',[])); st.write(l['reason'])
                st.write('Behavior assessment:',l.get('behavior_assessment',[]))
            st.subheader('Decision path: what caused the outcome?')
            selected_device = next((d for d in sim.devices if d.id == last.get('device_id')), None)
            selected_target = next((d for d in sim.devices if d.id == last.get('target_id')), None) if last.get('target_id') else None
            if selected_device:
                st.markdown(step_row_html(*classic_stages(selected_device, selected_target, last)), unsafe_allow_html=True)
                signals = [
                    {'Check':'Source identity','Observed value':f"{selected_device.id} / {selected_device.name}",'Effect':'Identifies which device is requesting access.'},
                    {'Check':'Device role','Observed value':selected_device.role,'Effect':'Attacker/suspicious roles trigger suspicious handling.' if selected_device.role in {'attacker','suspicious'} else 'No role-based suspicion flag.'},
                    {'Check':'Device posture','Observed value':selected_device.posture,'Effect':'Compromised posture is a hard BLOCK in Bayesian policy.' if selected_device.posture=='compromised' else 'Posture is not marked compromised.'},
                    {'Check':'Zero-knowledge proof','Observed value':'PASS' if last.get('zk_valid') else 'FAIL','Effect':'Failed proof is a hard BLOCK in Bayesian policy; valid proof alone does not guarantee trust.' if last.get('zk_valid') else 'Authentication gate failed; access must be blocked.'},
                    {'Check':'Observed behaviors','Observed value':', '.join(last.get('behaviors',[])),'Effect':'Risky behaviors reduce Bayesian trust and are supplied to the LLM evaluator.'},
                    {'Check':'Target sensitivity','Observed value':f"{selected_target.target_sensitivity:.0%}" if selected_target else f"{selected_device.target_sensitivity:.0%}",'Effect':'Higher sensitivity should demand stronger evidence; the LLM sees this context.'},
                    {'Check':'Topology','Observed value':'direct connection' if selected_target and (selected_device.id,selected_target.id) in sim.connections else ('no direct connection' if selected_target else 'no target selected'),'Effect':'The direct-connection flag is supplied to the LLM; missing topology alone is not a hard Bayesian block in this prototype.'},
                ]
                st.dataframe(pd.DataFrame(signals),use_container_width=True,hide_index=True)
            if b:
                st.markdown('**Bayesian posterior update trace**')
                trace_rows = explain_bayesian_trace(b, selected_device) if selected_device else []
                if trace_rows: st.dataframe(pd.DataFrame(trace_rows),use_container_width=True,hide_index=True)
                st.markdown(f"**Bayesian policy reason:** {b.get('reason','No reason returned')}  ")
                st.markdown(f"**Required validation:** {b.get('required_validation','Not specified')} · **Risk:** {b.get('risk',1-b.get('trust_score',0)):.1%}")
            if l:
                st.markdown('**LLM returned explanation (not hidden chain-of-thought)**')
                st.write(l.get('reason','No reason returned.'))
                st.write('**Returned factors:**')
                for factor in l.get('factors',[]): st.markdown(f'- {factor}')
                st.write('**Behavior risk assessments:**')
                assessments = l.get('behavior_assessment',[])
                if assessments: st.dataframe(pd.DataFrame(assessments),use_container_width=True,hide_index=True)
                else: st.caption('The model returned no per-behavior risk assessments for this run.')
                st.markdown(f"**LLM policy result:** {l.get('action')} · trust {l.get('trust_score',0):.1%} · risk {l.get('risk_score',1-l.get('trust_score',0)):.1%} · confidence {l.get('confidence',0):.1%}")
                st.markdown(f"**Required validation:** {l.get('required_validation','Not specified')}")
        else:
            r=b or l
            kpi_row([('Trust', fmt_trust(r['trust_score'])), ('Decision', r['action'])]); st.write(r['reason'])

with tabs[4]:
    if sim.events:
        records=[]
        for e in sim.events:
            row={'Time':e['timestamp'],'Source':e['device'],'Target':e['target'],'Role':e['role'],'ZK':'PASS' if e['zk_valid'] else 'FAIL'}
            for k,label in [('bayesian','Bayesian'),('llm','LLM')]:
                r=e.get(k)
                row[f'{label} Trust']=r['trust_score'] if r else None
                row[f'{label} Decision']=r['action'] if r else None
                row[f'{label} Latency']=r.get('_latency_ms') if r else None
            records.append(row)
        st.dataframe(pd.DataFrame(records),use_container_width=True,hide_index=True)
        st.download_button('Download audit (CSV)',pd.DataFrame(records).to_csv(index=False),'zt_iot_audit.csv','text/csv', key='classic_audit')
    else: st.info('No evaluations yet.')

st.caption('Educational laptop-only simulator. ZK parameters are intentionally tiny and are not production cryptography. LLM output is an advisory evaluation layer; do not use this prototype as a real network access controller.')
