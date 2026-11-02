"""Every hand-chosen number used by Live mode, in one place.

EVERYTHING IN THIS FILE IS A MODELLING ASSUMPTION for a teaching simulator.
None of it is measured from real IoT traffic. Each value has a one-line comment
saying what it means so it can be defended (or changed) in the report.
"""
from .bayesian_engine import BEHAVIOR_LIKELIHOODS as _CLASSIC

# --- Trust ledger ----------------------------------------------------------
LIVE_PRIOR = 0.80          # assumption: every enrolled device starts with the SAME prior (controller has no per-device knowledge)
TRUST_MIN = 0.01           # assumption: floor so trust never reaches certainty (keeps recovery possible)
TRUST_MAX = 0.99           # assumption: ceiling for the same reason
DECAY_LAMBDA = 0.98        # assumption: per tick, keep 98% of the log-odds and pull 2% back toward the prior
EVENT_WEIGHT = 0.5         # assumption: tempering for correlated evidence within one request (1.0 = plain Bayes; 1.0 makes trust jump 0.99<->0.01 per request)
SENSOR_WEIGHT = 0.15       # assumption: sensors watch the same traffic, so their evidence is discounted (1.0 = plain Bayes)

# --- Policy bands (allow threshold reused from Classic mode) ----------------
ALLOW_THRESHOLD = 0.72     # reused from Classic mode: trust >= this -> ALLOW (Trusted)
BLOCK_THRESHOLD = 0.45     # assumption: trust < this -> BLOCK (Isolated); in between -> STEP-UP (Suspicious)
HYSTERESIS = 0.03          # assumption: must cross a threshold by this margin to change band (prevents flicker)
QUARANTINE_TICKS = 30      # assumption: an Isolated device has its links cut for 30 ticks
PROBATION_TRUST = 0.60     # assumption: after quarantine a device restarts at 0.60 and must pass step-up before ALLOW

# --- Sliding windows and baselines ----------------------------------------
WINDOW_TICKS = 20          # assumption: count-based evidence only looks at the last 20 ticks
WARMUP_TICKS = 20          # assumption: controller learns each device's rate/payload baseline in its first 20 ticks (10 gave noisy baselines -> false rate spikes)
TOKEN_FREE_PER_WINDOW = 3  # assumption: up to 3 token requests per window are ordinary; more is evidence
RATE_SPIKE_FACTOR = 3.0    # assumption: window request count > 3x the device's own baseline is a rate spike...
RATE_MIN_EXCESS = 20       # assumption: ...and exceed it by at least 20 requests (honest windows vary by about +-8)
SCAN_DISTINCT_DSTS = 4     # assumption: > 4 distinct destinations in the window -> device_scan
SCAN_DISTINCT_PORTS = 5    # assumption: > 5 distinct ports in the window -> port_scan
PAYLOAD_FACTOR = 4.0       # assumption: payload > 4x the device's baseline mean -> unusual_payload
SENSITIVE_TARGET = 0.75    # assumption: target sensitivity >= 0.75 counts as a sensitive target

# --- Simulated clock --------------------------------------------------------
TICKS_PER_DAY = 100        # assumption: one simulated day = 100 ticks
NIGHT_START = 80           # assumption: ticks 80-99 of each day are "night" (off-hours)

# --- Sensors (rates taken from the existing Classic evidence) ---------------
SENSORS = {                # name: (true-positive rate, false-positive rate) — unchanged from Classic mode
    'IDS': (0.92, 0.04),
    'SIEM': (0.88, 0.06),
    'BM': (0.90, 0.05),    # Behavior Monitor
}

# --- Step-up challenge (environment side) -----------------------------------
MFA_PASS_HONEST = 0.97     # assumption: an honest device passes the simulated MFA 97% of the time
MFA_PASS_MALICIOUS = 0.15  # assumption: an active attacker passes it 15% of the time (e.g. stolen second factor)

# --- Agents -----------------------------------------------------------------
HONEST_REQUESTS_PER_TICK = (0, 2)  # assumption: honest devices send 0-2 requests per tick (uniform)
HONEST_TOKEN_EVERY = 15            # assumption: an honest device asks for a token about once every 15 ticks
DEFAULT_DORMANT_TICKS = 30         # assumption: attackers behave honestly for 30 ticks before activating
FLOOD_BURST = (4, 7)               # assumption: a flooding attacker sends 4-7 token requests per tick when active
NOISY_FIRMWARE_P = 0.02            # assumption: a noisy-honest device does a firmware update on ~2% of ticks
NOISY_REBOOT_P = 0.01              # assumption: ...reboots on ~1% of ticks (then 1-2 failed proofs before it recovers)
NOISY_MAINTENANCE_P = 0.015        # assumption: ...makes a legitimate maintenance access to an unlinked device on ~1.5% of ticks
REPLAY_PER_TICK = (2, 3)           # assumption: a replay attacker resends 2-3 captured proofs per tick
SCAN_PER_TICK = (3, 6)             # assumption: a scanner probes 3-6 destination/port pairs per tick
SCAN_PORTS = (22, 23, 80, 443, 502, 1883, 3389, 5683, 8080, 8443)  # assumption: commonly probed ports
COMPROMISED_PER_TICK = (1, 2)      # assumption: a compromised device makes 1-2 malicious moves per tick
EXFIL_FACTOR = 12                  # assumption: exfiltration payloads are ~12x the device's normal size
SLOW_LOW_P = 0.15                  # assumption: a slow-and-low attacker makes one malicious move on ~15% of ticks

# --- Event likelihood table -------------------------------------------------
# (P(event | legitimate device), P(event | malicious device)), per request.
# For the three ordinary kinds, P(event | legitimate) REUSES the Classic
# BEHAVIOR_LIKELIHOODS value; P(event | malicious) is set separately because an
# attacker also sends ordinary traffic, so ordinary traffic is only weak evidence.
EVENT_LIKELIHOODS = {
    'normal_telemetry':        (_CLASSIC['normal_telemetry'], 0.90),    # weak evidence of legitimacy
    'authorized_service':      (_CLASSIC['authorized_service'], 0.90),  # weak evidence of legitimacy
    'read_sensor':             (_CLASSIC['read_sensor'], 0.90),         # weak evidence of legitimacy
    'firmware_update':         (_CLASSIC['firmware_update'], 0.95),     # legitimate but noisy: slightly negative
    'reboot':                  (0.85, 0.90),   # legitimate but noisy: slightly negative
    'token_request':           (0.10, 0.70),   # applied only ABOVE the free allowance per window
    'zk_auth_fail':            (0.03, 0.40),   # honest devices rarely fail a proof (reboots); attackers often do
    'token_replay':            (0.005, 0.50),  # a reused nonce is almost never legitimate
    'rate_spike':              (0.03, 0.50),   # far above the device's own learned baseline
    'port_scan':               (0.01, 0.40),   # many distinct ports in the window
    'device_scan':             (0.01, 0.40),   # many distinct destinations in the window
    'cross_zone_access':       (0.05, 0.40),   # a zone this device's configured links never reach
    'sensitive_target_access': (0.05, 0.50),   # sensitive target WITHOUT a configured link
    'no_topology_edge':        (0.02, 0.50),   # request to a device with no configured directed link
    'unusual_payload':         (0.03, 0.40),   # payload far above the device's baseline
    'off_hours_access':        (0.10, 0.40),   # service/token request during the night window
    'step_up_pass':            (MFA_PASS_HONEST, MFA_PASS_MALICIOUS),          # passed fresh ZK + MFA
    'step_up_fail':            (1 - MFA_PASS_HONEST, 1 - MFA_PASS_MALICIOUS),  # failed it
}

# --- LLM engine in Live mode (World L) ------------------------------------------
LLM_EVERY_N = 10           # assumption: World L asks the LLM about a device after every 10 of its processed requests (was 5; fewer calls)
LLM_HARD_EVENTS = ('zk_auth_fail', 'token_replay', 'step_up_fail')  # assumption: these trigger an immediate evaluation
LLM_TIMEOUT_S = 30         # default per-call timeout in seconds (editable in the sidebar); a timeout counts as LLM_ERROR
LLM_MAX_WORKERS = 4        # concurrent LLM calls per tick (the tick waits for all of them)
LLM_RECENT_REQUESTS = 5    # how many of the device's latest requests are listed individually in the LLM context (was 10)
LLM_MIN_TICKS_BETWEEN_EVALS = 3   # no repeat evaluation of the same device within 3 ticks, unless a hard event occurs
LLM_COMPACT_CONTEXT = True        # drop zero / empty fields from window_facts (the context says omitted = 0)
LLM_MAX_OUTPUT_TOKENS = 160       # Ollama num_predict / OpenAI max_tokens: caps the length of the reply
LLM_KEEP_ALIVE = '30m'            # Ollama keep_alive: keep the model loaded between calls
LLM_NUM_CTX = 4096                # Ollama options.num_ctx: context window, set explicitly so the prompt is never truncated
LLM_THINK = False                 # Ollama "think": false for models with a thinking mode (sent only if the model accepts it)
LLM_TEMPERATURE = 0.0             # sampling temperature for Live mode
LLM_ASK_BEHAVIOR_ASSESSMENT = False  # the reply schema no longer asks for behavior_assessment (still accepted if sent)
LLM_OFFLINE_AFTER = 2      # consecutive "unreachable" errors before the UI shows "LLM offline"
PENDING_DEFAULT_ACTION = 'ALLOW'  # before a device's first LLM evaluation: ALLOW if the ZK check passes
                                  # (the same as a Bayesian device starting in the Trusted band)
REPEAT_RUNS_DEFAULT = 3    # how many repeats the "Repeat run" consistency check performs

# ----- Google Gemini (optional provider) ----------------------------------------------------------------------
GEMINI_MAX_RPM = 10            # client-side requests-per-minute cap (free tier limits are low; editable in the UI)
