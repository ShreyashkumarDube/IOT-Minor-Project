# Calculation view: how each engine turns behaviour into trust

This document covers the **Trust calculation** tab in Live mode, the attack-by-behaviour comparison and the
alarm-system breakdown. It starts with the Step-0 diagnosis written before any code was changed.

## Step 0: diagnosis (what the code stored before this work)

### Bayesian records (`core/controller.py`, `ZeroTrustController.process`)

Each processed request already stored:

- `trust_before`, `trust_after`, `band_before`, `band_after`, `action` and `reason`;
- `derived`: the anomaly events the controller found itself;
- `terms`: a list of `(label, Δ log-odds)`. These are the decay since the last logged update, the request-kind
  term (token requests only above the free allowance), one term per derived anomaly, and one term per sensor
  (`IDS!` fired / `IDS·` quiet);
- `step_up`: the fresh-proof and MFA result, the `step_up_pass/fail` term, and trust before and after it;
- `quarantined_until`, `token_count`, `zk`, `replay`, `alarms`, `port`, `bytes`.

**Missing for a full, hand-checkable calculation:**

1. Every number is rounded to 4 decimals (`terms`, `trust_before`, `trust_after`). Re-adding the rounded terms
   reproduces the stored trust only to about 1e-4, not 1e-9.
2. The `decay` term is informational. It was already applied at the start of the tick (`begin_tick`), so
   `trust_before` is the *post-decay* trust. The trust before decay is not stored.
3. The raw value behind each anomaly is not stored: window size against the rate threshold, distinct destinations
   and ports, the payload against the baseline, the device's usual zones, the target sensitivity, night.
4. The band before hysteresis (`raw_band`) and the probation state at the time of the request are not stored.

### LLM records (`core/live_llm.py`, `LLMController._apply`)

Each `llm_eval` record already stored:

- `trigger` (the schedule reason) and `latency_ms`;
- the **exact `context`** dict that was sent;
- the parsed `llm_trust`, `llm_action`, `confidence`, `reason`, `factors` and `behavior_assessment`;
- `clamped` (how many values were clamped), `ok`, `error` (type) and `message`;
- `trust_before`/`trust_after`, `decision_before`/`decision_after` and `quarantined_until`.

Each per-request record in World L stores `llm_decision_tick`: the evaluation whose decision applied.

**Missing:**

1. The **raw reply text**. `LLMEvaluator.evaluate_raw` parses the JSON and discards the text; the FakeLLM returns a
   Python dict.
2. The **model name**.
3. The system prompt is the constant `LIVE_SYSTEM_PROMPT`, so it does not need storing.
4. The **request ids** behind each window fact. `LLMLedger.window` holds facts without the request id.

### Window facts (`LLMController.build_context`)

`led.window` is World L's list of facts for the device's **non-dropped** requests with
`tick > current_tick − WINDOW_TICKS` (20). The evaluation runs at the end of the tick, so the current tick is
included. Each fact is computed from that list:

| fact | computed as |
|---|---|
| `requests` | number of requests in the window |
| `requests_per_tick` | requests ÷ min(20, ticks since enrolment) |
| `baseline_requests_per_tick` | the device's requests in its first 20 ticks ÷ 20 (frozen after warm-up) |
| `request_kind_counts`, `token_requests` | counts by request kind |
| `distinct_destinations` / `distinct_ports` | distinct `dst` (excluding the controller) / distinct ports |
| `requests_without_configured_link` | `dst` ≠ controller and (src, dst) not a configured link |
| `zone_changing_requests` | the target's zone differs from the device's own zone |
| `requests_to_sensitive_targets` | target sensitivity ≥ 0.75 |
| `max_payload_bytes_by_kind` / `baseline_mean_payload_bytes_by_kind` | the largest size per kind / the warm-up mean per kind |
| `zk_failures`, `nonce_reuse_detected`, `night_requests` | counts |
| `sensor_alarm_counts` | per sensor, the requests on which it fired |
| `step_up_challenges` | passed and failed step-ups on requests in the window |

Each count can be traced to request ids by rebuilding the window from World L's own `request` records for that
device and tick range. A test checks that the rebuilt counts equal the context values.

### Sensors (`core/sensors.py`, `core/live_sim.py`)

`LiveSimulation._handle` calls `SensorSuite.observe(action.malicious, rng('sensor'))`, using a per-request random
stream. For each sensor in order (IDS, SIEM, BM), it draws `u = rng.random()` and fires if
`u < (TP if malicious else FP)`. **Neither `u` nor `p` was recorded**; only the True/False alarms reached the
controller. The draw happens for every generated request, including those later DROPPED by a quarantine.

### Attack types (`core/agents.py`)

The agent types are `token_flood`, `credential_replay`, `scanning`, `compromised_valid` and `slow_low`, plus
`noisy_honest` and `honest`. `multi_attacker` is a **scenario** (three attackers), not an agent type. What each
attacker produces on the wire:

| agent | what is observable |
|---|---|
| `token_flood` | 4–7 extra `token_request`s to the controller per tick; the request rate rises far above the device's baseline |
| `credential_replay` | 2–3 `token_request`s per tick carrying captured proofs: its own old ones (the **nonce is reused**, ZK maths still passes) or other devices' (**ZK fails**, wrong key) |
| `scanning` | 3–6 `read_sensor` requests per tick to random devices on random ports from a list of 10 |
| `compromised_valid` | 1–2 moves per tick with a **valid proof**: lateral moves to unlinked other-zone devices, access to unlinked sensitive targets, or exfiltration (payload ×12) |
| `slow_low` | the same moves as `compromised_valid`, but only on about 15% of ticks |
| `noisy_honest` | occasional firmware updates, reboots followed by 1–2 failed proofs, and maintenance access to an unlinked device (about 1.5% of ticks) |

### Conflicts with the prompt's rules

There were none that required stopping. Two points of interpretation:

- **"Byte-for-byte identical logs."** Showing exact values requires new **display-only keys** on the records
  (`calc` on Bayesian records; `raw_reply` and `llm_model` on LLM evaluations). The golden test
  (`tests/golden_logs.py`) fingerprints every record with those keys removed, plus the rendered terminal lines, the
  log CSV and the trust history, and compares against a baseline committed **before** the change. The wall-clock
  `latency_ms` is excluded because it is never reproducible.
- **"Sensor draws next to `request_truth`."** They are stored in `LiveSimulation.sensor_draws[rid]`, an
  environment-only dict like `request_truth`. They are never passed to a controller.

## What each panel shows

Open **Trust calculation** in Live mode (pause first), then pick a device and a request.

| Panel | Shows | Source |
|---|---|---|
| 1A left, Bayesian per request | steps 1–9 (L₀, decay, every term with P(e\|legit), P(e\|attacker), ratio, ln, weight; sum; σ; clamp; band before and after hysteresis; policy; step-up; quarantine) and a log-odds waterfall | `calc` block of the record, `bayes_steps()` |
| 1A right, LLM per evaluation | the evaluation whose decision applied (or PENDING with its default rule); every window fact rebuilt and traced to request ids; exact context, system prompt, raw reply, parsed and clamped values; factor grounding; latency and model | `deciding_evaluation()`, `window_trace()`, `factor_grounding()` |
| 1B, same behaviour, two treatments | per behaviour in the last 20 ticks: Bayesian event(s) and term totals, and LLM field(s) and values; the factor that mentions it | `same_behaviour_table()` |
| 1C, how the calculations differ | update timing, memory, maths, behaviour mapping, sensors, zone/sensitive/night semantics, start, determinism, latency, failures, audit | `differences_table()` (numbers from config and the run) |
| 1D, trust over time | Bayesian solid, LLM dashed, ◆ LLM evaluations, ✕ LLM_ERROR, orange lines at the first tick of each new behaviour | trust history + records |
| 2, attacks by behaviour | the signatures; one card per device that matched a signature or was blocked; rules with values; behaviour first seen; both engines' detection, timing, trust at detection, top terms / LLM reason and factors, what each reacted to; truth-based numbers with Reveal on | `infer_attack_kind()`, `attack_cards()` |
| 3, alarm system | per sensor for the selected request: TP/FP, fired?, examiner-only truth, p and u, `u < p`, the Bayesian term formula, the LLM's `sensor_alarm_counts`; 8 combinations; whole-run observed against configured rates, precision and per-device log-odds | `request_alarm_rows()`, `alarm_combinations()`, `alarm_stats()` |

Examiner-only content (ground truth, p, u, truth-based card numbers, whole-run alarm statistics) is hidden until
**Reveal ground truth** is on.

## Worked example: seed 42, `compromised_valid_token`, DEV-008 at t=40

Request `40-DEV-008-1`: DEV-008 → DEV-002, `authorized_service`, ZK PASS. LLM world = rule-based FakeLLM.

**Bayesian.**

1. Trust before 0.9900 → L₀ = ln(0.99/0.01) = **+4.5951**.
2. No decay: DEV-008's previous update was in the same tick.
3. Evidence terms (weight 0.5):

   | term | P(e\|legit) / P(e\|attacker) | ln(ratio) | term |
   |---|---|---|---|
   | `authorized_service` | 0.97 / 0.90 | +0.0749 | **+0.0375** |
   | `no_topology_edge` (DEV-008 → DEV-002: no configured link) | 0.02 / 0.50 | −3.2189 | **−1.6094** |
   | `cross_zone_access` (target zone 'Public IoT' is not among the usual zones ['Private IoT']) | 0.05 / 0.40 | −2.0794 | **−1.0397** |

4. Sensors (weight 0.15): IDS fired 0.15·ln(0.04/0.92) = **−0.4703**; SIEM fired 0.15·ln(0.06/0.88) = **−0.4028**;
   BM fired 0.15·ln(0.05/0.90) = **−0.4336**.
5. Sum −3.9184 → L₁ = +0.6767 → σ = **0.6630**.
6. Raw band Suspicious, kept after hysteresis → policy **STEP-UP**.
7. Fresh proof PASS, MFA FAIL → `step_up_fail` = 0.5·ln(0.03/0.85) = −1.6720 → L = −0.9953 → σ = **0.2699** → Isolated → **BLOCK**.
8. Quarantined until t=70.
9. Check: +4.5951 + (−3.9184) = +0.6767 → σ(+0.6767) = 0.6630 ✔ (matches the stored trust).

**LLM (FakeLLM).** The request was decided by World L's evaluation at **t=32** (trigger *every 10 events*, the
current default). It said ALLOW with trust 0.92. All 21 window facts in that compact context are rebuilt from World
L's records, and each matches what was sent. Zero counters such as `requests_without_configured_link` are omitted
and read back as 0, because the attack only starts at t=40. So the request was **ALLOW** in the LLM world. The
FakeLLM's factor "requests within baseline" names no unusual fact. At t=45 the FakeLLM moved DEV-008 to STEP-UP
with the factor "7 requests without configured link", and the grounding check confirms that fact.

**Alarms (examiner view).** The request was malicious, so p is each sensor's TP rate: IDS p=0.92, u=0.1383; SIEM
p=0.88, u=0.2677; BM p=0.90, u=0.3472. All three u < p, so all three fired. All three firing adds 0.15·(ln(.04/.92) +
ln(.06/.88) + ln(.05/.90)) = **−1.3067** in the Bayesian world. In the LLM world they become `sensor_alarm_counts` in the next
evaluation's window.

**Attack kind.** DEV-008's observed behaviour (up to 33 suspicious valid-proof moves in one 20-tick window; 103 in the
whole run, 33 of them exfiltration-sized) matches **Compromised device with a valid key** ✔. Bayesian: quarantined in the same tick the attack started. FakeLLM:
STEP-UP after 5 ticks and never blocked; 15 malicious requests allowed (110-tick run).

## Accuracy of the behaviour signatures

Over every preset with seeds 0–9: compromised_valid 20/20, credential_replay 10/10, scanning 20/20, slow_low 8/10,
token_flood 20/20, and honest or noisy-honest device-runs not labelled as attacks 550/550.

**Version history.** Version 1 got honest devices right in only 504 of 550 runs. Two rules did not match the documented
noisy-honest behaviour:

- **Reboot glitches** were excused for a fixed 3 ticks. A device that stays idle after a reboot fails its proofs later
  than that. Moreover, a firmware update sent during the glitch period carries a *valid* proof (a simulator quirk).
- **Sensitive-target moves** counted as strong slow-and-low evidence, but maintenance access can reach sensitive targets.

Both rules were revised (reboot glitches are explained until the next valid proof on ordinary traffic; slow-and-low needs
an exfiltration-sized payload). The thresholds themselves were not tuned. The two remaining errors are slow-and-low
runs whose moves were dense enough in one window to look like a compromised device.
