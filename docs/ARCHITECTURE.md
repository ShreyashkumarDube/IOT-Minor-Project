# Architecture — ZT-IoT Sentinel Live Zero-Trust Controller

## 1. Big picture

```text
 ENVIRONMENT (knows ground truth)                         CONTROLLER (does not)
 ┌──────────────────────────────────────┐
 │ agents.py  honest / noisy-honest /   │  Action (dst, kind, port, bytes, proof, malicious*)
 │            5 attacker types          │──────────────┐
 │ live_sim.py seeded RNG, clock,       │              │  * `malicious` never leaves the environment
 │            fresh nonce + real ZK     │              v
 │            prove() per request       │      ┌──────────────────┐   ObservedEvent (tick, src, dst, type,
 │ sensors.py IDS/SIEM/BM fire with     │      │  network tap     │   port, bytes, zones, nonce, proof)
 │            TP or FP rate per request │─────>│  (drops truth)   │────────────┐
 └──────────────────────────────────────┘ alarms└──────────────────┘            v
                                          True/False                ┌────────────────────────────┐
                                                ───────────────────>│ controller.py               │
                                                                    │  verify ZK, remember nonces │
                                                                    │  sliding window, baselines  │
                                                                    │  derive anomalies           │
                                                                    │  log-odds Bayes + decay     │
                                                                    │  bands, step-up, quarantine │
                                                                    └──────────┬─────────────────┘
                                                                               │ record (terms, trust, action)
             ┌─────────────────────────────┬───────────────────────────────────┼──────────────────────┐
             v                             v                                   v                      v
   live terminal / status / topology   live_llm.py (every N events or    metrics.py (uses ground   exports
   trust chart / decision flow         band change, background thread)   truth AFTER decisions)    CSV/JSON/MD
```

The controller module never imports agents, sensors or the simulator, and a test checks this.

### Twin worlds (Bayesian vs LLM, fully isolated — `core/twin.py`)

```text
                    same scenario + same seed
                 ┌────────────┴────────────┐
        World B (Bayesian)          World L (LLM)
        environment copy B          environment copy L
        ZeroTrustController         LLMController
        own ledger, nonces,         own ledger, nonces,
        window, baselines,          window, baselines,
        policy + quarantine         LLM decisions + quarantine
        own event log               own event log
                 └────────────┬────────────┘
        comparison layer (core/comparison.py): reads both, writes neither
        + ground truth, used ONLY for scoring
```

- **Run modes:** Bayesian only / LLM only / Both. In a single-engine mode the other world is not created.
- **Nothing is shared.** Each world deep-copies the scenario. A test checks that ledgers, nonce sets, windows, logs and devices are separate objects, and that changing one doesn't affect the other.
- **Identical inputs.** Every random draw comes from `stream_rng(seed, device, tick, stream, request index)`: agent behavior, sensor alarms, ZK proof randomness, challenge proofs and MFA outcomes. There is no global RNG whose consumption order could drift, so both worlds generate exactly the same traffic.
- **What differs.** The traffic each world *processes* differs only when one world's own quarantine drops requests. Agents don't react to enforcement.
- **Lockstep.** Both worlds advance one tick at a time.
- **Every record** carries `engine` and a request id `rid`. The rid is the same in both worlds, which lets the comparison line up the two decisions on each request.

## 2. What the controller sees

`ObservedEvent` fields: `tick, src, dst, event_type, details{port, bytes, zone_from, zone_to, sensitivity}, zk_result, nonce, proof`.
Creating one with a ground-truth key in `details` (role, malicious, attack_type …) raises an error.

Enrollment (`ZeroTrustController.enroll`) accepts only `device_id, public_key, zone, target_sensitivity`.

## 3. Events and likelihoods (`core/live_config.py`)

Request kinds come from the wire. The anomalies are **derived by the controller** from what it has observed: its own nonce memory, sliding window, learned baselines, topology and clock.

| Event | P(e \| legit) | P(e \| malicious) | log ratio | × weight 0.5 | Derived from |
|---|---|---|---|---|---|
| `normal_telemetry` | 0.98 | 0.90 | +0.09 | +0.04 | request kind |
| `authorized_service` | 0.97 | 0.90 | +0.07 | +0.04 | request kind |
| `read_sensor` | 0.96 | 0.90 | +0.06 | +0.03 | request kind |
| `firmware_update` | 0.90 | 0.95 | −0.05 | −0.03 | request kind (legitimate but noisy) |
| `reboot` | 0.85 | 0.90 | −0.06 | −0.03 | request kind (legitimate but noisy) |
| `token_request` | 0.10 | 0.70 | −1.95 | −0.97 | only above 3 per 20-tick window |
| `zk_auth_fail` | 0.03 | 0.40 | −2.59 | −1.30 | `verify()` with the enrolled public key fails |
| `token_replay` | 0.005 | 0.50 | −4.61 | −2.30 | nonce already seen from this device |
| `rate_spike` | 0.03 | 0.50 | −2.81 | −1.41 | window count > max(3×, +20) of the device's own baseline |
| `port_scan` | 0.01 | 0.40 | −3.69 | −1.84 | > 5 distinct ports in window |
| `device_scan` | 0.01 | 0.40 | −3.69 | −1.84 | > 4 distinct destinations in window |
| `cross_zone_access` | 0.05 | 0.40 | −2.08 | −1.04 | target zone not reached by any configured link of the source |
| `sensitive_target_access` | 0.05 | 0.50 | −2.30 | −1.15 | target sensitivity ≥ 0.75 **and** no configured link |
| `no_topology_edge` | 0.02 | 0.50 | −3.22 | −1.61 | no configured directed link src → dst |
| `unusual_payload` | 0.03 | 0.40 | −2.59 | −1.30 | bytes > 4× the baseline for that request kind |
| `off_hours_access` | 0.10 | 0.40 | −1.39 | −0.69 | service/token request in the night window |
| `step_up_pass` | 0.97 | 0.15 | +1.87 | +0.93 | fresh ZK on challenge nonce + MFA passed |
| `step_up_fail` | 0.03 | 0.85 | −3.34 | −1.67 | either failed |

For the three ordinary kinds, P(e | legit) reuses the Classic `BEHAVIOR_LIKELIHOODS` value.

Sensors (TP/FP rates unchanged from Classic): IDS 0.92/0.04, SIEM 0.88/0.06, BM 0.90/0.05. Weighted by 0.15, an alarm adds −0.47/−0.40/−0.43 and silence adds +0.37/+0.31/+0.34.

Baselines (request rate, payload per kind) are learned in each device's first 20 ticks and then frozen. This is why every attacker is dormant for at least 30 ticks.

## 4. Trust update

Per device, in log-odds:

```
L = logit(trust)
once per tick:      L = λ·L + (1−λ)·logit(prior)                     λ = 0.98, prior = 0.80 (same for everyone)
per request:        L = L + Σ_events  w_e · log(P(e|legit)/P(e|mal))  w_e = 0.5
                      + Σ_sensors w_s · log(P(alarm_obs|legit)/P(alarm_obs|mal))   w_s = 0.15
trust = clamp(sigmoid(L), 0.01, 0.99)
```

**Why the weights?** With w = 1 (plain Bayes), the three sensors alone move L by about ±7 on every request. Trust jumped 0.99 → 0.01 on one malicious request and back to 0.92 on the next clean one, and an honest device fell to 0.016 after two simultaneous false alarms. Several pieces of evidence about the same request are correlated, so multiplying them as if independent overstates the evidence. Tempering with weights is a standard correction.

**Worked example** (token-flood attacker DEV-005, seed 7, the default Classic network; its attack starts at t=30):

```
[t=030] DEV-005 → CONTROLLER  token_request ×2 (window 20)  ZK:PASS  IDS:alert SIEM:alert BM:-
        Bayes trust 0.99 → 0.98  (log-odds: IDS! −0.47, SIEM! −0.40, BM· +0.34)
[t=030] DEV-005 → CONTROLLER  token_request ×3 (window 20)  ZK:PASS  IDS:alert SIEM:alert BM:alert
        Bayes trust 0.98 → 0.94  (log-odds: IDS! −0.47, SIEM! −0.40, BM! −0.43)
[t=030] DEV-005 → CONTROLLER  token_request ×4 (window 20)  ZK:PASS  IDS:alert SIEM:alert BM:alert
        Bayes trust 0.94 → 0.62  (log-odds: token_request×4 −0.97, IDS! −0.47, SIEM! −0.40, BM! −0.43)  band Trusted → Suspicious
        STEP-UP challenge → ZK:PASS MFA:FAIL (step_up_fail −1.67)  trust 0.62 → 0.23  band → Isolated → BLOCK
[t=030] DEV-005 QUARANTINED for 30 ticks — links cut
```

Check the third request by hand:
- logit(0.94) = ln(0.94/0.06) = 2.75.
- 2.75 − 0.97 − 0.47 − 0.40 − 0.43 = 0.48, and sigmoid(0.48) = 0.62.
- The first three token requests are within the free allowance of 3, so only the sensors move trust. The fourth adds the token-burst term.
- The device's ZK proof passed every time. It was caught from behavior (the token burst), the sensor alarms and the failed MFA.

## 5. Policy

| Condition (first match wins) | Action |
|---|---|
| ZK proof fails, or nonce replayed | BLOCK this request (hard gate) |
| device on probation | STEP-UP |
| trust ≥ 0.72 (Trusted) | ALLOW |
| 0.45 ≤ trust < 0.72 (Suspicious) | STEP-UP: fresh ZK proof on a challenge nonce + simulated MFA |
| trust < 0.45 (Isolated) | BLOCK + quarantine |

- **Hysteresis ±0.03:** a device must cross a threshold by 0.03 to change band.
- **Quarantine:** for 30 ticks, every request to or from the device is DROPPED and logged, with no trust update. The device is then on **probation**: trust is reset to 0.60, and it must pass a step-up before it can be ALLOWed again.
- **MFA (environment side):** an honest device passes with probability 0.97, an active attacker with 0.15. The controller only sees PASS or FAIL.
- The thresholds can be edited live in the sidebar. Classic `evaluate_trust()` / `policy_decide()` are unchanged.

## 6. Agents (`core/agents.py`)

| Type | Behavior |
|---|---|
| honest | 0–2 ordinary requests per tick along configured links or to the controller; a token request about every 15 ticks; telemetry only at night |
| noisy_honest | honest, plus firmware updates (2% of ticks), reboots (1%, followed by 1–2 damaged proofs) and maintenance access to an unlinked device in another zone (1.5%) |
| token_flood | after dormancy: 4–7 token requests per tick (rate spike) |
| credential_replay | 2–3 per tick of resent captured proofs: its own (nonce reuse) or another device's (wrong key) |
| scanning | 3–6 probes per tick to random devices on common ports |
| compromised_valid | valid secret, so ZK always passes; 1–2 per tick of lateral moves, sensitive-target access or exfiltration (12× payload) |
| slow_low | the same moves as compromised_valid, but one only on about 15% of ticks |

All attackers send honest cover traffic and are dormant (behave honestly) until `attack_start`.

## 7. The LLM world (`core/live_llm.py`)

The LLM is a full, independent evaluator in its own world. **Nothing computed by the Bayesian side ever reaches it.**

**What World L observes for itself (facts, no anomaly labels):**
- Verifying each ZK proof against the enrolled public key, and remembering nonces (so it detects reuse itself).
- Whether a configured link exists, whether the request crosses zones, and the target's sensitivity.
- Payload bytes, port, night or day, and sensor alarms (True/False).
- The device's own rate and payload baseline, learned in its first 20 ticks.
- Its own step-up outcomes.

**What the LLM receives per evaluation**, built only from World L:
- The device's public attributes: id, type, protocol, firmware, zone, sensitivity. No name, role or status.
- Its configured links.
- The LLM's own previous trust and decision, and its enforcement state (probation, times quarantined).
- A 20-tick fact summary: requests and rate against baseline, kind counts, token requests, distinct destinations and ports, requests without a configured link, zone-changing requests, requests to sensitive targets, max payload per kind against baseline, ZK failures, nonce reuse, night requests, sensor alarm counts, step-up passes and fails.
- The last 10 requests individually.
- Sensor TP/FP rates, and the policy thresholds as **guidance**.

It never receives Bayesian trust, band, terms or decisions. There is no option to add them. A test walks every context of full runs and checks for forbidden keys and values.

**Decisions and enforcement (World L enforces the LLM's own action):**

| Situation | Action on the next request |
|---|---|
| ZK proof fails or nonce replayed | BLOCK that request (cryptographic hard gate, same in both worlds) |
| on probation | STEP-UP (fresh ZK on a challenge nonce + simulated MFA) |
| LLM decided ALLOW / STEP-UP | apply it until the next evaluation |
| LLM decided BLOCK | quarantine the device for 30 ticks, then probation (same rules as World B) |
| no evaluation yet (`PENDING`) | ALLOW if the ZK check passes (the same default as a new Bayesian device in the Trusted band) |

**Scheduling and pacing:**
- A device is evaluated after every N of its processed requests (default 5).
- It is also evaluated at the end of the tick right after a *hard event*: `zk_auth_fail`, `token_replay` or `step_up_fail`, as well as on release from quarantine and when probation ends.
- All due devices are evaluated concurrently, and the tick waits for them, up to the timeout. The UI shows "LLM evaluating N devices…".
- Detection times are measured in ticks, which is fair to both engines. Latency is reported separately, in ms.

**Failures, never a Bayesian fallback:**
- A timeout, unreachable endpoint, reply without JSON, or a reply missing `trust_score` or a valid `action` is recorded as `LLM_ERROR <type>` and counted.
- The device keeps its last LLM decision (or `PENDING`) and is retried at its next scheduled point.
- After repeated "unreachable" errors the UI shows **LLM offline** in red, and World L keeps running.
- Out-of-range numbers are clamped and counted.

**The LLM's trust between evaluations** is its last returned value, so in charts it is a step line. Before the first evaluation it is `PENDING`, shown as a gap, never as a number.

## 8. Metrics (`core/metrics.py`) and the comparison (`core/comparison.py`)

The same definitions apply to both worlds. Each world's metrics come from its own log; ground truth is used only after the run.

| Metric | Definition |
|---|---|
| Detected | quarantined at or after the attack start |
| Time to STEP-UP / BLOCK | ticks from the attack start to the first flag (a challenge, the Suspicious/Isolated state, or an LLM STEP-UP/BLOCK verdict) / to the first quarantine |
| False positive (block / step-up) | an honest or noisy-honest device that was ever quarantined / ever flagged. Reported separately for honest and noisy-honest devices. |
| Hard-gate blocks | single requests blocked because the ZK proof failed. Counted separately; not device-level false positives. |
| Confusion matrix (devices) | positive = quarantined at least once |
| Per-request truth | whether the **sending device was an active attacker** at that tick |
| Policy action | a world's decision on a request before any step-up outcome. A quarantine drop counts as BLOCK. |
| Agreement | both worlds took the same policy action on the same request id |
| Brier score | mean of (trust − legit)², where legit = 1 for honest devices and dormant attackers, and 0 for active attackers. PENDING ticks are skipped and counted. |
| Volatility | standard deviation of the tick-to-tick trust change, per device |
| Recovery time | for noisy-honest devices, ticks from leaving Trusted until Trusted again |
| Latency | Bayesian: controller time per request. LLM: wall time per evaluation. |
| Confidence on correct / incorrect verdicts | an LLM verdict is correct if it is not ALLOW for an active attacker, or ALLOW otherwise |
| Factor overlap | keyword match between the LLM's factors and reason and the facts really present in its window. A heuristic: a negated mention still counts. |
| Consistency | the same scenario and seed run K times. Reports the share of requests with an identical decision in every run, and the mean trust standard deviation. |

The Bayesian vs LLM tab and `comparison_report.md` show **every parameter** as Bayesian | LLM | difference | better. `run_batch.py --mode both --seeds N --repeat K` reports mean ± std per parameter across seeds.

## 8b. Calculation view (`core/behaviour_analysis.py`, `calc_ui.py`)

This layer is display and examiner only. It reads finished logs and is never imported by a controller (tested).

- **Bayesian.** Each request record has a display-only `calc` block: the unrounded L before and after, every term, the
  raw band before hysteresis, and the observed values behind each anomaly (window size and threshold, distinct
  destinations and ports, payload against the baseline, usual zones, target sensitivity, night). `bayes_steps()` lays
  out steps 1–9, and a test re-adds the terms to 1e-9.
- **LLM.** Evaluation records also keep `raw_reply` and `llm_model`. `window_trace()` rebuilds every `window_facts`
  field from World L's own request records and lists the contributing request ids. `factor_grounding()` checks each
  stated factor against the window.
- **Environment.** `LiveSimulation.sensor_draws[rid]` keeps, per sensor, the rate p used and the random draw u. It sits
  next to `request_truth` and is never passed to a controller. `SensorSuite.observe_detailed()` consumes the random
  stream exactly like `observe()`.
- **Attack kinds.** `infer_attack_kind()` applies behaviour signatures to observed facts only (`wire_observations()`).
  The true kind is used only in `score_inference()`, `attack_cards(reveal=True)` and the batch table.
- **No decision changed.** `tests/golden_logs.py` fingerprints every record, terminal line, log CSV and trust history,
  recorded before this layer existed. The display-only keys are removed before hashing.

## 9. Assumptions and limitations

1. Every number in `core/live_config.py` is a hand-set modelling assumption.
2. **The sensors are driven by ground truth** and treated as conditionally independent. **Ablation** (`--sensor-weight 0`, 30 seeds): attackers are still caught in 100% of runs from behavior alone, but honest quarantine rises from 0.6–1.7% to 16–50% (43% on `normal_day`). Most of the low false-positive rate therefore comes from the optimistic sensor model. With sensors, every clean request adds about +1.0 in log-odds ("no alarm" ×3), which cushions legitimate-but-odd events. Without them, a noisy-honest device's maintenance access to an unlinked device in another zone (no_topology_edge + cross_zone_access + sometimes sensitive_target_access, −2.7 to −3.8 in log-odds) or a reboot's failed proofs push it straight below 0.45. In the ablation runs (normal_day, 30 seeds), those two causes account for the large majority of honest quarantines; a failed step-up MFA and false rate spikes account for the rest.
3. The attacks are blatant by design (several malicious requests per tick), so detection is immediate. Slow-and-low is the only stealthy case (median 8.5, maximum 64 ticks, Bayesian).
4. Cross-zone and sensitive-target access along configured links is treated as authorised. Otherwise ordinary sensor → gateway → actuator traffic would be penalised constantly.
5. Detection time is measured in ticks, not seconds. The traffic model is synthetic.
6b. The LLM's trust only changes at evaluations, every N requests or on hard events, so its curves are step functions. Its time-to-detect includes up to N requests of waiting. That cost is part of the design being compared.
6. ZK parameters are tiny (Q=113) and for teaching only.
7. LLM verdicts are nondeterministic. Batch LLM runs are slow and optional.

## 10. Classic mode

The original pipeline is unchanged: ZK authentication, then Bayesian TE/PE (`evaluate_trust`, `policy_decide`) and the LLM TE/PE on one request. In Classic mode the `role` label feeds the suspicious flag, which is why the UI labels it "demonstration only".
