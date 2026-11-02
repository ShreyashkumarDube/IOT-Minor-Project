# Demo script — ZT-IoT Sentinel (5–7 minutes)

**Before the panel arrives**
- Run `streamlit run app.py` and maximise the browser. Live Network mode opens by default.
- Leave the sidebar at its defaults: seed 42, speed 2 ticks/s, thresholds 0.72 / 0.45, **Reveal ground truth OFF**.
- **LLM (for step 6):**
  - Start Ollama with `llama3.2:3b` pulled.
  - Do one full Both-mode rehearsal beforehand: each tick waits for the LLM, so time it and choose N (sidebar → LLM settings (sidebar → Advanced)) so the attack arrives within about a minute.
  - Better still, run `python experiments/run_batch.py --mode both --scenario compromised_valid_token --seeds 3` the day before and keep `results/` open as a backup.
  - If Ollama isn't available, skip step 6.
- Have `results/summary.md` and `results/*.png` open in another window for questions.

All the tick numbers below are for **seed 42**. Every run with the same scenario and seed is identical.

---

### 1. The setup (≈45 s)
**Show:** the hero banner, Mode = Live network, the topology map.
**Say:**
> "This is a simulated IoT network: nine devices (sensors, a camera, a lock, a smart meter, an infusion pump, two gateways) and one Zero-Trust controller, the blue hexagon. The arrows are the configured links. Every request carries a fresh zero-knowledge proof. The important rule: **the controller is never told which device is an attacker.** It only sees observed events and sensor alarms, and it has to infer the attacker from behavior."

### 2. A normal day: no false blocks (≈1 min)
**Do:** Scenario = `normal_day` → Reset → Start. Let it run to about t=60.
**Point at:**
- The terminal. Every line shows the request, the ZK result, the three sensors and the trust calculation, e.g. `Bayes trust 0.99 → 0.99 (log-odds: normal_telemetry +0.04, IDS· +0.37 …) → ALLOW`.
- The status table: all green.
- Some devices are "noisy-honest": they reboot, update firmware or do maintenance. Look for a `ZK:FAIL … flags: zk_auth_fail → BLOCK (ZK proof rejected (hard gate))` line. On seed 42 the first is at t=4 (DEV-010, the edge gateway), and there are 12 in a 160-tick run, yet no honest device is ever quarantined or stepped up.

> "That request is blocked because its proof is broken, but the device stays Trusted. One glitch doesn't get a legitimate device quarantined. Over 30 seeds, honest devices on a normal day were quarantined in 1.5% of device-runs."

**Do:** Pause.

### 3. The key attack: a compromised device with a VALID key (≈2 min)
**Do:** Scenario = `compromised_valid_token` → Reset → Start. At speed 2, tick 40 arrives after about 20 s. Pause at about t=41, or use Step from t=38.
**Say while it is dormant:**
> "One of these nine devices has been compromised, but it's behaving normally for now. Can you tell which one? Neither can the controller. Every row is green."

**At t=40, point at DEV-008 (the smart meter):**
```
[t=040] DEV-008 → DEV-002  authorized_service  ZK:PASS  IDS:alert SIEM:alert BM:alert  flags: no_topology_edge, cross_zone_access
        Bayes trust 0.99 → 0.66  (… no_topology_edge −1.61, cross_zone_access −1.04, IDS! −0.47, SIEM! −0.40, BM! −0.43)  band: Trusted → Suspicious
        STEP-UP challenge → ZK:PASS MFA:FAIL (step_up_fail −1.67)  trust 0.66 → 0.27  band → Isolated → BLOCK
[t=040] DEV-008 QUARANTINED for 30 ticks — links cut
```
> "Look at the ZK column: **PASS, every time**. The attacker has the real secret, so authentication alone would let it in. But on its very first malicious move it talks to a device it has no link to, in another zone. Trust drops to the step-up band, and it fails the extra MFA check. The controller cuts its links: on the map its node turns grey and its link is dashed red."

**Do:** Decision flow → Device = DEV-008 → stage **Trust calculation**. Show the table of terms and the running trust.
> "This is the exact calculation, term by term. Nothing is hidden."

**Do:** Let it run to t=70.
```
[t=070] DEV-008 released from quarantine → PROBATION (trust reset to 0.60; must pass step-up)
[t=070] DEV-008 → CONTROLLER  read_sensor … STEP-UP challenge → ZK:PASS MFA:PASS … trust 0.81 → 0.92 → ALLOW
[t=070] DEV-008 → DEV-004  authorized_service … flags: no_topology_edge, cross_zone_access, sensitive_target_access
        Bayes trust 0.97 → 0.31  → BLOCK (band Isolated)
[t=070] DEV-008 QUARANTINED for 30 ticks — links cut
```
> "After 30 ticks it's released on probation. This time it gets lucky and passes MFA, so it's trusted again. But a few requests later it goes after the critical industrial actuator, which it has no link to, and it's straight back in quarantine. Trust is earned continuously, not granted once."

### 4. Reveal the ground truth (≈30 s)
**Do:** Toggle **Reveal ground truth** on.
> "Only now do we show the answer: DEV-008 is `compromised_valid`, attack start t=40. The controller never had this column. A test in the suite checks that no role or label ever reaches the controller or the LLM."

### 5. Results tab (≈1 min)
**Do:** Pause → **Results**.
**Point at:** attackers detected 1/1, ticks to BLOCK = 0 (same tick), honest quarantined 0%, the confusion matrix, and the trust curves with the dotted attack-start line.
> "These metrics compare decisions with ground truth after the fact, and they're never fed back. Over 30 seeds per scenario, every attacker type was caught. The stealthy slow-and-low attacker took a median of 8.5 ticks, up to 64."

Mention the export buttons, and that `experiments/run_batch.py` reproduces the report's figures.

### 6. Bayesian vs LLM — two isolated worlds (≈1.5 min)
**Do:** Sidebar → **Engines: Both — isolated comparison** → Reset → Start (same scenario and seed).
**Point at:**
- The **two terminals**. The left is the Bayesian world; the right is the LLM world, with purple `LLM eval(DEV-…)` lines showing trust, action, confidence, reason and factors.
- The **status table**: `Bayes …` columns and `LLM …` columns side by side. `PENDING` means the LLM hasn't assessed that device yet. `Agree` shows yes / no.
- The trust chart: solid lines are Bayes, dashed are LLM.

> "These are two separate simulations with exactly the same traffic: the same scenario, seed and random streams per device. The left world is run by the Bayesian controller, the right world by an LLM controller. Each makes its own decisions and enforces them in its own world, and no number ever passes between them. The LLM only sees facts its own world observed: counts, destinations, ZK results, nonce reuse, alarms. It never sees the Bayesian score, and never a role."

**Do:** Pause → **Bayesian vs LLM** tab.
> "Here every parameter is side by side: detection, time-to-detect in ticks, false positives on honest and noisy devices, trust separation and calibration, per-request decisions, agreement, latency in milliseconds, errors and consistency. Each row says which engine did better."

If the LLM is unreachable: the red **LLM offline** banner shows, and the LLM columns stay `PENDING`/`LLM_ERROR`. Say so: "There's no fallback. If the LLM can't answer, we don't pretend it did."

### 7. Close (≈15 s)
> "So: Zero Trust means never trust and always verify. Authentication is necessary but not sufficient, and trust has to be earned continuously from behavior. Everything here is reproducible from a scenario and a seed."

---

## Likely examiner questions

**How does the controller know who the attacker is?**
It doesn't. It gets `ObservedEvent` objects (who, to whom, port, bytes, the ZK proof) and True/False sensor alarms. Ground truth exists only in the simulated environment. A test checks the event fields, the controller's method signatures, its imports, and every LLM context.

**How do you know the LLM isn't just copying the Bayesian answer?**
- They run in separate worlds. The LLM context is built only from the LLM world's own observations.
- Tests check this four ways:
  - **Breaking one engine doesn't affect the other.** If the Bayesian controller is made to crash, an LLM-only run still completes; if the LLM fails, the Bayesian run is unchanged.
  - **The LLM columns show only the LLM's own answers.** A fake LLM that always answers 0.37 / STEP-UP shows 0.37 / STEP-UP in every LLM cell and log line, never a Bayesian number.
  - **The LLM context never contains Bayesian data.** Every context from full runs is scanned for Bayesian trust, band, decision or labels.
  - **No state is shared.** The two worlds' objects are all separate.
- If the LLM fails, its cells show `PENDING` / `LLM_ERROR`, never a substituted value.

**Is the comparison fair?**
- **Same traffic.** Both worlds see the same requests and sensor alarms, generated from per-device random streams.
- **Same rules.** Quarantine and probation work the same way in both worlds, and the ZK hard gate applies to both.
- **Same time scale.** Detection is measured in ticks, with latency reported separately.
- **The real difference: the LLM is slower.** It decides every N requests rather than on every request, so its time-to-detect includes that wait. That's a real property of using an LLM, and it's reported.

**Isn't a valid ZK proof enough?**
No. `compromised_valid_token` passes ZK on every request and is still quarantined within 0–2 ticks of starting (30/30 seeds), because of its behavior: lateral movement, sensitive targets and unusual payloads.

**Why log-odds Bayes?**
Each piece of evidence becomes an additive term, `log P(e|legit)/P(e|malicious)`, so the update is explainable line by line and cheap enough to run on every event.

**Where do the numbers come from?**
They're hand-set modelling assumptions, all in `core/live_config.py` with a comment each. The batch runner lets you change them and re-measure. This is a simulator, not a trained detector.

**Why are the evidence weights 0.5 and 0.15 instead of 1?**
With plain Bayes, the three sensors alone moved the log-odds by about ±7 per request. Trust jumped from 0.99 to 0.01 and back on consecutive requests, and an honest device fell to 0.016 after two false alarms at once. The pieces of evidence about one request are correlated, so treating them as independent overcounts. Tempering is a standard correction.

**Aren't the sensors cheating?**
Partly, and it's reported. They fire with their TP/FP rates based on whether the request really was malicious. In the ablation with sensors off, attackers are still caught 100% of the time from behavior alone, but honest quarantine rises from about 1.5% to about 43% on a normal day. So the low false-positive rate depends on the optimistic sensor model. That's a stated limitation.

**What causes the remaining false positives?**
Mostly noisy-honest maintenance access to an unlinked device in another zone, which looks exactly like lateral movement (13 of 16 honest quarantines across 30 seeds × 6 scenarios). A failed step-up MFA (honest devices fail 3% of the time) caused 2 of them. A quarantined honest device recovers through probation. In a real deployment, maintenance windows would be pre-authorised, which would remove most of these.

**Why not just use the LLM?**
In the LLM world it *does* decide, and that's the point of the comparison. Compare the two engines on the Bayesian vs LLM tab:
- **Latency:** hundreds of ms to seconds per call against microseconds.
- **Consistency:** run Repeat run. The Bayesian world is 100% identical; the LLM's real consistency is measured.
- **Auditability:** term-by-term Bayesian updates against the LLM's returned factors. Factor overlap checks whether those factors match facts that really occurred.

**What if an attacker stays quiet?**
Then it isn't attacking yet, and every attacker here starts dormant. The slow-and-low attacker hides rare malicious moves in normal traffic. It was still caught, but later (median 9, max 42 ticks), mainly when it ended up in STEP-UP and failed the MFA.

**Why does trust decay toward the prior?**
So a device can recover after a burst ends, and so trust never saturates at 0 or 1. Count-based evidence also uses a 20-tick sliding window, so one burst isn't punished forever.

**What is hysteresis?**
A device must cross a threshold by 0.03 before it changes band, so it doesn't flicker between ALLOW and STEP-UP.

**Is it reproducible?**
Yes. One seeded RNG drives agents, sensors, ZK randomness and MFA outcomes. The tests check that the same scenario and seed produce an identical log and trust history.

**Could this run on a real network?**
Not as it is. The traffic, sensors and likelihoods are synthetic and the ZK group is tiny. A real system would need measured likelihoods, correlated real sensors, production cryptography and formally specified policies, with the LLM kept advisory.
