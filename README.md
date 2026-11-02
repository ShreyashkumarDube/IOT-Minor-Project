# ZT-IoT Sentinel
### Zero-Trust Authentication and Continuous Trust Evaluation for IoT Networks

A research-oriented IoT security simulator that authenticates every request and continuously evaluates device trust using two independent engines:

- **Bayesian trust engine** — updates trust from controller-derived evidence and simulated IDS/SIEM/behavior-monitor alarms.
- **LLM trust engine** — evaluates a facts-only JSON summary using a local Ollama model, Google Gemini API, or an OpenAI-compatible endpoint.
- **Twin-world comparison** — runs both engines against identical seeded traffic in isolated worlds, then compares their decisions and metrics.

> **Scope and safety:** This is a laptop-friendly simulation. It does not capture real network traffic or control physical devices. The small cryptographic parameters are for demonstration and are not production-grade. Do not use the simulator as a real security boundary.

## Features

- Simulated IoT topology with public, private, and critical zones
- Nonce-bound Schnorr-style proof on each request
- Continuous Bayesian trust updates in log-odds space
- Graded policy decisions: **ALLOW**, **STEP-UP**, and **BLOCK** with quarantine
- Independent LLM evaluation with strict JSON validation and no Bayesian fallback
- Five attacker behaviours: token flood, credential replay, scanning, compromised device with valid credentials, and slow-and-low
- Live dashboard with topology, trust curves, event terminal, calculation details, and results
- Reproducible scenarios and seeded random streams
- Batch experiments and automated tests

## Quick start

### Requirements

- Python **3.10 or newer**
- pip
- A modern browser
- Optional: [Ollama](https://ollama.com/) for local LLM evaluation, or a Gemini API key / OpenAI-compatible endpoint

### 1. Get the repository

Clone your GitHub repository (replace the URL with your repository URL):

```bash
git clone https://github.com/<YOUR-USERNAME>/ZT-IoT-Sentinel.git
cd ZT-IoT-Sentinel
```

Alternatively, download the repository ZIP from GitHub and extract it.

### 2. Create and activate a virtual environment

**Windows PowerShell**
```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, use Command Prompt:
```bat
.venv\Scripts\activate.bat
```

**macOS / Linux**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Launch the dashboard

```bash
streamlit run app.py
```

Streamlit prints a local address (usually `http://localhost:8501`). Open it in your browser.

Convenience launchers are also included:
- Windows: double-click `run_windows.bat`
- macOS/Linux: `bash run_linux_mac.sh`

## Try the simulation

1. Open the **Live Network** view.
2. Choose a scenario, such as `normal_day` or `compromised_valid_token`.
3. Select **Bayesian only**, **LLM only**, or **Both — isolated comparison**.
4. Set a seed (for repeatable runs, keep the same seed).
5. Press **Start**. Use **Pause** and **Step** to inspect events one tick at a time.
6. Explore the topology, device state table, trust curves, terminal log, and **Results** tab.
7. Open **Trust calculation** to inspect the evidence and Bayesian arithmetic or the LLM context, raw response, and validation.

### Suggested demonstration

- Start with `normal_day` to see ordinary device behaviour.
- Reset and run `compromised_valid_token` to observe a device that can still authenticate but behaves maliciously.
- Select **Both — isolated comparison** to compare the engines on identical traffic.
- Use **Trigger attack** to introduce an attack during a live run.

The controller does not receive attacker labels. Ground truth is reserved for scoring after the run.

## Configure an LLM provider (optional)

The Bayesian simulator and dashboard can run without an LLM provider. To use an LLM, select the provider in the sidebar's LLM settings.

### Option A — Ollama (local)

1. Install Ollama from [ollama.com](https://ollama.com/).
2. Start Ollama and download the default model:

   ```bash
   ollama pull llama3.2:3b
   ```

3. In the app, select **Ollama** and model `llama3.2:3b`.
4. Optionally press **Warm up model** before starting the simulation.

The local Ollama service must remain running while the app uses it.

### Option B — Google Gemini API

1. Install the optional provider package if it is not already installed:

   ```bash
   pip install google-genai
   ```

2. Create an API key in [Google AI Studio](https://aistudio.google.com/apikey).
3. Copy `.env.example` to `.env` and add your key, or set `GEMINI_API_KEY` in your environment:

   **Windows PowerShell**
   ```powershell
   Copy-Item .env.example .env
   $env:GEMINI_API_KEY="YOUR_API_KEY"
   ```

   **macOS / Linux**
   ```bash
   cp .env.example .env
   export GEMINI_API_KEY="YOUR_API_KEY"
   ```

4. In the app, choose **Google Gemini (API)** and select a model supported by your key.

Keep `.env` private. It is excluded by `.gitignore`. Never commit API keys.

### Option C — OpenAI-compatible endpoint

Run a compatible server (for example, `llama-server`) and configure the provider in the app with its endpoint, model name, and optional API key. The exact endpoint and supported structured-output features depend on the server.

### LLM evaluation notes

- The LLM receives observed facts, not attack labels or Bayesian trust values.
- Outputs are validated against the expected schema. Invalid output or timeouts are recorded as `LLM_ERROR`; the previous LLM verdict is retained.
- The rule-based stand-in is intended for pipeline tests only. **It is not a real LLM and its results must not be presented as LLM performance.**
- Cloud-provider experiments send simulated context to that provider. Do not put real personal, institutional, or production data into the simulator.

## Run tests

Install the project requirements, then:

```bash
python -m pytest -q tests
```

Some tests exercise provider integration and may be skipped or require provider configuration. The test suite is the best first check after changing code.

## Run experiments

### Bayesian-only batch

```bash
python experiments/run_batch.py --scenario all --seeds 30
```

### Behaviour-only ablation (disable sensor contribution)

```bash
python experiments/run_batch.py --scenario all --seeds 30 --sensor-weight 0 --out results/ablation_no_sensors
```

### Twin-world comparison with a real local model

Start Ollama first, then:

```bash
python experiments/run_batch.py --mode both --scenario all --seeds 3 --provider ollama --model llama3.2:3b --out results/comparison
```

### Pipeline check without a real model

```bash
python experiments/run_batch.py --mode both --scenario all --seeds 3 --fake-llm --out results/fake_llm_check
```

Outputs are written under `results/` (ignored by Git by default). Use the exact provider and model ID when reporting LLM results.

## Repository structure

```text
.
├── app.py                 # Streamlit application entry point
├── live_ui.py             # Live simulation dashboard
├── calc_ui.py             # Trust calculation and evidence views
├── core/
│   ├── live_config.py     # Simulation and policy parameters
│   ├── events.py          # Observed events and simulation clock
│   ├── agents.py          # Device and attacker behaviour
│   ├── sensors.py         # Simulated IDS/SIEM/behaviour monitor
│   ├── controller.py      # Bayesian trust and enforcement policy
│   ├── live_sim.py        # Single-world simulation
│   ├── twin.py            # Isolated twin-world execution
│   ├── live_llm.py        # LLM context, assessment, and enforcement
│   └── metrics.py         # Evaluation metrics
├── scenarios/              # Scenario configurations
├── experiments/            # Batch and model-speed experiments
├── tests/                  # Automated tests
├── docs/                   # Report and presentation
├── requirements.txt
└── .env.example
```

## Reported results and interpretation

The project report describes seeded Bayesian experiments across seven scenarios. The reported simulator results include attacker detection in all attacker runs and a low honest-device quarantine rate when sensor evidence is enabled. Slow-and-low behaviour takes longer to detect than overt attacks.

These are **simulation results under the project's configured assumptions**, not guarantees for real IoT deployments. The LLM comparison should be rerun with the selected real provider/model; results from the rule-based stand-in are only pipeline validation.

## Troubleshooting

- **`streamlit` not found:** activate `.venv` and reinstall with `python -m pip install -r requirements.txt`.
- **Wrong Python version:** check `python --version`; use Python 3.10+.
- **PowerShell activation blocked:** use `.\.venv\Scripts\activate.bat` in Command Prompt, or adjust your local execution policy if appropriate.
- **Ollama offline/model missing:** start Ollama and run `ollama pull llama3.2:3b`.
- **Gemini errors:** confirm `GEMINI_API_KEY`, model availability, and provider quota/rate limits.
- **LLM is slow:** warm up the model, reduce evaluation frequency in settings, or use a smaller local model. The Bayesian engine does not require an LLM.

## Academic context

**Project:** Zero-Trust Authentication and Continuous Trust Evaluation for IoT Networks Using Bayesian and Large Language Model Engines  
**Institution:** Department of Computer Science & Engineering, Motilal Nehru National Institute of Technology Allahabad  
**Supervisor:** Neelam Dayal, Assistant Professor  
**Authors:** Shreyash Kumar Dube, Sparsh Singh, Siddhesh Choubey, and Shubham Daya Vadher

## License

No license has been added yet. Unless a license is included in this repository, all rights remain with the authors and reuse is not automatically permitted. Add a license only after confirming the team's preference.
