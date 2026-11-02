"""Headless multi-seed experiments for the report.

Examples:
    python experiments/run_batch.py --scenario all --seeds 30                         # Bayesian only (default)
    python experiments/run_batch.py --scenario all --seeds 30 --sensor-weight 0       # ablation
    python experiments/run_batch.py --mode both --scenario all --seeds 10 --repeat 3  # isolated comparison, real LLM
    python experiments/run_batch.py --mode llm --scenario replay_attack --seeds 5     # LLM world only
    python experiments/run_batch.py --mode both --seeds 3 --fake-llm                  # pipeline check, NOT a real result

Mode bayesian/llm writes the single-world files below for that world. Mode both writes:
    comparison_runs.csv       one row per (scenario, seed, parameter): Bayesian, LLM
    comparison_mean_std.csv   per scenario and parameter: mean ± std over seeds for each engine, and which is better
    comparison_mean_std.md    the same as Markdown tables
    consistency.csv           (with --repeat K) decision consistency and trust std over K repeats of the first seed
    attack_kind_comparison.csv/.md  per attack kind: detection, ticks to detect, malicious requests allowed,
                              honest FPR, inferred-kind accuracy and the behaviours that triggered detection (mean ± std)
    attack_kind_<metric>.png  one grouped bar chart per metric
Every mode also writes alarm_stats.csv (per scenario × seed × sensor: observed vs configured TP/FP rates, precision).
    <scenario>_seed<first>/   the full single-run comparison export (CSVs, report, charts) for the first seed
LLM settings come from --provider/--model/--endpoint or the env vars OLLAMA_MODEL, OLLAMA_ENDPOINT, LLM_API_KEY.
Gemini: --provider gemini --model <id> --rpm 10, key from GEMINI_API_KEY (or .env).

Single-world outputs:
Writes to results/ (or --out):
    runs_devices.csv        one row per (scenario, seed, device) with truth, detection and timing
    runs_summary.csv        one row per (scenario, seed) with rates
    summary_by_scenario.csv aggregated over seeds
    summary.md              the same table in Markdown
    detection_time_boxplot.png, fpr_bar.png, trust_curves_examples.png
LLM runs are optional and off by default (they are slow); they use Ollama unless configured otherwise.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from collections import Counter
from pathlib import Path
from statistics import mean, median

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from statistics import pstdev  # noqa: E402

from core import live_config as C  # noqa: E402
from core import behaviour_analysis as BA  # noqa: E402
from core.comparison import compare, repeat_consistency, write_exports  # noqa: E402
from core.live_llm import LLMController, make_live_evaluator, resolve_llm_args  # noqa: E402
from core.live_sim import list_scenarios, simulation_from_scenario  # noqa: E402
from core.metrics import compute_metrics  # noqa: E402
from core.twin import TwinSimulation  # noqa: E402


def llm_factory(args):
    """Builds World L's controller: a real LLM endpoint, or the rule-based fake for pipeline checks."""
    if args.fake_llm:
        from fake_llm import RuleBasedFakeLLM
        return lambda: LLMController(RuleBasedFakeLLM(), every_n=args.llm_every, synchronous=True, min_ticks_between=args.min_ticks)
    return lambda: LLMController(make_live_evaluator(args.provider, args.model, args.endpoint, args.api_key,
                                                    args.llm_timeout, max_tokens=args.max_tokens, rpm=args.rpm),
                                 every_n=args.llm_every, timeout_s=args.llm_timeout, min_ticks_between=args.min_ticks)


def make_twin(args, scenario: str, seed: int, mode: str) -> TwinSimulation:
    return TwinSimulation.from_scenario(scenario, seed=seed, mode=mode,
                                        llm_controller_factory=llm_factory(args) if mode != 'bayesian' else None)


def run_one(scenario: str, seed: int, ticks: int | None, args=None, mode: str = 'bayesian'):
    """One single-world run (Bayesian or LLM). Returns (world, metrics)."""
    if mode == 'bayesian':
        live = simulation_from_scenario(scenario, seed=seed)
        live.run(ticks or live.scenario.get('ticks_suggested', 160))
        return live, compute_metrics(live)
    twin = make_twin(args, scenario, seed, 'llm')
    twin.run(ticks or twin.primary.scenario.get('ticks_suggested', 160))
    twin.shutdown()
    return twin.llm, compute_metrics(twin.llm)


def _num(v):
    return v if isinstance(v, (int, float)) and not isinstance(v, bool) else None


def run_comparison(args, scenarios, out: Path) -> int:
    """--mode both: isolated twin runs for every scenario x seed, then mean ± std per parameter."""
    rows, consistency_rows, directions, kind_rows, alarm_rows = [], [], {}, [], []
    for scenario in scenarios:
        for seed in range(args.seeds):
            twin = make_twin(args, scenario, seed, 'both')
            ticks = args.ticks or twin.primary.scenario.get('ticks_suggested', 160)
            twin.run(ticks)
            cons = None
            if args.repeat and seed == 0:
                cons = repeat_consistency(lambda: make_twin(args, scenario, seed, 'both'), args.repeat, ticks)
                for engine, c in cons.items():
                    consistency_rows.append({'scenario': scenario, 'seed': seed, 'engine': engine, 'repeats': c['repeats'],
                                             'decision_consistency': c['decision_consistency'], 'mean_trust_std': c['mean_trust_std']})
            cmp = compare(twin, cons)
            kind_rows += [{'scenario': scenario, 'seed': seed, **r} for r in BA.attack_kind_rows(twin)]
            alarm_rows += BA.alarm_rows_for_csv(twin.bayes, scenario, seed)
            if seed == 0 and not args.no_plots:
                write_exports(cmp, twin, str(out / f'{scenario}_seed{seed}'))
            twin.shutdown()
            for r in cmp['rows']:
                directions[r['parameter']] = r['better'] != '—' or directions.get(r['parameter'], False)
                rows.append({'scenario': scenario, 'seed': seed, 'section': r['section'], 'parameter': r['parameter'],
                             'bayesian': r['bayesian'], 'llm': r['llm'], 'better': r['better']})
        print(f'{scenario}: {args.seeds} seeds compared', flush=True)
    write_csv(out / 'comparison_runs.csv', rows)
    write_csv(out / 'alarm_stats.csv', alarm_rows)
    llm_note = 'rule-based FakeLLM (pipeline check only — NOT a real-LLM result)' if args.fake_llm else args.llm_note
    write_attack_kind_outputs(out, kind_rows, llm_note, args.seeds, plots=not args.no_plots)
    if consistency_rows:
        write_csv(out / 'consistency.csv', consistency_rows)
    summary = []
    for scenario in scenarios:
        params = list(dict.fromkeys(r['parameter'] for r in rows if r['scenario'] == scenario))
        for p in params:
            sel = [r for r in rows if r['scenario'] == scenario and r['parameter'] == p]
            bvals = [x for x in (_num(r['bayesian']) for r in sel) if x is not None]
            lvals = [x for x in (_num(r['llm']) for r in sel) if x is not None]
            wins = Counter(r['better'] for r in sel)
            fmt = lambda xs: '' if not xs else f'{mean(xs):.4g} ± {pstdev(xs):.2g}'  # noqa: E731
            summary.append({'scenario': scenario, 'section': sel[0]['section'], 'parameter': p, 'n_seeds': len(sel),
                            'bayesian_mean': round(mean(bvals), 6) if bvals else '', 'bayesian_std': round(pstdev(bvals), 6) if bvals else '',
                            'llm_mean': round(mean(lvals), 6) if lvals else '', 'llm_std': round(pstdev(lvals), 6) if lvals else '',
                            'bayesian (mean ± std)': fmt(bvals), 'llm (mean ± std)': fmt(lvals),
                            'seeds won B / L / tie': f"{wins.get('Bayesian', 0)} / {wins.get('LLM', 0)} / {wins.get('tie', 0)}"
                            if directions.get(p) else '—'})
    write_csv(out / 'comparison_mean_std.csv', summary)
    note = ' · LLM = rule-based FAKE (pipeline check only, not a result)' if args.fake_llm else f' · LLM = {args.llm_note}'
    md = [f'# Bayesian vs LLM — isolated comparison over {args.seeds} seed(s){note}', '']
    for scenario in scenarios:
        md += [f'## {scenario}', '', '| section | parameter | Bayesian (mean ± std) | LLM (mean ± std) | seeds won B / L / tie |', '|---|---|---|---|---|']
        md += [f"| {s['section']} | {s['parameter']} | {s['bayesian (mean ± std)'] or '—'} | {s['llm (mean ± std)'] or '—'} | {s['seeds won B / L / tie']} |"
               for s in summary if s['scenario'] == scenario]
        md.append('')
    if consistency_rows:
        md += ['## Consistency (repeats of seed 0)', '', '| scenario | engine | repeats | decision consistency | mean trust std |', '|---|---|---|---|---|']
        md += [f"| {c['scenario']} | {c['engine']} | {c['repeats']} | {c['decision_consistency']:.1%} | {c['mean_trust_std']:.4f} |" for c in consistency_rows]
    (out / 'comparison_mean_std.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    print('wrote:', ', '.join(p.name for p in sorted(out.iterdir())))
    return 0


def write_attack_kind_outputs(out: Path, kind_rows: list, llm_note: str, seeds: int, plots: bool = True) -> list:
    """attack_kind_comparison.csv/.md (+ one grouped bar chart per metric) from per-attacker rows of --mode both."""
    agg = BA.aggregate_attack_kinds(kind_rows)
    if not agg:
        return []
    write_csv(out / 'attack_kind_comparison.csv', [{'llm': llm_note, **a} for a in agg])
    pm = lambda m, sd, pct=False: '—' if m is None else (f'{m:.0%} ± {sd:.0%}' if pct else f'{m:.2f} ± {sd:.2f}')  # noqa: E731
    v = lambda x: '—' if x is None else (f'{x:g}' if isinstance(x, (int, float)) else str(x))  # noqa: E731
    md = [f'# Bayesian vs LLM by attack kind ({seeds} seed(s) per scenario)', '', f'**LLM used: {llm_note}.**', '',
          'Attack kinds are the true agent types (used only for scoring). "Recognised from behaviour" = how often the '
          'observed-behaviour signature (core/behaviour_analysis.py) named the right kind. Mean ± std over attacker-runs.', '',
          '| attack kind | runs | recognised from behaviour | detection B / L | median (max) ticks to BLOCK, B / L | '
          'malicious requests allowed, B / L | honest FPR, B / L |', '|---|---|---|---|---|---|---|']
    for a in agg:
        md.append(f"| {a['attack_kind']} | {a['attacker_runs']} | {pm(a['inferred_kind_accuracy'], a['inferred_kind_accuracy_std'], True)} | "
                  f"{pm(a.get('bayesian_detection_rate'), a.get('bayesian_detection_rate_std'), True)} / {pm(a.get('llm_detection_rate'), a.get('llm_detection_rate_std'), True)} | "
                  f"{v(a.get('bayesian_median_ticks_to_block'))} ({v(a.get('bayesian_max_ticks_to_block'))}) / {v(a.get('llm_median_ticks_to_block'))} ({v(a.get('llm_max_ticks_to_block'))}) | "
                  f"{pm(a.get('bayesian_malicious_allowed_mean'), a.get('bayesian_malicious_allowed_std'))} / {pm(a.get('llm_malicious_allowed_mean'), a.get('llm_malicious_allowed_std'))} | "
                  f"{pm(a.get('bayesian_honest_fpr_mean'), a.get('bayesian_honest_fpr_std'), True)} / {pm(a.get('llm_honest_fpr_mean'), a.get('llm_honest_fpr_std'), True)} |")
    md += ['', '## Behaviours that triggered detection', '', '| attack kind | signature (observed facts) | Bayesian reacted to | LLM reacted to |', '|---|---|---|---|']
    md += [f"| {a['attack_kind']} | {a['behaviour signature']} | {a.get('bayesian_top_triggers', '—')} | {a.get('llm_top_triggers', '—')} |" for a in agg]
    (out / 'attack_kind_comparison.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    files = [out / 'attack_kind_comparison.csv', out / 'attack_kind_comparison.md']
    if not plots:
        return files
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    metrics = [('detection_rate', 'detection rate', '_std'), ('median_ticks_to_block', 'median ticks to BLOCK', None),
               ('malicious_allowed_mean', 'malicious requests allowed (mean)', '_std'), ('honest_fpr_mean', 'honest false-positive rate (mean)', '_std')]
    kinds = [a['attack_kind'] for a in agg]
    for key, title, sd in metrics:
        fig, ax = plt.subplots(figsize=(8, 4))
        for i, (engine, colour) in enumerate((('bayesian', '#2563eb'), ('llm', '#9333ea'))):
            vals = [a.get(f'{engine}_{key}') for a in agg]
            errs = [a.get(f"{engine}_{key.replace('_mean', '')}{sd}") if sd else None for a in agg]
            ax.bar([x + (i - 0.5) * 0.4 for x in range(len(kinds))], [x if x is not None else float('nan') for x in vals], 0.4,
                   yerr=[e or 0 for e in errs] if sd else None, capsize=3, color=colour, label='Bayesian' if engine == 'bayesian' else 'LLM')
        ax.set_xticks(range(len(kinds))); ax.set_xticklabels(kinds, rotation=15)
        ax.set_title(f'{title} by attack kind'); ax.legend()
        ax.text(0.99, 0.98, f'LLM: {llm_note}', transform=ax.transAxes, ha='right', va='top', fontsize=7, color='#6b21a8')
        fig.tight_layout(); p = out / f'attack_kind_{key}.png'; fig.savefig(p, dpi=130); plt.close(fig); files.append(p)
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(range(len(kinds)), [a['inferred_kind_accuracy'] for a in agg], 0.5, color='#0f766e')
    ax.set_xticks(range(len(kinds))); ax.set_xticklabels(kinds, rotation=15); ax.set_ylim(0, 1.05)
    ax.set_title('Attack kind recognised from observed behaviour (accuracy)')
    fig.tight_layout(); p = out / 'attack_kind_inferred_accuracy.png'; fig.savefig(p, dpi=130); plt.close(fig); files.append(p)
    return files


def write_csv(path: Path, rows: list) -> None:
    if not rows:
        return
    with path.open('w', newline='', encoding='utf-8') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(run_rows: list, device_rows: list) -> list:
    out = []
    for scenario in sorted({r['scenario'] for r in run_rows}):
        runs = [r for r in run_rows if r['scenario'] == scenario]
        atk = [d for d in device_rows if d['scenario'] == scenario and d['role'] == 'attacker']
        hon = [d for d in device_rows if d['scenario'] == scenario and d['role'] == 'honest']
        blocks = [d['ttd_block'] for d in atk if d['ttd_block'] != '']
        out.append({
            'scenario': scenario, 'seeds': len(runs), 'attacker_runs': len(atk),
            'detection_rate': round(mean(d['detected'] for d in atk), 3) if atk else '',
            'median_ticks_to_block': median(blocks) if blocks else '',
            'max_ticks_to_block': max(blocks) if blocks else '',
            'honest_quarantine_rate': round(mean(d['quarantined'] for d in hon), 3) if hon else '',
            'honest_step_up_rate': round(mean(d['any_step_up'] for d in hon), 3) if hon else '',
            'honest_hard_gate_blocks_per_run': round(sum(d['hard_gate_blocks'] for d in hon) / len(runs), 2) if runs else '',
            'mean_f1': round(mean(r['f1'] for r in runs if r['f1'] != ''), 3) if any(r['f1'] != '' for r in runs) else '',
        })
    return out


def make_plots(out: Path, device_rows: list, summary: list, examples: dict) -> list:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    files = []
    # 1) detection time per attack type
    types = sorted({d['truth'] for d in device_rows if d['role'] == 'attacker'})
    if types:
        data = [[d['ttd_block'] for d in device_rows if d['truth'] == t and d['ttd_block'] != ''] or [float('nan')] for t in types]
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.boxplot(data, showmeans=True)
        ax.set_xticks(range(1, len(types) + 1))
        ax.set_xticklabels(types)  # works on old and new matplotlib (the boxplot `labels` argument was renamed)
        ax.set_ylabel('ticks from attack start to quarantine')
        ax.set_title('Bayesian time-to-detect per attack type')
        for i, t in enumerate(types, 1):
            n_all = sum(1 for d in device_rows if d['truth'] == t)
            n_det = sum(1 for d in device_rows if d['truth'] == t and d['detected'])
            ax.annotate(f'{n_det}/{n_all} caught', (i, ax.get_ylim()[1]), ha='center', va='top', fontsize=8)
        plt.xticks(rotation=15)
        fig.tight_layout(); p = out / 'detection_time_boxplot.png'; fig.savefig(p, dpi=150); plt.close(fig); files.append(p)
    # 2) false positives per scenario
    fig, ax = plt.subplots(figsize=(8, 4.5))
    names = [s['scenario'] for s in summary]
    q = [s['honest_quarantine_rate'] or 0 for s in summary]
    su = [s['honest_step_up_rate'] or 0 for s in summary]
    x = range(len(names))
    ax.bar([i - 0.2 for i in x], su, width=0.4, label='any STEP-UP', color='#f59e0b')
    ax.bar([i + 0.2 for i in x], q, width=0.4, label='quarantined (false block)', color='#ef4444')
    ax.axhline(0.05, ls='--', color='grey', lw=1, label='5% target')
    ax.set_xticks(list(x)); ax.set_xticklabels(names, rotation=20, ha='right')
    ax.set_ylabel('fraction of honest device-runs'); ax.set_title('False positives on honest / noisy-honest devices')
    ax.legend(); fig.tight_layout(); p = out / 'fpr_bar.png'; fig.savefig(p, dpi=150); plt.close(fig); files.append(p)
    # 3) example trust curves (seed 0 of each scenario)
    n = len(examples)
    if n:
        cols = 2
        rows = (n + 1) // 2
        fig, axes = plt.subplots(rows, cols, figsize=(12, 3.2 * rows), squeeze=False)
        for ax, (scenario, live) in zip(axes.flat, sorted(examples.items())):
            ticks = [h['tick'] for h in live.trust_history]
            for d in sorted(live.devices):
                attacker = live.ground_truth[d]['attack_start'] is not None
                ax.plot(ticks, [h[d] for h in live.trust_history], lw=2.2 if attacker else 0.9,
                        color=None if attacker else '#94a3b8', label=f"{d} ({live.ground_truth[d]['agent']})" if attacker else None)
                if attacker:
                    ax.axvline(live.ground_truth[d]['attack_start'], ls=':', color='#a855f7')
            ax.axhline(live.controller.allow_threshold, ls='--', color='#22c55e', lw=0.8)
            ax.axhline(live.controller.block_threshold, ls='--', color='#ef4444', lw=0.8)
            ax.set_title(f'{scenario} (seed {live.seed}) — grey = honest'); ax.set_ylim(0, 1)
            if ax.get_legend_handles_labels()[0]:
                ax.legend(fontsize=7, loc='lower left')
        for ax in list(axes.flat)[n:]:
            ax.axis('off')
        fig.tight_layout(); p = out / 'trust_curves_examples.png'; fig.savefig(p, dpi=150); plt.close(fig); files.append(p)
    return files


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--scenario', default='all', help='preset name or "all"')
    ap.add_argument('--seeds', type=int, default=30)
    ap.add_argument('--ticks', type=int, default=None, help='default: the scenario file\'s ticks_suggested')
    ap.add_argument('--mode', choices=['bayesian', 'llm', 'both'], default=None, help='which world(s) to run (default bayesian)')
    ap.add_argument('--llm', dest='llm', action='store_true', help='shortcut for --mode both')
    ap.add_argument('--no-llm', dest='llm', action='store_false', help='shortcut for --mode bayesian (default)')
    ap.add_argument('--repeat', type=int, default=0, help='with --mode both: repeat the first seed K times for consistency')
    ap.add_argument('--fake-llm', action='store_true', help='use the rule-based fake LLM (pipeline check only)')
    ap.add_argument('--provider', default=os.getenv('LLM_PROVIDER', 'ollama'), choices=['ollama', 'gemini', 'openai_compatible'],
                    help='gemini reads the key from GEMINI_API_KEY (or .env)')
    ap.add_argument('--model', default=None, help='default: OLLAMA_MODEL or llama3.2:3b; for gemini GEMINI_MODEL or a Flash-Lite model')
    ap.add_argument('--endpoint', default=None, help='default: OLLAMA_ENDPOINT or http://localhost:11434/api/chat (unused for gemini)')
    ap.add_argument('--rpm', type=int, default=C.GEMINI_MAX_RPM, help='gemini: client-side requests-per-minute limit')
    ap.add_argument('--llm-timeout', type=int, default=C.LLM_TIMEOUT_S)
    ap.add_argument('--llm-every', type=int, default=C.LLM_EVERY_N)
    ap.add_argument('--min-ticks', type=int, default=C.LLM_MIN_TICKS_BETWEEN_EVALS, help='min ticks between non-hard evaluations of a device')
    ap.add_argument('--max-tokens', type=int, default=C.LLM_MAX_OUTPUT_TOKENS, help='num_predict / max_tokens for the LLM reply')
    ap.add_argument('--out', default=str(ROOT / 'results'))
    ap.add_argument('--no-plots', action='store_true')
    ap.add_argument('--sensor-weight', type=float, default=None,
                    help='override SENSOR_WEIGHT (e.g. 0 = behaviour-only ablation: ignore IDS/SIEM/BM entirely)')
    ap.set_defaults(llm=False)
    args = resolve_llm_args(ap.parse_args(argv))
    if args.sensor_weight is not None:
        C.SENSOR_WEIGHT = args.sensor_weight
    mode = args.mode or ('both' if args.llm else 'bayesian')
    scenarios = list_scenarios() if args.scenario == 'all' else [args.scenario]
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    if mode == 'both':
        return run_comparison(args, scenarios, out)
    run_rows, device_rows, examples, alarm_rows = [], [], {}, []
    for scenario in scenarios:
        for seed in range(args.seeds):
            live, m = run_one(scenario, seed, args.ticks, args, mode)
            alarm_rows += BA.alarm_rows_for_csv(live, scenario, seed)
            if seed == 0:
                examples[scenario] = live
            run_rows.append({'scenario': scenario, 'seed': seed, 'ticks': m['ticks'],
                             'detection_rate': m['detection_rate'] if m['detection_rate'] is not None else '',
                             'fpr_block': m['fpr_block'], 'fpr_step_up': m['fpr_step_up'],
                             'precision': m['precision'] if m['precision'] is not None else '',
                             'recall': m['recall'] if m['recall'] is not None else '',
                             'f1': m['f1'] if m['f1'] is not None else ''})
            for a in m['attackers']:
                device_rows.append({'scenario': scenario, 'seed': seed, 'device': a['device'], 'role': 'attacker', 'truth': a['truth'],
                                    'detected': int(a['detected']), 'ttd_step_up': '' if a['ttd_step_up'] is None else a['ttd_step_up'],
                                    'ttd_block': '' if a['ttd_block'] is None else a['ttd_block'],
                                    'quarantined': int(a['detected']), 'any_step_up': '', 'hard_gate_blocks': ''})
            for h in m['honest']:
                device_rows.append({'scenario': scenario, 'seed': seed, 'device': h['device'], 'role': 'honest', 'truth': h['truth'],
                                    'detected': '', 'ttd_step_up': '', 'ttd_block': '',
                                    'quarantined': int(h['quarantined']), 'any_step_up': int(h['any_step_up']), 'hard_gate_blocks': h['hard_gate_blocks']})
        print(f'{scenario}: {args.seeds} seeds done', flush=True)
    summary = aggregate(run_rows, device_rows)
    write_csv(out / 'runs_summary.csv', run_rows)
    write_csv(out / 'runs_devices.csv', device_rows)
    write_csv(out / 'summary_by_scenario.csv', summary)
    write_csv(out / 'alarm_stats.csv', alarm_rows)
    header = list(summary[0]) if summary else []
    md = ['# Batch results', '', f"Seeds per scenario: {args.seeds} · world: {mode} · "
          f"sensor weight {C.SENSOR_WEIGHT} · event weight {C.EVENT_WEIGHT}", '',
          '| ' + ' | '.join(header) + ' |', '|' + '---|' * len(header)]
    md += ['| ' + ' | '.join(str(s[h]) for h in header) + ' |' for s in summary]
    (out / 'summary.md').write_text('\n'.join(md) + '\n', encoding='utf-8')
    files = [] if args.no_plots else make_plots(out, device_rows, summary, examples)
    print('\n'.join(md))
    print('wrote:', ', '.join(str(p.name) for p in [out / 'runs_summary.csv', out / 'runs_devices.csv', out / 'summary_by_scenario.csv', out / 'summary.md'] + files))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
