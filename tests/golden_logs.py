"""Fingerprints of complete world logs, used to prove that display-only work changed no decision.

A fingerprint is the SHA-256 of a canonical JSON dump of every record in a world (Bayesian world and
FakeLLM world), of the rendered terminal lines, of the log CSV and of the per-tick trust history.
Keys that were added later purely for display (the exact-precision `calc` block, the LLM's raw reply
and model name) are removed first, and the wall-clock `latency_ms` of LLM calls is dropped.
Regenerate the baseline only on purpose:  python tests/golden_logs.py --write
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'experiments'))

GOLDEN = Path(__file__).resolve().parent / 'golden' / 'log_fingerprints.json'
CASES = [('compromised_valid_token', 0), ('compromised_valid_token', 42), ('multi_attacker', 0),
         ('multi_attacker', 42), ('replay_attack', 0), ('replay_attack', 42)]
DISPLAY_ONLY_KEYS = {'calc', 'raw_reply', 'llm_model'}   # added for the calculation view; not decisions
_MS = re.compile(r'\] \d+ ms:')


def _strip(obj):
    if isinstance(obj, dict):
        return {k: _strip(v) for k, v in obj.items() if k not in DISPLAY_ONLY_KEYS and k != 'latency_ms'}
    if isinstance(obj, (list, tuple)):
        return [_strip(v) for v in obj]
    return obj


def _h(text: str) -> str:
    return hashlib.sha256(text.encode('utf-8')).hexdigest()


# The baseline was recorded with these LLM-world settings (the defaults before the speed changes).
LEGACY_LLM = dict(every_n=5, min_ticks_between=0, recent_requests=10, compact_context=False)


def fingerprint(scenario: str, seed: int, llm_settings: dict = None) -> dict:
    """llm_settings=None reproduces the baseline's LLM settings (LEGACY_LLM); pass {} for the current defaults."""
    from core.live_llm import LLMController
    from core.live_sim import render_record
    from core.metrics import log_rows, to_csv
    from core.twin import TwinSimulation
    from fake_llm import RuleBasedFakeLLM
    twin = TwinSimulation.from_scenario(scenario, seed=seed, mode='both',
                                        llm_controller_factory=lambda: LLMController(RuleBasedFakeLLM(), synchronous=True,
                                                                                     **(LEGACY_LLM if llm_settings is None else llm_settings)))
    twin.run(twin.primary.scenario.get('ticks_suggested', 160))
    out = {}
    for engine, world in twin.worlds.items():
        lines = [_MS.sub('] ms:', line) for r in world.records for line in render_record(r)]
        rows = [{k: (_MS.sub('] ms:', v) if isinstance(v, str) else v) for k, v in row.items()} for row in log_rows(world)]
        out[engine] = {'records': _h(json.dumps(_strip(world.records), sort_keys=True, default=str)),
                       'lines': _h('\n'.join(lines)), 'log_csv': _h(to_csv(rows)),
                       'trust_history': _h(json.dumps(world.trust_history, sort_keys=True)),
                       'n_records': len(world.records)}
    twin.shutdown()
    return out


def all_fingerprints(llm_settings: dict = None) -> dict:
    return {f'{s}@{seed}': fingerprint(s, seed, llm_settings) for s, seed in CASES}


if __name__ == '__main__':
    if '--write' in sys.argv:
        GOLDEN.parent.mkdir(exist_ok=True)
        GOLDEN.write_text(json.dumps(all_fingerprints(), indent=2) + '\n', encoding='utf-8')
        print('wrote', GOLDEN)
    else:
        print(json.dumps(all_fingerprints(), indent=2))
