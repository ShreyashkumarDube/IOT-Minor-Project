# Batch results

Seeds per scenario: 30 · world: bayesian · sensor weight 0.15 · event weight 0.5

| scenario | seeds | attacker_runs | detection_rate | median_ticks_to_block | max_ticks_to_block | honest_quarantine_rate | honest_step_up_rate | honest_hard_gate_blocks_per_run | mean_f1 |
|---|---|---|---|---|---|---|---|---|---|
| compromised_valid_token | 30 | 30 | 1 | 0.0 | 1 | 0.008 | 0.029 | 7.13 | 0.978 |
| multi_attacker | 30 | 90 | 1 | 0.0 | 1 | 0.006 | 0.017 | 3.6 | 0.995 |
| normal_day | 30 | 0 |  |  |  | 0.015 | 0.033 | 9.43 |  |
| replay_attack | 30 | 30 | 1 | 0.0 | 1 | 0.013 | 0.033 | 9.43 | 0.967 |
| scanning_attack | 30 | 30 | 1 | 0.0 | 0 | 0.013 | 0.025 | 6.93 | 0.967 |
| stealth_slow_and_low | 30 | 30 | 1 | 8.5 | 64 | 0.017 | 0.058 | 15.47 | 0.956 |
| token_flood_attack | 30 | 30 | 1 | 0.0 | 0 | 0.013 | 0.025 | 7.47 | 0.967 |
