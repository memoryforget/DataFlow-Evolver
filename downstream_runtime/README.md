# Downstream experiment runtime

This tracked directory contains the reproducible launchers, exact formal experiment
record, task definition, result summarizer, final-best SFT/full-benchmark runner, and
isolated comparison implementations.

## Layout

- `math_mix_v3_periodic_seed42_formal_v1.md`: frozen formal experiment runbook.
- `dataflow_math_periodic.local.yaml`: exact DataFlow-Evolver protocol configuration.
- `run_math_periodic*.sh`: formal run launch and recording wrappers.
- `run_final_best_full_benchmark.py`: final incumbent SFT plus eight full benchmarks.
- `start_embedding_server.sh`: dedicated GPU-7 embedding service launcher.
- `comparisons/<experiment>/`: one self-contained directory per comparison protocol.

Generated runs belong under `workspace/runs/` and `artifacts/experiment_runs/`; both
remain ignored. Model weights, source corpora, caches, credentials and large benchmark
data are also intentionally excluded from Git.

See `ENVIRONMENT_INVENTORY.md` for exact source-server versions and migration inputs.
