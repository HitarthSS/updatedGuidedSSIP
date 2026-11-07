# Expanded Guided SSIP evaluation

Prepare the pinned [upstream checkout](https://github.com/0scarJ1ang/guided-streaming-policy/tree/c68fb13152880645edd134417e7f2d1de62665f0), including its Push-T dependencies, dataset, and released `pusht_state_1000ep_si.ckpt`. Point the run directory at that checkout. The runner verifies the pinned source, checkpoint, and dataset hashes before starting. Activate the upstream Python environment before running the commands.

```sh
export GUIDED_SSIP_RUN_DIR="$PWD/runs/expanded"
mkdir -p "$GUIDED_SSIP_RUN_DIR"
ln -s /absolute/path/to/guided-streaming-policy "$GUIDED_SSIP_RUN_DIR/upstream"
python experiments/expanded.py freeze
CUDA_VISIBLE_DEVICES=0 python -u experiments/expanded.py shard --shard 0
```

From a second terminal with the same environment and run directory, launch shard 1 with `CUDA_VISIBLE_DEVICES=7 python -u experiments/expanded.py shard --shard 1`. Run `python experiments/expanded.py report` to refresh the aggregate. Each shard smoke-tests all three modes before contributing 150 matched pairs. The GPU IDs mirror the measured run.

The authors and v4 arms share a frozen checkpoint, seeded initial states, obstacle starts, and a 250-action limit. Intercept and Oscillate separately record the native obstacle-placement reference rollout, capped at 200 actions. Chase initializes the obstacle directly. Both final-reward and peak-reward success are recorded, alongside collision counts, model input rows and shared-GPU rollout timing.

The [protocol](../results/expanded-protocol.json) records the actual run, and the [report](../results/expanded-report.json) contains all six group summaries. This portable runner uses the existing frozen v4 implementation in `reference/` and exposes a configurable output directory. Its packaging was checked independently of the completed Sudarshan run.
