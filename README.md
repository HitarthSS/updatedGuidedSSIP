# My harness for Guided SSIP / STEG — goal settling

Experimental, frozen-policy harness code for the [paper](https://arxiv.org/abs/2605.10051). The evaluated behavior is in [`accuracy_harness.py`](accuracy_harness.py). Original authorship and source are credited in [NOTICE.md](NOTICE.md).

## Results

| Reference | Result |

| --- | --- |

| Paper / official release | Oscillate STEG: 56.0% |

| Our run of the authors' harness | Released seeds: 19/50 (38.0%); fresh extension: 15/50 (30.0%) |

| **My harness** | **Released seeds: 25/50 (50.0%); fresh extension: 19/50 (38.0%)** |


Oscillate only. Seeds 500–549 are repeated released evaluation; seeds 550–599 form a separate predeclared robustness extension. Do not pool the cohorts.


These are reduced or explicitly qualified experiments. They do not establish full-paper reproduction or a confirmed benchmark win. See [evaluation details](EVALUATION.md) and the compact records in [results/](results/).

## What my harness changes

1. Remove the final ensemble transition whose output never contributes to the guidance objective, retaining all costs and random draws.

2. For the selected settle_only variant, hold the preceding agent target only while observed block position/angle lie within the fixed visible goal window.

3. Resume native actions outside that window. No reward/oracle input, early termination or extra action allowance.


Categories: `sampler_guidance`, `execution_monitoring`, `uncertainty_compute`.

## Models and inputs

- Released pusht_state_1000ep_si.ckpt, SHA256 e3b722c9a87425944a45dae7bd211ef4721b2bb92e50d93244ae795642257127.

- Training-free STEG on a frozen low-dimensional pushing policy; this is not a VLA or an LLM planner.


Policy weights remain frozen. Model checkpoints and benchmark assets must be obtained from their original releases under their own access terms. No paid LLM/VLM API is required by the published variant.

## Start here

1. Use a Linux GPU host for simulator/model evaluation. Read the [pinned upstream setup](https://github.com/0scarJ1ang/guided-streaming-policy) before installing its model stack.
2. Controller dependency: Use the pinned upstream PyTorch/Push-T environment.
3. Inspect `configs/method.json` and `provenance/upstream.json`.
4. Verify the source-only export and inspect the core API:

```sh
python3 tools/verify_export.py
python -m compileall -q accuracy_harness.py speed_harness.py
```


CPU checks validate code behavior or packaging only. They do not run the robot benchmark or demonstrate numerical parity on a new GPU/runtime.

## Integrate with the frozen policy

Read [the integration contract and arm definitions](docs/INTEGRATION.md).

Use build(author_module, original_rollout_function, "settle_only") from accuracy_harness in the pinned upstream Push-T environment. speed_harness.py retains the separately evaluated compute-only transform.


The core harness is reusable. Integration snapshots and historical experiment drivers are included for inspection; full fresh-machine simulator reproduction has not been revalidated during this publication. Prepare the pinned native environment and required assets before using those drivers. No benchmark is automatically launched by cloning, importing, or verifying this repository.

## Evaluation budget

250 physical actions, 1.0x. The original 200-action obstacle-placement reference is separately recorded benchmark preparation.


Use identical tasks, initial states, seeds, models and budgets for the authors' harness and my harness. Charge every executed recovery/intervention to the action limit. Separate development and final cases, freeze the design before evaluation, and report all completed matched cases.

## Repository map

- `accuracy_harness.py`: main harness behavior.
- `configs/`: method and evaluation metadata.
- `results/`: compact measured outcomes and their interpretation.
- `provenance/`: source/model pins and export hashes.
- `tools/verify_export.py`: offline source integrity check.
- `reference/`, `integration/`, or `experiments/`, when present: native-loop snapshots.

## Known limitations

- Primary success is final reward >0.85 and no collision, following the paper. Released code uses maximum reward instead.

- Under the secondary release criterion, released-seed counts are 26/50 versus 25/50; fresh counts are 26/50 versus 26/50.

- Earlier released-seed aggregate was known; the fresh extension is reported separately.

- Paper seed manifest is unverified and timing is on shared hardware. These results do not establish unseen robot task types.


Shared-GPU wall-clock measurements are descriptive. New checkpoints, a new serving runtime, different seeds or changed horizon enforcement require a newly labeled comparison.

## Publication checks

Source syntax and recorded-core hashes checked. All exported Python parses, shell scripts pass `bash -n`, relative documentation links resolve, and source files pass a credential-pattern scan. See [validation](provenance/validation.json). A complete GPU benchmark was not rerun for this publication.
