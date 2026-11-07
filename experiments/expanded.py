"""Matched expansion of frozen Guided SSIP / STEG v4 across dynamic Push-T modes."""
import argparse
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import sys
import time

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("GUIDED_SSIP_RUN_DIR", str(REPO / "runs" / "expanded"))).resolve()
V4 = REPO / "reference"
sys.path.insert(0, str(V4))
import accuracy_campaign as frozen_runner

MODES = {"intercept": 2, "oscillate": 1, "chase": 3}
PAPER_PERCENT = {"intercept": 68.0, "oscillate": 56.0, "chase": 36.0}
COHORTS = {"release_range": range(500, 550), "fresh": range(600, 650)}
GPUS = (0, 7)
CANDIDATE = "settle_only"
frozen_runner.ROOT = ROOT


def load(path):
    return json.loads(path.read_text())


def save(name, data):
    frozen_runner.write(ROOT / name, data)


def freeze():
    published = load(REPO / "results" / "expanded-protocol.json")
    if published["variant"] != CANDIDATE:
        raise RuntimeError("Expected frozen v4 settle_only")
    for name, expected in published["source_hashes"].items():
        if name == "ACCURACY_PLAN.md":
            continue  # Historical development notes are outside the runtime.
        path = ROOT / name if name.startswith("upstream/") else V4 / name
        if frozen_runner.digest(path) != expected:
            raise RuntimeError(f"Changed frozen input: {name}")
    content = dict(
        modes=MODES, paper_table_ii_percent=PAPER_PERCENT,
        cohorts={k: list(v) for k, v in COHORTS.items()},
        cohort_note="Released code hard-codes Oscillate on seeds 500–549. Intercept and Chase use "
                    "the same release-range seeds with native mode switches; paper seed manifest unverified.",
        variant=CANDIDATE, source_hashes=published["source_hashes"],
        expansion_sha256=frozen_runner.digest(Path(__file__)),
        episode_horizon=250, reference_horizon=200,
        chase_reference="None in native release",
        primary_success="final reward > 0.85 and no collision",
        secondary_success="release metric: max reward > 0.85 and no collision",
        start_obstacle_position=[0.5, 0.5], guidance_scale=7.0,
        gpu_by_seed_parity=list(GPUS), paired_cases=300, physical_rollouts=600, paid_api_calls=0,
    )
    path = ROOT / "FROZEN.json"
    if path.exists():
        if load(path) != content:
            raise RuntimeError("Expanded evaluation freeze changed")
    else:
        save("FROZEN.json", content)


class DynamicRunner(frozen_runner.Runner):
    def __init__(self):
        super().__init__()
        self.mode = None
        self.obstacle_start = None
        self.freeze_digest = frozen_runner.digest(ROOT / "FROZEN.json")
        base = self.m.DynamicObstacleManager
        runner = self

        class ObservedObstacle(base):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                runner.obstacle_start = [float(x) for x in self.start_pos]

        self.m.DynamicObstacleManager = ObservedObstacle
        for name, original in self.functions.items():
            self.functions[name] = self.adapt(original)

    def adapt(self, original):
        def execute(*, obstacle_mode, **kwargs):
            imgs, metrics = original(obstacle_mode=self.mode, **kwargs)
            # Chase initializes its obstacle directly, without a clean trajectory.
            if self.mode == MODES["chase"]:
                if "reference" in self.record:
                    raise RuntimeError("Chase unexpectedly ran an obstacle setup trajectory")
                self.record["reference"] = dict(steps=0, wall_seconds=0.0, trajectory_sha256=None)
            return imgs, metrics
        return execute

    def run(self, split, variant, seed):
        result_path = ROOT / "runs-accuracy" / split / variant / f"{seed}.json"
        if result_path.exists():
            row = load(result_path)
            if row.get("freeze_sha256") == self.freeze_digest and row.get("mode") == self.mode:
                return row
            raise RuntimeError(f"Incomplete or incompatible saved episode: {result_path}")
        self.obstacle_start = None
        row = super().run(split, variant, seed)
        if self.obstacle_start is None:
            raise RuntimeError("Obstacle initialization was not captured")
        row.update(mode=self.mode, obstacle_start=self.obstacle_start, freeze_sha256=self.freeze_digest,
                   physical_gpu=int(os.environ["CUDA_VISIBLE_DEVICES"]))
        frozen_runner.write(result_path, row)
        return row

    def paired(self, cohort, mode, seed):
        self.mode = MODES[mode]
        pair = self.pair(f"{cohort}-{mode}", CANDIDATE, seed, seed + self.mode)
        left, right = pair["authors"], pair["my_harness"]
        if left["obstacle_start"] != right["obstacle_start"] or left["physical_gpu"] != right["physical_gpu"]:
            raise RuntimeError("Different paired obstacle starts or GPUs")
        for row in (left, right):
            if row["steps"] > 250 or row["horizon_violations"]:
                raise RuntimeError("Invalid episode horizon")
        return pair


def summarize_pairs():
    groups = {}
    for cohort, seeds in COHORTS.items():
        for mode in MODES:
            rows = []
            for seed in seeds:
                path = ROOT / "pairs" / cohort / mode / f"{seed}.json"
                if path.exists():
                    rows.append(load(path))
            value = frozen_runner.summary(rows)
            a, b = value["my_only_wins"], value["authors_only_wins"]
            discordant = a + b
            value["mcnemar_exact_p"] = min(
                1.0, 2 * sum(math.comb(discordant, j) for j in range(min(a, b) + 1)) / 2**discordant
            )
            value["planned_pairs"] = len(seeds)
            value["paper_steg_percent"] = PAPER_PERCENT[mode] if cohort == "release_range" else None
            groups[f"{cohort}/{mode}"] = value
    return dict(completed_pairs=sum(g["pairs"] for g in groups.values()), planned_pairs=300, groups=groups)


def run_shard(shard):
    freeze()
    gpu = GPUS[shard]
    if os.environ.get("CUDA_VISIBLE_DEVICES") != str(gpu):
        raise RuntimeError(f"Shard {shard} requires GPU {gpu}")
    with (ROOT / f"shard-{shard}.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        def status(stage, **details):
            data = dict(stage=stage, shard=shard, gpu=gpu, time=time.time(), **details)
            save(f"status-{shard}.json", data)
            print(json.dumps(data), flush=True)
        runner = None
        try:
            status("loading")
            runner = DynamicRunner()
            status("smoke")
            for mode in MODES:
                runner.paired("smoke", mode, 900701 + shard)
            status("evaluating", matched_smoke_pairs=3)
            for i in range(shard, 50, 2):
                for cohort, seeds in COHORTS.items():
                    for mode in MODES:
                        seed = seeds[i]
                        output = ROOT / "pairs" / cohort / mode / f"{seed}.json"
                        if output.exists():
                            continue
                        status("evaluating", cohort=cohort, mode=mode, seed=seed)
                        pair = runner.paired(cohort, mode, seed)
                        compact = {}
                        for arm, row in pair.items():
                            compact[arm] = {
                                k: row[k] for k in (
                                    "paper_success", "release_success", "collision", "final_reward",
                                    "episode_wall_seconds", "model_calls", "horizon_violations", "steps",
                                    "obstacle_start", "physical_gpu",
                                )
                            }
                            compact[arm]["episode"] = {
                                k: row["episode"][k] for k in ("simulator_seconds", "initial_state_sha256")
                            }
                            compact[arm]["reference"] = row["reference"]
                        frozen_runner.write(output, compact)
            status("complete")
        except BaseException as exc:
            status("failed", error=str(exc))
            raise
        finally:
            if runner is not None:
                runner.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("freeze", "shard", "report"))
    parser.add_argument("--shard", type=int, choices=(0, 1))
    args = parser.parse_args()
    if args.action == "freeze":
        freeze()
    elif args.action == "shard":
        run_shard(args.shard)
    else:
        report = summarize_pairs()
        save("REPORT.json", report)
        print(json.dumps(report, indent=2))
