# Reproducing the Main Experiment

`scripts/run_loop.sh` runs three rounds of AeroRSI from the supervised policy (π<sub>0</sub>) and evaluates each
round on the 3,005 test routes. Every setting is in `configs/aerorsi.yaml`.

```bash
GPUS="0 1 2 3" OUT=outputs/aerorsi bash scripts/run_loop.sh
```

## Stages of one round

Round *k* uses the policy of round *k*−1, π<sub>k−1</sub> (`round0/adapter` for *k* = 1).

| Stage | Command | Output | Time (4 GPUs) |
|:--|:--|:--|:--|
| Fly all training routes | `aerorsi rollout` | per-route results; inputs of the last 17 steps before each failure, and of three normal states of each success | 17–20 h |
| Normal-state inputs (round 1 only) | `aerorsi normal-inputs` | `normal_inputs.jsonl`: three states from each of 840 successful flights | seconds |
| Anchor failures | `aerorsi anchor` | `plan.json`: 2,400 anchors; normal-input split | minutes (CPU) |
| Verify repairs | `aerorsi verify` | every candidate's continuation and the decision at each anchor | about 6 h |
| Pool repairs | `aerorsi pool` | `repairs.jsonl`: accepted anchors with their verified action | seconds |
| Relabel normal states | `aerorsi relabel` | π<sub>k−1</sub>'s greedy actions on the normal inputs | minutes (1 GPU) |
| Assemble | `aerorsi assemble` | `train_rows.jsonl`: weighted rows in training order | seconds |
| Train | `aerorsi train` | `final/`: the adapter of π<sub>k</sub> | 11–23 min (1 GPU) |
| Evaluate | `aerorsi evaluate` | per-route results and metrics on the test splits | about 11 h |

A round costs about 100 GPU hours. Each stage writes `config.json` (the effective settings) and `metrics.json`
(its summary) into its output directory, and refuses to overwrite an existing one.

## What one round produces

In the paper's first round (from π<sub>0</sub>), 2,400 anchors gave 1,048 verified repairs. After holding out 20% of
the repair trajectory groups, π<sub>1</sub> was trained on 837 repairs and 1,990 normal states (2,827 rows, 2 epochs,
global batch 16). From round 2 on, the previous round's training repairs are replayed with 30% of the repair weight.

## Expected results

SR on each test split, and SR, SPL and CR over all 3,005 test routes:

| Round | Seen | Unseen Object | Unseen Map | SR | SPL | CR |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 (SFT) | 48.38 | 56.44 | 36.85 | 46.39 | 37.37 | 38.64 |
| 1 | 62.62 | 67.89 | 43.42 | 57.60 | 46.96 | 29.48 |
| 2 | 73.91 | 79.01 | 45.82 | 66.02 | 51.96 | 21.26 |
| 3 | 75.32 | 82.19 | 46.24 | 67.49 | 51.00 | 19.10 |

Starting from AeroDPO (`INIT=checkpoints/AeroDPO-2B-384-LoRA`, DPO init.):

| Round | Seen | Unseen Object | Unseen Map | SR | SPL | CR |
|:-:|:-:|:-:|:-:|:-:|:-:|:-:|
| 0 (AeroDPO) | 60.01 | 66.45 | 45.93 | 56.87 | 45.20 | 28.95 |
| 1 | 69.96 | 73.93 | 46.76 | 63.39 | 49.84 | 22.46 |
| 2 | 72.50 | 78.54 | 48.85 | 66.22 | 50.07 | 19.07 |
| 3 | 76.52 | 79.65 | 53.55 | 69.85 | 53.52 | 17.90 |

Closed-loop flight is stochastic (the simulator's physics is not bit-deterministic), so a rerun produces different
anchors and repairs and its numbers vary around these values: evaluating the same policy twice changes the outcome
of about a quarter of the routes and the overall SR by up to about two points.

## Notes

* **Normal states.** Retention keeps the current policy's own actions at the normal-state inputs, which the release
  takes from the round-0 policy's successful rollouts: before the first moving action, about 30 m from the target,
  and before the final stop, for 840 flights. In the paper's SFT-init. run these states came from AeroDPO's
  successful flights on the same routes; taking them from the policy's own flights instead gave a first-round SR of
  58.97 against 57.60 (a paper ablation), a difference within the run-to-run noise. The DPO-init. run takes them
  from AeroDPO, exactly as in the paper.
* **Protocol.** Evaluation uses the benchmark's native action controller, its termination rules and 200 actions per
  episode. Collection pauses the simulator after every action so that states can be saved and restored; this only
  affects training data, never evaluation.
* **Interrupted routes.** A route whose simulator fails has no result and is never counted as a failure. Fly the
  missing routes again with `--skip-done <previous runs>` and pass all runs to `aerorsi metrics`.
* **Smoke test.** Every navigation stage accepts `--limit N`, e.g.
  `aerorsi evaluate --adapter checkpoints/AeroRSI-2B-SFTinit --output outputs/smoke --gpus 0 --workers-per-gpu 1 --limit 2`.
