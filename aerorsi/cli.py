"""Command line of every stage. Run ``aerorsi <command> --help`` for the options of one stage.

A round k of the loop, with the policy of round k-1 in ``A``::

    aerorsi rollout  --adapter A --output R --gpus 0 1 2 3      # fly every training route, keep failures
    aerorsi normal-inputs --rollouts R --output M               # round 1 only: normal states of successful flights
    aerorsi anchor   --rollouts R --normal-inputs M/normal_inputs.jsonl --output P
    aerorsi verify   --plan P --adapter A --output C --gpus 0 1 2 3
    aerorsi pool     --collection C --output D                  # the accepted repairs
    aerorsi relabel  --adapter A --inputs P/normal_train.jsonl --output N
    aerorsi assemble --repairs D --plan P --predictions N --output W [--replay W_prev]
    aerorsi train    --rows W --adapter A --output T            # the next policy is T/final

``scripts/run_loop.sh`` chains these for the three rounds of the paper.
"""
import argparse
from functools import partial
import json
from pathlib import Path
import sys
import time

import yaml

from .utils import read_json, write_json

ROOT = Path(__file__).resolve().parents[1]


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    resolve = lambda p: str(p if Path(p).is_absolute() else (ROOT / p).resolve())
    config["paths"] = {k: resolve(v) for k, v in config["paths"].items()}
    config["model"]["base"] = resolve(config["model"]["base"])
    return config


def start(output, config, args):
    """Create the stage's output directory with the effective configuration and command."""
    output = Path(output).resolve()  # absolute: simulator processes run from their own directories
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / "config.json", {"command": sys.argv, "args": {k: str(v) for k, v in vars(args).items()},
                                        "config": config, "started": time.strftime("%Y-%m-%d %H:%M:%S")})
    return output


def finish(output, metrics):
    write_json(Path(output) / "metrics.json", metrics, overwrite=True)
    print(json.dumps(metrics, indent=1)[:3000])


def training_routes(config):
    from .data import read_manifest
    return [r for r in read_manifest(config["paths"]["manifest"])["routes"] if r["split"] == "train"]


def select(records, args):
    """Routes of a navigation stage: scene-sorted (one simulator start per scene and worker), minus routes that
    already have a result in the --skip-done runs, then the first --limit."""
    done = {read_json(p)["route_id"] for run in args.skip_done for p in Path(run).glob("tasks/*/*/result.json")}
    records = sorted((r for r in records if r["id"] not in done), key=lambda r: (r["scene"], r["id"]))
    return records[:args.limit] if args.limit else records


def navigation(config, args, records, factory, output):
    from .workers import make_slots, run_pool
    res = config["resources"]
    slots = make_slots(args.gpus, args.workers_per_gpu or res["workers_per_gpu"], args.port_base or res["port_base"])
    return run_pool(records, slots, factory, output / "tasks", startup_stagger=res["startup_stagger_seconds"],
                    respawns=res["worker_respawns"])


def main(argv=None):
    from .sim.rpc import prefer_compiled_msgpack
    prefer_compiled_msgpack()
    parser = argparse.ArgumentParser(prog="aerorsi", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=str(ROOT / "configs/aerorsi.yaml"))
    sub = parser.add_subparsers(dest="command", required=True)

    def command(name, help_text, nav=False, output=True):
        p = sub.add_parser(name, help=help_text)
        if output:
            p.add_argument("--output", required=True, help="new directory for this stage's outputs")
        if nav:
            p.add_argument("--gpus", type=int, nargs="+", required=True, help="physical GPU ids for simulator workers")
            p.add_argument("--workers-per-gpu", type=int)
            p.add_argument("--port-base", type=int)
            p.add_argument("--limit", type=int, help="only the first N routes (smoke tests)")
            p.add_argument("--skip-done", nargs="+", default=[], help="skip routes that have a result in these runs")
        return p

    p = command("prepare", "build the route manifest from the TravelUAV dataset", output=False)
    p.add_argument("--dataset-root", required=True, help="AeroVLA's dataset_raw/ directory")
    p = command("init-adapter", "write the round-0 adapter (the SFT policy in the trainable layout)")
    p = command("rollout", "fly every training route with a policy, keeping the inputs before failures", nav=True)
    p.add_argument("--adapter", required=True)
    p = command("normal-inputs", "normal-state inputs from the round-0 rollouts' successful flights (once)")
    p.add_argument("--rollouts", nargs="+", required=True)
    p = command("anchor", "select this round's anchors from rollouts")
    p.add_argument("--rollouts", nargs="+", required=True)
    p.add_argument("--normal-inputs", required=True, help="normal_inputs.jsonl written by `aerorsi normal-inputs`")
    p = command("verify", "verify candidate repairs at the anchors by continuation", nav=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--adapter", required=True)
    p = command("pool", "collect the accepted repairs of a verification run")
    p.add_argument("--collection", required=True)
    p = command("relabel", "the current policy's own outputs on the normal-state inputs (one GPU)")
    p.add_argument("--adapter", required=True)
    p.add_argument("--inputs", required=True, help="normal_train.jsonl of this round's plan")
    p = command("assemble", "weighted training rows of one round")
    p.add_argument("--repairs", required=True)
    p.add_argument("--plan", required=True)
    p.add_argument("--predictions", required=True, help="relabel output directory")
    p.add_argument("--replay", help="previous round's assemble directory (round 2 on)")
    p = command("train", "train the next policy from the previous adapter (one GPU)")
    p.add_argument("--rows", required=True, help="assemble output directory")
    p.add_argument("--adapter", required=True)
    p = command("evaluate", "closed-loop evaluation on the official test splits", nav=True)
    p.add_argument("--adapter", required=True)
    p.add_argument("--splits", nargs="+", default=["seen", "unseen_object", "unseen_map"])
    p = command("metrics", "SR/OSR/SPL/NE/CR tables of one or more evaluation runs (their union)", output=False)
    p.add_argument("--runs", nargs="+", required=True)
    args = parser.parse_args(argv)
    config = load_config(args.config)

    if args.command == "prepare":
        from .data import build_manifest
        print(json.dumps(build_manifest(args.dataset_root, config["paths"]["manifest"]), indent=1))
        return
    if args.command == "metrics":
        from .data import read_manifest
        from .metrics import summarize, table
        results = [read_json(p) for run in args.runs for p in sorted(Path(run).glob("tasks/*/*/result.json"))]
        expected = [r["id"] for r in read_manifest(config["paths"]["manifest"])["routes"] if r["split"] == "test"]
        summary = summarize(results, expected)
        print(table(summary))
        if summary["status"] != "complete":
            print(f"\nIncomplete: {len(summary['missing'])} test routes have no result. Fly them with "
                  "`aerorsi evaluate ... --skip-done <these runs>` and pass all runs to `aerorsi metrics`.")
        return

    output = start(args.output, config, args)
    if args.command == "init-adapter":
        from .policy import init_adapter
        metrics = init_adapter(config["model"]["base"], output / "adapter", config["seed"])
    elif args.command == "rollout":
        from .flight import FlightWorker
        factory = partial(FlightWorker, config=config, mode="rollout", adapter=str(Path(args.adapter).resolve()))
        metrics = navigation(config, args, select(training_routes(config), args), factory, output)
    elif args.command == "normal-inputs":
        from .data import read_manifest
        from .repair.normal import collect
        metrics = collect(args.rollouts, read_manifest(config["paths"]["manifest"])["identity"], config["seed"],
                          config["normal"]["flights"], output)
    elif args.command == "anchor":
        from .data import read_manifest
        from .repair.anchors import plan
        metrics = plan(config, read_manifest(config["paths"]["manifest"]), args.rollouts, args.normal_inputs, output)
    elif args.command == "verify":
        from .repair.verify import RepairWorker
        plan = read_json(Path(args.plan) / "plan.json")
        if plan["settings"] != config["repair"]:
            raise ValueError("The repair settings differ from those the anchors were planned with")
        (output / "plan.json").symlink_to(Path(args.plan).resolve() / "plan.json")
        factory = partial(RepairWorker, config=config, adapter=str(Path(args.adapter).resolve()))
        metrics = navigation(config, args, select(plan["records"], args), factory, output)
    elif args.command == "pool":
        from .repair.assemble import pool
        metrics = pool(args.collection, output)
    elif args.command == "relabel":
        from .repair.assemble import relabel
        metrics = relabel(config, args.adapter, args.inputs, output)
    elif args.command == "assemble":
        from .repair.assemble import assemble
        metrics = assemble(args.repairs, args.plan, Path(args.predictions) / "predictions.jsonl", output, config["seed"],
                           config["assembly"], args.replay)
    elif args.command == "train":
        from .train import train
        metrics = train(config, Path(args.rows) / "train_rows.jsonl", args.adapter, output)
    elif args.command == "evaluate":
        from .data import read_manifest
        from .flight import FlightWorker
        from .metrics import summarize
        routes = select([r for r in read_manifest(config["paths"]["manifest"])["routes"]
                         if r["split"] == "test" and r["category"] in args.splits], args)
        factory = partial(FlightWorker, config=config, mode="evaluate", adapter=str(Path(args.adapter).resolve()))
        queue = navigation(config, args, routes, factory, output)
        results = [read_json(p) for p in sorted((output / "tasks").glob("*/*/result.json"))]
        metrics = {"queue": queue, **summarize(results, [r["id"] for r in routes])}
    finish(output, metrics)


if __name__ == "__main__":
    main()
