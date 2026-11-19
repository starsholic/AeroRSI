"""Run whole-route tasks on simulator workers, several per GPU.

Each worker process owns one simulator (its own RPC port) and one policy copy, and takes routes from a shared queue
until none remain, so a slow route never blocks the others. A task that raises leaves an ``infrastructure_error.json``
and is never retried silently; its worker is replaced (on a fresh port) a bounded number of times.
"""
import multiprocessing as mp
import os
from pathlib import Path
from queue import Empty
import signal
import time
import traceback


def make_slots(gpus, workers_per_gpu, port_base):
    """Worker slots, interleaved over GPUs so that a partial allocation still spreads the load."""
    if not 1 <= workers_per_gpu <= 8 or len(set(gpus)) != len(gpus):
        raise ValueError("Use distinct GPUs and 1-8 workers per GPU")
    ordered = [gpu for _ in range(workers_per_gpu) for gpu in gpus]
    if not 1024 <= port_base or port_base + len(ordered) > 65535:
        raise ValueError("Invalid RPC port range")
    return [{"id": str(i), "gpu": gpu, "port": port_base + i} for i, gpu in enumerate(ordered)]


def _worker(tasks, queue, slot, factory, folder):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(slot["gpu"])
    with factory(slot, folder) as worker:
        while True:
            try:
                index = queue.get_nowait()
            except Empty:
                return
            try:
                worker.run(tasks[index])
            except BaseException:
                (folder / f"task_{index}_error.txt").write_text(traceback.format_exc())
                raise


def run_pool(tasks, slots, factory, output, *, startup_stagger=3.0, respawns=2):
    """Run every task once; returns {'tasks', 'complete', 'worker_exitcodes'}.

    factory(slot, folder) returns a context manager with run(task), which must write its results under folder.
    """
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    context = mp.get_context("spawn")
    queue = context.Queue()
    for index in range(len(tasks)):
        queue.put(index)
    workers, replaced = [], {}

    def start(slot):
        folder = output / slot["id"]
        folder.mkdir()
        process = context.Process(target=_worker, args=(tasks, queue, slot, factory, folder))
        process.start()
        workers.append((process, slot))
        time.sleep(startup_stagger)

    try:
        for slot in slots[:len(tasks)]:
            start(slot)
        live = list(workers)
        while live:
            time.sleep(1)
            for item in list(live):
                process, slot = item
                if process.is_alive():
                    continue
                process.join()
                live.remove(item)
                base = slot.get("base", slot["id"])
                count = replaced.get(base, 0)
                if process.exitcode != 0 and count < respawns and not queue.empty():
                    replaced[base] = count + 1
                    new = {**slot, "id": f"{base}_respawn{count + 1}", "base": base,
                           "port": slot["port"] + 100 * (count + 1)}  # a fresh port: the old one may still be bound
                    start(new)
                    live.append(workers[-1])
    finally:
        for process, _ in workers:
            if process.is_alive():  # interrupt, so that each worker closes its own simulator
                os.kill(process.pid, signal.SIGINT)
                process.join(timeout=30)
    complete = len(list(output.glob("*/*/result.json")))
    return {"tasks": len(tasks), "complete": complete, "worker_exitcodes": [p.exitcode for p, _ in workers]}
