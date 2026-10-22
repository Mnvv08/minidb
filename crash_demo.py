"""Kill a database writer with SIGKILL, over and over, and check that nothing acknowledged is lost.

    python crash_demo.py            # 15 rounds
    python crash_demo.py 40         # 40 rounds

Each round starts a writer process that calls put() in a loop. After every put() has returned
(= the database promised it is durable) the writer records the key in a separate file. The
parent kills the writer at a random moment with SIGKILL, which gives the process no chance to
clean up. Then the database is reopened and checked:

  1. the tree must be structurally valid,
  2. every key the writer was told was stored must be there.
"""
import os
import random
import signal
import subprocess
import sys
import time

from btree import BTree

DB = "crashdemo.db"
ACKED = "crashdemo.acked"


def worker(start):
    tree = BTree(DB)                                    # real fsync on every commit
    fd = os.open(ACKED, os.O_WRONLY | os.O_APPEND | os.O_CREAT)
    key = start
    while True:
        tree.put(key, b"value-%d" % key)
        os.write(fd, b"%d\n" % key)                     # only after put() returned
        key += 1


def acked_keys():
    if not os.path.exists(ACKED):
        return []
    return [int(x) for x in open(ACKED).read().split()]


def cleanup():
    for f in (DB, DB + "-wal", ACKED):
        if os.path.exists(f):
            os.remove(f)


def main(rounds):
    cleanup()
    rng = random.Random()
    next_key, replayed_total, failures, unacked_before = 0, 0, 0, 0
    for r in range(1, rounds + 1):
        proc = subprocess.Popen([sys.executable, __file__, "worker", str(next_key)])
        time.sleep(rng.uniform(0.15, 0.8))
        proc.send_signal(signal.SIGKILL)
        proc.wait()

        acked = acked_keys()
        with BTree(DB) as tree:                         # opening runs recovery
            replayed = tree.pager.recovered_txns
            keys = tree.validate()
            missing = [k for k in acked if tree.get(k) is None]
            unacked_total = keys - len(acked)           # keys stored although never acknowledged
            newest = max((k for k, _ in tree.items()), default=-1)
        replayed_total += replayed
        status = "OK" if not missing else f"LOST {len(missing)} ACKNOWLEDGED KEYS"
        failures += bool(missing)
        print(f"round {r:2}: killed after {len(acked):5} acknowledged puts | "
              f"recovery replayed {replayed:3} logged transaction(s) | "
              f"{unacked_total - unacked_before} un-acknowledged key(s) also survived | tree valid | {status}")
        unacked_before = unacked_total
        next_key = newest + 1
    print()
    print(f"{rounds} kills, {replayed_total} transactions replayed from the log, "
          f"{len(acked_keys())} keys acknowledged in total: "
          + ("nothing acknowledged was lost." if not failures else "DATA WAS LOST."))
    cleanup()
    return 1 if failures else 0


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "worker":
        worker(int(sys.argv[2]))
    else:
        sys.exit(main(int(sys.argv[1]) if len(sys.argv) > 1 else 15))
