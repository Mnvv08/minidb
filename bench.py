"""Benchmarks: minidb against Python's built-in sqlite3 on the same workload.

    python bench.py            (about 20 seconds; python bench.py 20000 for more rows)

sqlite3 is a C library with decades of tuning, so it should win; the point is to see WHERE minidb
is slow and why, and to check that the primary-key planner really changes the cost of a query.
"""
import os
import random
import sqlite3
import sys
import tempfile
import time

from db import Database

N = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
AUTOCOMMIT_ROWS = 200


def timed(fn):
    start = time.perf_counter()
    result = fn()
    return time.perf_counter() - start, result


def minidb_run(path):
    d = Database(path)
    out = {}
    d.execute("CREATE TABLE t (id INT PRIMARY KEY, grp INT, name TEXT)")
    rng = random.Random(1)
    rows = [(i, rng.randrange(100), f"user{i}") for i in range(N)]

    def bulk():
        d.execute("BEGIN")
        for i in range(0, N, 500):
            d.execute("INSERT INTO t VALUES " + ", ".join(f"({a}, {b}, '{c}')" for a, b, c in rows[i:i + 500]))
        d.execute("COMMIT")
    out["bulk insert (1 transaction)"] = timed(bulk)[0]
    out["point lookups (x1000, by key)"] = timed(lambda: [d.execute(f"SELECT name FROM t WHERE id = {rng.randrange(N)}") for _ in range(1000)])[0]
    out["range scan (100 rows, x100)"] = timed(lambda: [d.execute(f"SELECT * FROM t WHERE id BETWEEN {s} AND {s + 99}") for s in (rng.randrange(N - 100) for _ in range(100))])[0]
    out["full scan with filter (x5)"] = timed(lambda: [d.execute("SELECT count(*) FROM t WHERE grp = 7") for _ in range(5)])[0]
    out["ORDER BY + LIMIT 10 (x5)"] = timed(lambda: [d.execute("SELECT id FROM t ORDER BY grp, id LIMIT 10") for _ in range(5)])[0]
    out["update 1000 rows, one statement"] = timed(lambda: d.execute(f"UPDATE t SET grp = grp + 1 WHERE id < 1000"))[0]
    out["delete 1000 rows, one statement"] = timed(lambda: d.execute(f"DELETE FROM t WHERE id >= {N - 1000}"))[0]
    base = N + 10

    def autocommit():
        for i in range(AUTOCOMMIT_ROWS):
            d.execute(f"INSERT INTO t VALUES ({base + i}, 1, 'x')")
    out[f"{AUTOCOMMIT_ROWS} single inserts (fsync each)"] = timed(autocommit)[0]
    d.tree.validate()
    d.close()
    return out


def sqlite_run(path):
    c = sqlite3.connect(path, isolation_level=None)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA synchronous=FULL")
    out = {}
    c.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, grp INT, name TEXT)")
    rng = random.Random(1)
    rows = [(i, rng.randrange(100), f"user{i}") for i in range(N)]

    def bulk():
        c.execute("BEGIN")
        c.executemany("INSERT INTO t VALUES (?, ?, ?)", rows)
        c.execute("COMMIT")
    out["bulk insert (1 transaction)"] = timed(bulk)[0]
    out["point lookups (x1000, by key)"] = timed(lambda: [c.execute("SELECT name FROM t WHERE id = ?", (rng.randrange(N),)).fetchall() for _ in range(1000)])[0]
    out["range scan (100 rows, x100)"] = timed(lambda: [c.execute("SELECT * FROM t WHERE id BETWEEN ? AND ?", (s, s + 99)).fetchall() for s in (rng.randrange(N - 100) for _ in range(100))])[0]
    out["full scan with filter (x5)"] = timed(lambda: [c.execute("SELECT count(*) FROM t WHERE grp = 7").fetchall() for _ in range(5)])[0]
    out["ORDER BY + LIMIT 10 (x5)"] = timed(lambda: [c.execute("SELECT id FROM t ORDER BY grp, id LIMIT 10").fetchall() for _ in range(5)])[0]
    out["update 1000 rows, one statement"] = timed(lambda: c.execute("UPDATE t SET grp = grp + 1 WHERE id < 1000"))[0]
    out["delete 1000 rows, one statement"] = timed(lambda: c.execute(f"DELETE FROM t WHERE id >= {N - 1000}"))[0]
    base = N + 10

    def autocommit():
        for i in range(AUTOCOMMIT_ROWS):
            c.execute("INSERT INTO t VALUES (?, 1, 'x')", (base + i,))
    out[f"{AUTOCOMMIT_ROWS} single inserts (fsync each)"] = timed(autocommit)[0]
    c.close()
    return out


def main():
    with tempfile.TemporaryDirectory() as tmp:
        mine = minidb_run(os.path.join(tmp, "bench.db"))
        theirs = sqlite_run(os.path.join(tmp, "bench.sqlite"))
    print(f"{N} rows. Times in milliseconds (lower is better).\n")
    print(f"{'workload':38} {'minidb':>10} {'sqlite3':>10} {'slower by':>10}")
    for name in mine:
        a, b = mine[name] * 1000, theirs[name] * 1000
        print(f"{name:38} {a:10.1f} {b:10.1f} {a / b:9.1f}x")


if __name__ == "__main__":
    main()
