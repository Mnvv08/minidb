# minidb

A small relational database engine written from scratch in Python, **standard library only**.
It stores data in a single file using a disk-based B+ tree, survives power cuts through a
write-ahead log, and speaks a useful subset of SQL, with a query planner and an interactive shell.

```
minidb> CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL, age INT);
minidb> INSERT INTO users VALUES (1, 'ann', 30), (2, 'bob', 25), (3, 'cy', 41);
minidb> .plan on
minidb> SELECT name FROM users WHERE id BETWEEN 2 AND 3 ORDER BY age DESC;
+------+
| name |
+------+
| cy   |
| bob  |
+------+
(2 rows)
plan: SEARCH users USING PRIMARY KEY (id BETWEEN 2 AND 3)
```

## Run it

Python 3.10 or newer (tested here on 3.13). No dependencies to run; `pytest` to test.

```bash
python shell.py my.db                      # interactive shell (.help lists commands)
python shell.py my.db -c "SELECT 1 + 1"    # one-off
python shell.py my.db < script.sql         # run a script
python crash_demo.py                       # watch a simulated power cut get recovered
python bench.py                            # compare with sqlite3
pytest -q                                  # the test suite
```

From Python:

```python
from db import Database
with Database("my.db") as db:
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
    db.execute("INSERT INTO t VALUES (1, 'hello')")
    print(db.execute("SELECT * FROM t").rows)     # [(1, 'hello')]
```

## How it is built

Each layer only talks to the one below it.

| File | Layer | What it does |
|---|---|---|
| `pager.py` | storage | Fixed 4096-byte pages in one file, a page cache, a free list. |
| `slotted_page.py` | storage | Variable-length records inside a page (slot directory, compaction). |
| `btree.py` | index | Disk B+ tree: insert, lookup, range scan along the leaf chain, delete with merge/redistribute, `validate()` integrity checker, `transaction()` context manager. |
| `wal.py` | durability | Write-ahead log: full-page redo records with CRC and a commit marker, one fsync per commit, checkpoints, idempotent crash recovery. |
| `sql.py` | language | Tokenizer and recursive-descent parser to an immutable AST. Errors carry line, column and a caret. `parse(ast.sql()) == ast` round-trips. |
| `db.py` | engine | Catalog, row format, expression evaluator with SQL's three-valued logic, planner, executor, transactions. |
| `shell.py` | interface | Interactive prompt with multi-line input, history and dot-commands. |

Key design decisions:

* **One tree holds everything.** A key is `(table id << 40) + primary key + 2**39`, so every table owns a contiguous slice of key space and a table scan is a range scan. Table 0 is the catalog.
* **Statements are atomic.** Each one validates and plans *before* its first write, then runs inside a WAL transaction. `BEGIN ... COMMIT` groups several into one fsync.
* **The planner can only make queries faster, never change answers.** Primary-key conditions (`=`, `<`, `<=`, `>`, `>=`, `BETWEEN`, joined by `AND`) become a point lookup or range scan. The full `WHERE` is still applied to every fetched row, and a randomized test compares indexed against unindexed execution.
* **SQL semantics are real.** `NULL = NULL` is NULL, `x NOT IN (1, NULL)` is never true, integer division truncates toward zero, int64 overflow is an error, comparing INT with TEXT is an error, `LIKE` runs in linear time (no regex backtracking).

## SQL supported

`CREATE TABLE [IF NOT EXISTS]`, `DROP TABLE [IF EXISTS]`, `INSERT` (multi-row, optional column list, automatic keys), `SELECT` (expressions, `*`, aliases, `WHERE`, `ORDER BY` multiple keys, `LIMIT`/`OFFSET`), `UPDATE`, `DELETE`, `BEGIN`/`COMMIT`/`ROLLBACK`.
Types: `INT`, `REAL`, `TEXT`, `BOOL`. Operators: `+ - * / %`, comparisons, `AND OR NOT`, `IS [NOT] NULL`, `[NOT] BETWEEN`, `[NOT] IN (...)`, `[NOT] LIKE`.
Functions: `abs length lower upper round coalesce`, aggregates `count sum avg min max`.

## Testing

The test suite is the most important part of this project; it checks the engine in ways example-based tests cannot:

* **Crash tests.** A fault hook stops the process at *every* I/O event of a workload, the files are copied exactly as they were, and the database is reopened. After every possible crash point the data is either entirely the old state or entirely the new one, and `validate()` passes. Done for the B+ tree, and again for multi-table SQL transactions and every autocommitted statement.
* **Model-based tests.** Random operations run against both minidb and a plain Python dict/list; results must match after every step, including across reopen, rollback and `validate()`.
* **Oracle and round-trip tests.** `LIKE` is compared with a regex reference, the SQL printer with the parser, the planner with a full scan.
* **Fuzzing.** Thousands of mutated garbage statements must never crash the engine or corrupt a file.
* **Mutation testing.** I deliberately broke the code in dozens of small ways to check the tests notice. Nearly all were caught; the survivors were equivalent mutants (changes that do not alter behaviour).

## Honest limits

* A row (and a table definition) must fit in one B+ tree value: 1012 bytes. No overflow pages.
* `INT PRIMARY KEY` is limited to +-2**39.
* Only the primary key is indexed. Other filters scan the whole table.
* No joins, `GROUP BY`, `DISTINCT`, `ALTER TABLE`, secondary indexes, or concurrent access (one process at a time, no file locking).
* The planner uses only top-level `AND` conditions on the key (no `OR` / `IN`).
* Aggregates cannot be combined with `ORDER BY`.
* Every autocommitted statement fsyncs, so single inserts are slow; wrap bulk loads in `BEGIN ... COMMIT`.
* On macOS, `fsync` does not force data to the platter (that needs `F_FULLFSYNC`), so power-loss durability there is weaker than on Linux.
* It is pure Python: expect it to be one to two orders of magnitude slower than SQLite (see `bench.py`). That is the price of readable code, not a bug.
