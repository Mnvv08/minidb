import os
import random
import re
import shutil

import pytest

import sql as S
from db import (INT_MAX, INT_MIN, PK_MAX, PK_MIN, Database, SQLError, Table, TransactionAborted,
                format_result, key_pk, like, pack_key, table_range)
from wal import FRAME_SIZE


@pytest.fixture
def db(tmp_path):
    d = Database(str(tmp_path / "t.db"))
    yield d
    d.close()


def rows(db, text):
    return db.execute(text).rows


def one(db, text):
    r = db.execute(text).rows
    assert len(r) == 1 and len(r[0]) == 1
    return r[0][0]


def fails(db, text, message):
    with pytest.raises(SQLError, match=message):
        db.execute(text)


def same(got, expected):
    """Equal AND the same type (True must not pass for 1, 1.0 must not pass for 1)."""
    assert (type(got), got) == (type(expected), expected)


def script(db, text):
    return db.execute_script(text)


# ---------- keys and rows ----------

def test_key_packing_keeps_tables_apart_and_in_order():
    lo1, hi1 = table_range(1)
    lo2, hi2 = table_range(2)
    assert hi1 < lo2
    assert pack_key(1, PK_MIN) == lo1 and pack_key(1, PK_MAX) == hi1
    for pk in (PK_MIN, -1, 0, 1, 12345, PK_MAX):
        assert key_pk(pack_key(7, pk)) == pk
    assert pack_key(3, -5) < pack_key(3, 0) < pack_key(3, 5)
    assert -(2 ** 63) <= pack_key(0, PK_MIN) and pack_key((1 << 22) - 1, PK_MAX) < 2 ** 63


def test_row_codec_roundtrip_with_every_type_and_nulls():
    cols = [S.ColumnDef("id", "INT", True, True), S.ColumnDef("a", "INT"), S.ColumnDef("b", "REAL"),
            S.ColumnDef("c", "BOOL"), S.ColumnDef("d", "TEXT")]
    t = Table(5, "x", cols)
    rng = random.Random(1)
    samples = [INT_MIN, INT_MAX, 0, -1, 1, 2 ** 40]
    for _ in range(500):
        row = (rng.randrange(-1000, 1000),
               rng.choice(samples + [None]),
               rng.choice([None, 0.0, -0.0, 1.5, -1e300, 1e-300, 123456.789]),
               rng.choice([None, True, False]),
               rng.choice([None, "", "plain", "héllo wörld ✓", "x" * rng.randrange(0, 300), "日本語"]))
        key = pack_key(5, row[0])
        back = t.decode(key, t.encode(row))
        assert back == row
        assert [type(v) for v in back] == [type(v) for v in row]


def test_row_codec_without_a_primary_key_and_with_only_a_key():
    t = Table(1, "x", [S.ColumnDef("a", "INT"), S.ColumnDef("b", "TEXT")])
    assert t.pk is None and t.decode(pack_key(1, 9), t.encode((3, "q"))) == (3, "q")
    k = Table(2, "k", [S.ColumnDef("id", "INT", True, True)])
    assert k.encode((7,)) == b"" and k.decode(pack_key(2, 7), b"") == (7,)


def test_corrupt_row_is_reported_not_crashed():
    t = Table(1, "x", [S.ColumnDef("a", "INT"), S.ColumnDef("b", "TEXT")])
    for bad in (b"", b"\x00", b"\x00" + b"\x01" * 4, b"\x00" + (5).to_bytes(8, "little") + b"\xff\xff"):
        with pytest.raises(SQLError, match="corrupt row"):
            t.decode(pack_key(1, 1), bad)


# ---------- basic statements ----------

def test_create_insert_select_all_types(db):
    script(db, """CREATE TABLE t (id INT PRIMARY KEY, n INT, x REAL, ok BOOL, s TEXT);
                  INSERT INTO t VALUES (1, -5, 2.5, TRUE, 'héllo'), (2, NULL, NULL, NULL, NULL), (3, 9, 1, FALSE, '')""")
    r = db.execute("SELECT * FROM t")
    assert r.columns == ("id", "n", "x", "ok", "s")
    assert r.rows == [(1, -5, 2.5, True, "héllo"), (2, None, None, None, None), (3, 9, 1.0, False, "")]
    same(r.rows[2][2], 1.0)                                        # an INT stored in a REAL column is a float
    assert r.rowcount == 3


def test_column_list_and_defaults(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, a INT, b TEXT)")
    db.execute("INSERT INTO t (b, id) VALUES ('x', 5)")
    assert rows(db, "SELECT * FROM t") == [(5, None, "x")]


def test_automatic_primary_key(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT)")
    db.execute("INSERT INTO t (v) VALUES ('a'), ('b')")
    db.execute("INSERT INTO t VALUES (10, 'c')")
    db.execute("INSERT INTO t (v) VALUES ('d')")
    db.execute("INSERT INTO t VALUES (5, 'e')")
    db.execute("INSERT INTO t (v) VALUES ('f')")
    assert rows(db, "SELECT id, v FROM t") == [(1, "a"), (2, "b"), (5, "e"), (10, "c"), (11, "d"), (12, "f")]


def test_select_headers_aliases_and_expressions(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, a INT); INSERT INTO t VALUES (1, 10)")
    r = db.execute("SELECT a, a + 1, a * 2 AS dbl, upper('x') FROM t")
    assert r.columns == ("a", "a + 1", "dbl", "upper('x')")
    assert r.rows == [(10, 11, 20, "X")]
    assert db.execute("SELECT t.a FROM t").rows == [(10,)]


def test_select_without_a_table(db):
    assert db.execute("SELECT 1 + 1, 'a'").rows == [(2, "a")]
    fails(db, "SELECT *", "needs a FROM")
    fails(db, "SELECT x", "no such column")


def test_everything_persists_across_close_and_reopen(tmp_path):
    path = str(tmp_path / "p.db")
    with Database(path) as d:
        script(d, """CREATE TABLE a (id INT PRIMARY KEY, v TEXT); CREATE TABLE b (x INT, y REAL);
                     INSERT INTO a VALUES (1, 'one'), (2, 'two'); INSERT INTO b VALUES (1, 1.5), (2, 2.5);
                     UPDATE a SET v = 'TWO' WHERE id = 2; DELETE FROM b WHERE x = 1""")
    with Database(path) as d:
        assert sorted(d.tables) == ["a", "b"]
        assert rows(d, "SELECT * FROM a") == [(1, "one"), (2, "TWO")]
        assert rows(d, "SELECT * FROM b") == [(2, 2.5)]
        d.execute("INSERT INTO a (v) VALUES ('three')")
        assert rows(d, "SELECT id FROM a WHERE v = 'three'") == [(3,)]
        d.tree.validate()


def test_schema_text(db):
    script(db, "CREATE TABLE a (id INT PRIMARY KEY, n TEXT NOT NULL, r REAL); CREATE TABLE b (x BOOL)")
    assert db.schema("a") == "CREATE TABLE a (id INT PRIMARY KEY, n TEXT NOT NULL, r REAL)"
    assert db.schema().splitlines() == [db.schema("a"), db.schema("b")]


def test_format_result_table_and_messages(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, s TEXT); INSERT INTO t VALUES (1, 'ab'), (2, NULL)")
    out = format_result(db.execute("SELECT * FROM t"))
    assert out.splitlines() == ["+----+------+", "| id | s    |", "+----+------+", "| 1  | ab   |",
                                "| 2  | NULL |", "+----+------+", "(2 rows)"]
    assert format_result(db.execute("INSERT INTO t VALUES (3, 'z')")) == "INSERT 1"
    assert format_result(db.execute("SELECT 1")).endswith("(1 row)")


# ---------- constraints and errors ----------

@pytest.mark.parametrize("statement, message", [
    ("INSERT INTO t VALUES (1, 'dup', 1)", "UNIQUE constraint failed: t.id \\(1\\)"),
    ("INSERT INTO t VALUES (7, NULL, 1)", "NOT NULL constraint failed: t.name"),
    ("INSERT INTO t VALUES (NULL, 'x', 1)", "NOT NULL constraint failed: t.id"),
    ("INSERT INTO t VALUES ('7', 'x', 1)", "column 'id' expects INT but got TEXT"),
    ("INSERT INTO t VALUES (7, 5, 1)", "column 'name' expects TEXT but got INT"),
    ("INSERT INTO t VALUES (7, 'x', 'q')", "column 'age' expects INT but got TEXT"),
    ("INSERT INTO t VALUES (7, 'x', 1.5)", "column 'age' expects INT but got REAL"),
    ("INSERT INTO t VALUES (7, 'x', TRUE)", "column 'age' expects INT but got BOOL"),
    ("INSERT INTO t VALUES (7, 'x', 9223372036854775807 + 1)", "integer overflow"),
    ("INSERT INTO t VALUES (7, 'x')", "has 3 columns but 2 values"),
    ("INSERT INTO t (id, nope) VALUES (7, 1)", "no column named nope"),
    ("INSERT INTO t (id, id) VALUES (7, 1)", "specified twice"),
    ("INSERT INTO t VALUES (549755813888, 'x', 1)", "out of range"),
    ("INSERT INTO t VALUES (-549755813889, 'x', 1)", "out of range"),
    ("INSERT INTO t VALUES (7, 'x', y)", "no such column: y"),
    ("INSERT INTO nope VALUES (1)", "no such table: nope"),
    ("INSERT INTO t VALUES (2, 'ok', 1), (2, 'same id twice', 1)", "UNIQUE constraint failed"),
    ("UPDATE t SET nope = 1", "no column named nope"),
    ("UPDATE t SET age = 1, age = 2", "assigned twice"),
    ("UPDATE t SET name = NULL", "NOT NULL constraint failed: t.name"),
    ("UPDATE t SET age = 'x'", "expects INT but got TEXT"),
    ("UPDATE t SET id = 2 WHERE id = 1", "UNIQUE constraint failed"),
    ("UPDATE t SET id = NULL", "NOT NULL constraint failed: t.id"),
    ("UPDATE t SET id = 0", "UNIQUE constraint failed"),
    ("UPDATE t SET age = count(*)", "not allowed"),
    ("UPDATE t SET age = 1 WHERE nope = 1", "no such column: nope"),
    ("UPDATE nope SET a = 1", "no such table"),
    ("DELETE FROM t WHERE nope = 1", "no such column: nope"),
    ("DELETE FROM t WHERE age", "WHERE must be a boolean expression, got INT"),
    ("SELECT * FROM t WHERE name", "WHERE must be a boolean expression, got TEXT"),
    ("SELECT nope FROM t", "no such column: nope"),
    ("SELECT x.id FROM t", "no such table: x"),
    ("SELECT * FROM nope", "no such table: nope"),
    ("SELECT * FROM t ORDER BY nope", "no such column: nope"),
    ("CREATE TABLE t (a INT)", "already exists"),
    ("CREATE TABLE u (a TEXT PRIMARY KEY)", "must be INT"),
    ("DROP TABLE nope", "no such table"),
])
def test_statement_errors(db, statement, message):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, name TEXT NOT NULL, age INT);"
               "INSERT INTO t VALUES (1, 'a', 10), (2, 'b', 20), (0, 'z', 5)")
    before = rows(db, "SELECT * FROM t")
    fails(db, statement, message)
    assert rows(db, "SELECT * FROM t") == before                  # a failed statement changes nothing
    db.tree.validate()


def test_if_exists_and_if_not_exists(db):
    db.execute("CREATE TABLE t (a INT)")
    db.execute("CREATE TABLE IF NOT EXISTS t (a INT, b INT)")
    assert db.table("t").names == ["a"]
    db.execute("DROP TABLE IF EXISTS nope")
    db.execute("DROP TABLE IF EXISTS t")
    assert "t" not in db.tables


def test_primary_key_range_edges(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY)")
    db.execute(f"INSERT INTO t VALUES ({PK_MIN}), ({PK_MAX}), (0)")
    assert rows(db, "SELECT id FROM t") == [(PK_MIN,), (0,), (PK_MAX,)]
    assert rows(db, f"SELECT id FROM t WHERE id >= {PK_MAX}") == [(PK_MAX,)]
    assert rows(db, f"SELECT id FROM t WHERE id <= {PK_MIN}") == [(PK_MIN,)]
    assert rows(db, f"SELECT id FROM t WHERE id > {PK_MAX} * 4") == []
    assert rows(db, f"SELECT id FROM t WHERE id < {PK_MIN} * 4") == []
    assert len(rows(db, f"SELECT id FROM t WHERE id > {PK_MIN} * 4")) == 3


def test_non_key_int_columns_use_the_full_64_bits(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, n INT)")
    db.execute(f"INSERT INTO t VALUES (1, {INT_MIN}), (2, {INT_MAX})")
    assert rows(db, "SELECT n FROM t") == [(INT_MIN,), (INT_MAX,)]


def test_row_size_limit_is_exact(db):
    db.execute("CREATE TABLE t (id INT PRIMARY KEY, s TEXT)")
    biggest = db.tree.max_value - 3                                # 1 bitmap byte + 2 length bytes
    db.execute(f"INSERT INTO t VALUES (1, '{'x' * biggest}')")
    fails(db, f"INSERT INTO t VALUES (2, '{'x' * (biggest + 1)}')", "row is too large")
    fails(db, f"UPDATE t SET s = '{'y' * (biggest + 1)}' WHERE id = 1", "row is too large")
    assert len(one(db, "SELECT s FROM t")) == biggest
    db.tree.validate()


def test_table_definition_size_limit(db):
    many = ", ".join(f"column_number_{i} INT" for i in range(60))
    fails(db, f"CREATE TABLE wide ({many})", "table definition is too large")
    assert "wide" not in db.tables
    db.execute("CREATE TABLE ok (" + ", ".join(f"c{i} INT" for i in range(30)) + ")")


# ---------- statement atomicity ----------

def test_multi_row_insert_that_fails_midway_inserts_nothing(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT NOT NULL); INSERT INTO t VALUES (1, 'a')")
    fails(db, "INSERT INTO t VALUES (2, 'b'), (3, 'c'), (1, 'dup')", "UNIQUE")
    fails(db, "INSERT INTO t VALUES (2, 'b'), (3, NULL)", "NOT NULL")
    fails(db, "INSERT INTO t VALUES (2, 'b'), (3, 1/0)", "division by zero")
    assert rows(db, "SELECT id FROM t") == [(1,)]
    assert db.tables["t"].next_rowid == 2                         # even the key counter is untouched
    db.tree.validate()


def test_update_that_fails_on_the_last_row_changes_nothing(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, n INT); INSERT INTO t VALUES (1, 1), (2, 2), (3, 0)")
    fails(db, "UPDATE t SET n = 10 / n", "division by zero")
    assert rows(db, "SELECT n FROM t") == [(1,), (2,), (0,)]


def test_failed_statement_inside_begin_keeps_the_transaction_alive(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v INT); BEGIN; INSERT INTO t VALUES (1, 1)")
    fails(db, "INSERT INTO t VALUES (2, 2), (1, 9)", "UNIQUE")
    assert db.in_transaction
    db.execute("INSERT INTO t VALUES (3, 3)")
    db.execute("COMMIT")
    assert rows(db, "SELECT id FROM t") == [(1,), (3,)]


# ---------- expressions ----------

@pytest.mark.parametrize("expression, expected", [
    ("1 + 2 * 3", 7), ("(1 + 2) * 3", 9), ("10 - 4 - 3", 3), ("2 * -3", -6), ("-(-5)", 5), ("- 3", -3),
    ("7 / 2", 3), ("-7 / 2", -3), ("7 / -2", -3), ("-7 / -2", 3), ("7 % 3", 1), ("-7 % 3", -1), ("7 % -3", 1),
    ("7.0 / 2", 3.5), ("7 / 2.0", 3.5), ("1.5 + 1", 2.5), ("5.5 % 2", 1.5), ("2 * 0.5", 1.0),
    ("9223372036854775807", INT_MAX), ("-9223372036854775807 - 1", INT_MIN), ("-9223372036854775808", INT_MIN),
    ("NULL + 1", None), ("1 * NULL", None), ("-NULL", None),
    ("1 = 1", True), ("1 = 1.0", True), ("1 < 2", True), ("2 <= 1", False), ("'a' < 'b'", True),
    ("'a' = 'A'", False), ("'é' > 'e'", True), ("TRUE = TRUE", True), ("FALSE < TRUE", True), ("1 != 2", True),
    ("NULL = NULL", None), ("NULL != 1", None), ("1 < NULL", None),
    ("NOT TRUE", False), ("NOT NULL", None),
    ("TRUE AND TRUE", True), ("TRUE AND FALSE", False), ("TRUE AND NULL", None), ("NULL AND FALSE", False),
    ("FALSE AND NULL", False), ("NULL AND NULL", None), ("FALSE OR FALSE", False), ("NULL OR TRUE", True),
    ("TRUE OR NULL", True), ("NULL OR FALSE", None), ("FALSE OR NULL", None), ("NULL OR NULL", None),
    ("FALSE AND 1/0 = 1", False), ("TRUE OR 1/0 = 1", True),
    ("NULL IS NULL", True), ("1 IS NULL", False), ("1 IS NOT NULL", True), ("NULL IS NOT NULL", False),
    ("5 BETWEEN 1 AND 10", True), ("5 BETWEEN 5 AND 5", True), ("5 NOT BETWEEN 1 AND 10", False),
    ("15 BETWEEN 1 AND 10", False), ("10 BETWEEN 10 AND 1", False), ("NULL BETWEEN 1 AND 2", None),
    ("5 BETWEEN NULL AND 10", None), ("15 BETWEEN NULL AND 10", False), ("5 NOT BETWEEN NULL AND 10", None),
    ("1.5 BETWEEN 1 AND 2", True), ("'b' BETWEEN 'a' AND 'c'", True),
    ("3 IN (1, 2, 3)", True), ("4 IN (1, 2, 3)", False), ("4 IN (1, NULL)", None), ("3 IN (3, NULL)", True),
    ("4 NOT IN (1, 2)", True), ("4 NOT IN (1, NULL)", None), ("3 NOT IN (3, NULL)", False), ("NULL IN (1)", None),
    ("1 IN (1.0)", True), ("'a' IN ('a', 'b')", True),
    ("'hello' LIKE 'h%o'", True), ("'hello' LIKE 'h_llo'", True), ("'hello' LIKE 'H%'", False),
    ("'hello' LIKE '%'", True), ("'' LIKE '%'", True), ("'' LIKE '_'", False), ("'hello' LIKE 'hell'", False),
    ("'hello' NOT LIKE 'x%'", True), ("NULL LIKE 'a'", None), ("'a' LIKE NULL", None), ("'50%' LIKE '50%'", True),
    ("abs(-5)", 5), ("abs(-2.5)", 2.5), ("abs(NULL)", None), ("length('héllo')", 5), ("length('')", 0),
    ("lower('AbC')", "abc"), ("upper('AbC')", "ABC"), ("upper(NULL)", None),
    ("round(2.5)", 3.0), ("round(-2.5)", -3.0), ("round(2.4)", 2.0), ("round(2.345, 2)", 2.35), ("round(7)", 7),
    ("round(NULL)", None), ("coalesce(NULL, NULL, 3)", 3), ("coalesce(NULL)", None), ("coalesce(1, 1/0)", 1),
    ("coalesce(NULL, 'x')", "x"),
])
def test_expression_values(db, expression, expected):
    same(one(db, f"SELECT {expression}"), expected)


@pytest.mark.parametrize("expression, message", [
    ("1 / 0", "division by zero"), ("1 % 0", "division by zero"), ("1.0 / 0", "division by zero"),
    ("1 / 0.0", "division by zero"),
    ("9223372036854775807 + 1", "integer overflow"), ("-9223372036854775807 - 2", "integer overflow"),
    ("9223372036854775807 * 2", "integer overflow"), ("-(-9223372036854775807 - 1)", "integer overflow"),
    ("(-9223372036854775807 - 1) / -1", "integer overflow"), ("abs(-9223372036854775807 - 1)", "integer overflow"),
    ("9223372036854775808", "integer overflow"),
    ("'a' + 1", "cannot apply '\\+' to TEXT and INT"), ("TRUE + 1", "cannot apply"), ("1 + 'a'", "cannot apply"),
    ("1 < 'a'", "cannot compare INT with TEXT"), ("TRUE = 1", "cannot compare BOOL with INT"),
    ("'a' BETWEEN 1 AND 2", "cannot compare"), ("1 IN ('a')", "cannot compare"),
    ("NOT 1", "NOT needs boolean operands, got INT"), ("1 AND TRUE", "AND needs boolean operands"),
    ("FALSE OR 'x'", "OR needs boolean operands, got TEXT"), ("-'a'", "unary minus needs a number"),
    ("'a' LIKE 1", "LIKE needs TEXT"), ("1 LIKE 'a'", "LIKE needs TEXT"),
    ("abs('x')", "abs\\(\\) needs a number"), ("length(5)", "length\\(\\) needs TEXT"), ("upper(1)", "needs TEXT"),
    ("nosuch(1)", "no such function: nosuch"), ("abs()", "takes 1 argument"), ("abs(1, 2)", "takes 1 argument"),
    ("round(1.5, 99)", "digits must be"), ("round(1.5, 'x')", "digits must be"), ("abs(*)", "takes 1 argument"),
    ("coalesce()", "at least one"), ("count(*) + a", "must appear inside an aggregate"), ("nosuch + 1", "no such column"),
])
def test_expression_errors(db, expression, message):
    with pytest.raises(SQLError, match=message):
        db.execute(f"SELECT {expression}")


def test_like_agrees_with_a_regex_reference():
    rng = random.Random(5)

    def reference(s, p):
        rx = "".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in p)
        return re.fullmatch(rx, s, re.DOTALL) is not None

    for _ in range(5000):
        s = "".join(rng.choice("ab") for _ in range(rng.randrange(0, 9)))
        p = "".join(rng.choice("ab%_") for _ in range(rng.randrange(0, 7)))
        assert like(s, p) == reference(s, p), (s, p)


def test_like_is_not_exponential():
    assert like("a" * 5000, "%a" * 40 + "b") is False             # a backtracking regex would take forever
    assert like("a" * 5000, "%" * 100 + "a") is True


def test_where_with_null_filters_the_row_out(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, a INT); INSERT INTO t VALUES (1, 1), (2, NULL), (3, 3)")
    assert rows(db, "SELECT id FROM t WHERE a > 0") == [(1,), (3,)]
    assert rows(db, "SELECT id FROM t WHERE NOT a > 0") == []     # NOT NULL is still NULL
    assert rows(db, "SELECT id FROM t WHERE a IS NULL") == [(2,)]
    assert rows(db, "SELECT id FROM t WHERE a = NULL") == []
    assert rows(db, "SELECT id FROM t WHERE a IN (1, NULL)") == [(1,)]
    assert rows(db, "SELECT id FROM t WHERE a NOT IN (1, NULL)") == []


def test_boolean_columns(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, ok BOOL); INSERT INTO t VALUES (1, TRUE), (2, FALSE), (3, NULL)")
    assert rows(db, "SELECT id FROM t WHERE ok") == [(1,)]
    assert rows(db, "SELECT id FROM t WHERE NOT ok") == [(2,)]
    assert rows(db, "SELECT id FROM t WHERE ok = FALSE") == [(2,)]
    assert rows(db, "SELECT id FROM t WHERE ok OR id = 3") == [(1,), (3,)]


# ---------- aggregates ----------

@pytest.fixture
def nums(db):
    script(db, """CREATE TABLE n (id INT PRIMARY KEY, v INT, s TEXT, f REAL);
                  INSERT INTO n VALUES (1, 10, 'b', 1.5), (2, 20, 'a', NULL), (3, NULL, 'c', 2.5), (4, 30, NULL, 3.5)""")
    return db


def test_aggregates(nums):
    r = nums.execute("SELECT count(*), count(v), count(s), sum(v), avg(v), min(v), max(v), min(s), max(s), "
                     "sum(f), avg(f), min(f), max(f) FROM n")
    assert r.rows == [(4, 3, 3, 60, 20.0, 10, 30, "a", "c", 7.5, 2.5, 1.5, 3.5)]
    same(r.rows[0][3], 60)
    same(r.rows[0][4], 20.0)


def test_aggregates_use_where_and_may_be_combined_in_expressions(nums):
    assert rows(nums, "SELECT count(*), sum(v) FROM n WHERE id > 1") == [(3, 50)]
    assert rows(nums, "SELECT max(v) - min(v), count(*) * 2 + 1, abs(min(v) - 100) FROM n") == [(20, 9, 90)]
    assert rows(nums, "SELECT sum(v) / count(v) AS mean FROM n") == [(20,)]
    assert nums.execute("SELECT count(*) AS how_many FROM n").columns == ("how_many",)
    assert rows(nums, "SELECT count(*), count(*) FROM n") == [(4, 4)]
    assert rows(nums, "SELECT sum(v + 1), sum(v) + 1 FROM n") == [(63, 61)]
    assert rows(nums, "SELECT max(v) FROM n WHERE id = 2") == [(20,)]
    assert rows(nums, "SELECT count(*) FROM n LIMIT 1 OFFSET 0") == [(4,)]


def test_aggregates_over_nothing(db):
    script(db, "CREATE TABLE e (id INT PRIMARY KEY, v INT)")
    assert rows(db, "SELECT count(*), count(v), sum(v), avg(v), min(v), max(v) FROM e") == [(0, 0, None, None, None, None)]
    assert one(db, "SELECT count(*)") == 1                         # no table: a single empty row
    assert one(db, "SELECT count(*) FROM e WHERE v > 0") == 0


def test_aggregate_limit_and_offset(nums):
    assert rows(nums, "SELECT count(*) FROM n LIMIT 0") == []
    assert rows(nums, "SELECT count(*) FROM n LIMIT 1 OFFSET 1") == []


@pytest.mark.parametrize("statement, message", [
    ("SELECT v, count(*) FROM n", "must appear inside an aggregate"),
    ("SELECT count(*), v + 1 FROM n", "must appear inside an aggregate"),
    ("SELECT * FROM n WHERE count(*) > 1", "not allowed in WHERE"),
    ("SELECT sum(s) FROM n", "sum\\(\\) needs a number"),
    ("SELECT avg(s) FROM n", "avg\\(\\) needs a number"),
    ("SELECT sum(count(*)) FROM n", "not allowed here"),
    ("SELECT count(v, v) FROM n", "takes exactly 1 argument"),
    ("SELECT count() FROM n", "takes exactly 1 argument"),
    ("SELECT count(nope) FROM n", "no such column"),
    ("SELECT count(*) FROM n ORDER BY id", "ORDER BY is not supported"),
    ("SELECT *, count(*) FROM n", "cannot mix"),
    ("SELECT sum(*) FROM n", "not valid"),
    ("SELECT min(v + s) FROM n", "cannot apply"),
])
def test_aggregate_errors(nums, statement, message):
    fails(nums, statement, message)


def test_sum_overflow_is_an_error(db):
    script(db, f"CREATE TABLE b (n INT); INSERT INTO b VALUES ({INT_MAX}), (1)")
    fails(db, "SELECT sum(n) FROM b", "integer overflow")
    assert one(db, "SELECT avg(n) FROM b") == (INT_MAX + 1) / 2


# ---------- ORDER BY, LIMIT, OFFSET ----------

@pytest.fixture
def people(db):
    script(db, """CREATE TABLE p (id INT PRIMARY KEY, name TEXT, age INT, city TEXT);
                  INSERT INTO p VALUES (1, 'ann', 30, 'x'), (2, 'bob', 25, 'y'), (3, 'cy', NULL, 'x'),
                                       (4, 'di', 25, 'x'), (5, NULL, 40, NULL), (6, 'ed', 30, 'y')""")
    return db


def ids(db, text):
    return [r[0] for r in db.execute(text).rows]


def test_order_by_basics(people):
    assert ids(people, "SELECT id FROM p ORDER BY age") == [3, 2, 4, 1, 6, 5]           # NULL first, ties by key
    assert ids(people, "SELECT id FROM p ORDER BY age DESC") == [5, 1, 6, 2, 4, 3]     # NULL last
    assert ids(people, "SELECT id FROM p ORDER BY name") == [5, 1, 2, 3, 4, 6]
    assert ids(people, "SELECT id FROM p ORDER BY id DESC") == [6, 5, 4, 3, 2, 1]
    assert ids(people, "SELECT id FROM p ORDER BY id ASC") == [1, 2, 3, 4, 5, 6]


def test_order_by_several_keys_with_mixed_directions(people):
    assert ids(people, "SELECT id FROM p ORDER BY age DESC, name") == [5, 1, 6, 2, 4, 3]
    assert ids(people, "SELECT id FROM p ORDER BY age, name DESC") == [3, 4, 2, 6, 1, 5]
    assert ids(people, "SELECT id FROM p ORDER BY city, age DESC, id DESC") == [5, 1, 4, 3, 6, 2]


def test_order_by_expressions_aliases_and_hidden_columns(people):
    assert ids(people, "SELECT id FROM p WHERE age IS NOT NULL ORDER BY age * -1, id") == [5, 1, 6, 2, 4]
    assert [r[0] for r in people.execute("SELECT age + 1 AS next FROM p WHERE id < 4 ORDER BY next DESC").rows] == [31, 26, None]
    assert ids(people, "SELECT id FROM p ORDER BY length(name) DESC, id") == [1, 2, 3, 4, 6, 5]
    assert ids(people, "SELECT age AS id FROM p ORDER BY id") == [None, 25, 25, 30, 30, 40]   # the alias wins
    assert [r[0] for r in people.execute("SELECT age AS id FROM p ORDER BY id DESC").rows] == [40, 30, 30, 25, 25, None]


def test_limit_and_offset(people):
    assert ids(people, "SELECT id FROM p LIMIT 2") == [1, 2]
    assert ids(people, "SELECT id FROM p LIMIT 2 OFFSET 2") == [3, 4]
    assert ids(people, "SELECT id FROM p LIMIT 0") == []
    assert ids(people, "SELECT id FROM p LIMIT 100 OFFSET 4") == [5, 6]
    assert ids(people, "SELECT id FROM p LIMIT 2 OFFSET 100") == []
    assert ids(people, "SELECT id FROM p ORDER BY age DESC LIMIT 2 OFFSET 1") == [1, 6]
    assert ids(people, "SELECT id FROM p ORDER BY id DESC LIMIT 2") == [6, 5]
    assert ids(people, "SELECT id FROM p WHERE id > 2 LIMIT 2 OFFSET 1") == [4, 5]


# ---------- the planner ----------

def test_plans(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v INT); CREATE TABLE h (a INT)")
    for where, plan in [
        ("id = 5", "(id = 5)"), ("5 = id", "(id = 5)"), ("id > 10", "(id >= 11)"), ("id >= 10", "(id >= 10)"),
        ("id < 10", "(id <= 9)"), ("id <= 10", "(id <= 10)"), ("10 > id", "(id <= 9)"), ("id BETWEEN 3 AND 9", "(id BETWEEN 3 AND 9)"),
        ("id >= 5 AND id < 9", "(id BETWEEN 5 AND 8)"), ("id > 5.5", "(id >= 6)"), ("id < 5.5", "(id <= 5)"),
        ("id >= 5.5", "(id >= 6)"), ("id <= 5.5", "(id <= 5)"), ("id = 5.0", "(id = 5)"), ("t.id = 4", "(id = 4)"),
        ("id > 1 AND id > 5 AND id < 20 AND id <= 12", "(id BETWEEN 6 AND 12)"), ("id = 3 AND v > 1", "(id = 3)"),
        ("id = 2 + 3", "(id = 5)"), ("id = -2", "(id = -2)"), ("id BETWEEN 1 + 1 AND 10 - 1", "(id BETWEEN 2 AND 9)"),
        ("(id = 5)", "(id = 5)"), ("id = 5 AND (id < 9 AND v = 1)", "(id = 5)"),
    ]:
        assert db.execute(f"SELECT * FROM t WHERE {where}").plan == f"SEARCH t USING PRIMARY KEY {plan}", where
    for where in ["id = 5.5", "id = NULL", "id BETWEEN 9 AND 3", "id = 3 AND id = 4", "id < 5 AND id > 8", "NULL < id",
                  "id > 9 AND id < 3"]:
        assert db.execute(f"SELECT * FROM t WHERE {where}").plan == "SEARCH t USING PRIMARY KEY (no matching range)", where
    for where in ["v = 3", "id = 5 OR id = 6", "id != 5", "NOT id < 5", "id + 1 = 5", "id = v", "id IN (1, 2)",
                  "id NOT BETWEEN 1 AND 5", "id = 'x'", "id = TRUE", "abs(id) = 5", "id * 1 = 5"]:
        assert db.execute(f"SELECT * FROM t WHERE {where}").plan == "SCAN t", where
    assert db.execute("SELECT * FROM t").plan == "SCAN t"
    assert db.execute("SELECT * FROM h WHERE a = 1").plan == "SCAN h"
    assert db.execute("SELECT 1").plan == "no table"
    assert db.execute("UPDATE t SET v = 1 WHERE id = 3").plan.endswith("(id = 3)")
    assert db.execute("DELETE FROM t WHERE id > 100").plan.endswith("(id >= 101)")


def test_type_errors_still_surface_with_the_planner(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY); INSERT INTO t VALUES (1)")
    fails(db, "SELECT * FROM t WHERE id = 'x'", "cannot compare INT with TEXT")
    fails(db, "SELECT * FROM t WHERE id = TRUE", "cannot compare INT with BOOL")
    fails(db, "SELECT * FROM t WHERE id = 1 AND id = 1 / 0", "division by zero")


@pytest.fixture
def big(tmp_path):
    d = Database(str(tmp_path / "big.db"), max_internal_keys=10)     # narrow nodes -> a taller tree
    d.execute("CREATE TABLE b (id INT PRIMARY KEY, v INT, pad TEXT)")
    d.execute("BEGIN")
    for i in range(0, 4000, 200):
        d.execute("INSERT INTO b VALUES " + ", ".join(f"({j}, {j % 7}, '{'p' * 40}')" for j in range(i, i + 200)))
    d.execute("COMMIT")
    yield d
    d.close()


def reads(db, text):
    before = db.tree.page_reads
    result = db.execute(text)
    return db.tree.page_reads - before, result


def test_index_use_really_reads_fewer_pages(big):
    height = big.tree.height()
    assert height >= 3
    full, r = reads(big, "SELECT count(*) FROM b WHERE v = 3")
    assert r.rows == [(571,)] and full > 50
    point, r = reads(big, "SELECT * FROM b WHERE id = 2500")
    assert r.rows[0][0] == 2500 and point <= height
    rng, r = reads(big, "SELECT * FROM b WHERE id BETWEEN 1000 AND 1009")
    assert len(r.rows) == 10 and rng <= height + 2
    half, r = reads(big, "SELECT count(*) FROM b WHERE id >= 3000")
    assert r.rows == [(1000,)] and full / 6 < half < full / 2 * 1.5
    miss, r = reads(big, "SELECT * FROM b WHERE id = 99999")
    assert r.rows == [] and miss <= height
    none, r = reads(big, "SELECT * FROM b WHERE id > 5 AND id < 3")
    assert none == 0


def test_limit_stops_reading_early(big):
    full, _ = reads(big, "SELECT * FROM b WHERE v = 99")
    few, r = reads(big, "SELECT * FROM b LIMIT 5")
    assert len(r.rows) == 5 and few <= big.tree.height() + 1 < full / 10
    few2, r = reads(big, "SELECT id FROM b WHERE id > 3000 LIMIT 3 OFFSET 2")
    assert r.rows == [(3003,), (3004,), (3005,)] and few2 <= big.tree.height() + 1
    sorted_all, _ = reads(big, "SELECT id FROM b ORDER BY v LIMIT 3")
    assert sorted_all > full / 2                                  # a real sort has to see every row
    asc, _ = reads(big, "SELECT id FROM b ORDER BY id LIMIT 3")
    assert asc <= big.tree.height() + 1


def test_updates_and_deletes_use_the_index_too(big):
    r = big.execute("UPDATE b SET v = -1 WHERE id = 1234")
    assert r.rowcount == 1 and big.tree.page_reads > 0
    cost, r = reads(big, "DELETE FROM b WHERE id BETWEEN 100 AND 104")
    assert r.rowcount == 5 and cost < 60
    assert one(big, "SELECT count(*) FROM b") == 3995
    big.tree.validate()


def test_planner_never_changes_the_answer(db):
    """Run random primary-key conditions twice: as written (index) and disguised as (id + 0) (full scan)."""
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v INT); BEGIN")
    db.execute("INSERT INTO t VALUES " + ", ".join(f"({i}, {i % 5})" for i in range(-30, 120)))
    db.execute("COMMIT")
    rng = random.Random(11)

    def const():
        return rng.choice([str(rng.randrange(-40, 130)), f"{rng.randrange(-40, 130)}.{rng.choice(['0', '5', '25'])}",
                           "NULL", f"{rng.randrange(1, 9)} * {rng.randrange(1, 9)}", f"-{rng.randrange(1, 40)}",
                           "9999999999999999", "-9999999999999999"])

    def condition(col):
        k = rng.randrange(7)
        if k == 0:
            return f"{col} {rng.choice(['=', '<', '<=', '>', '>=', '!='])} {const()}"
        if k == 1:
            return f"{const()} {rng.choice(['=', '<', '<=', '>', '>='])} {col}"
        if k == 2:
            return f"{col} BETWEEN {const()} AND {const()}"
        if k == 3:
            return f"{col} NOT BETWEEN {const()} AND {const()}"
        if k == 4:
            return f"v {rng.choice(['=', '<', '>'])} {rng.randrange(0, 5)}"
        if k == 5:
            return f"{col} IN ({const()}, {const()})"
        return f"({col} < {const()} OR {col} > {const()})"

    for _ in range(600):
        parts = [rng.randrange(10 ** 9) for _ in range(rng.randrange(1, 4))]
        state = rng.getstate()
        indexed = " AND ".join(condition("id") for _ in parts)
        rng.setstate(state)
        scanned = " AND ".join(condition("(id + 0)") for _ in parts)
        a, b = db.execute(f"SELECT id FROM t WHERE {indexed}"), db.execute(f"SELECT id FROM t WHERE {scanned}")
        assert a.rows == b.rows, indexed
        assert b.plan == "SCAN t"


# ---------- UPDATE and DELETE ----------

def test_update_basics(people):
    r = people.execute("UPDATE p SET age = age + 1 WHERE city = 'x'")
    assert r.rowcount == 3 and r.message == "UPDATE 3"
    assert rows(people, "SELECT id, age FROM p WHERE city = 'x'") == [(1, 31), (3, None), (4, 26)]
    assert people.execute("UPDATE p SET name = upper(name)").rowcount == 6
    assert people.execute("UPDATE p SET age = 0 WHERE id = 999").rowcount == 0
    assert rows(people, "SELECT name FROM p WHERE id <= 2") == [("ANN",), ("BOB",)]


def test_update_right_hand_sides_see_the_old_row(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, a INT, b INT); INSERT INTO t VALUES (1, 10, 20)")
    db.execute("UPDATE t SET a = b, b = a")
    assert rows(db, "SELECT a, b FROM t") == [(20, 10)]


def test_update_can_change_the_primary_key(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b'), (3, 'c')")
    db.execute("UPDATE t SET id = id + 1")                        # keys 1,2,3 -> 2,3,4 : moves onto each other
    assert rows(db, "SELECT * FROM t") == [(2, "a"), (3, "b"), (4, "c")]
    db.execute("UPDATE t SET id = 10 WHERE id = 2")
    assert rows(db, "SELECT id FROM t") == [(3,), (4,), (10,)]
    fails(db, "UPDATE t SET id = 4 WHERE id = 3", "UNIQUE")
    fails(db, "UPDATE t SET id = 7", "UNIQUE")                    # all rows would get the same key
    db.execute("UPDATE t SET id = 3 - id + 4 WHERE id <= 4")      # swap 3 and 4
    assert rows(db, "SELECT * FROM t") == [(3, "c"), (4, "b"), (10, "a")]
    assert db.table("t").next_rowid == 11
    db.execute("INSERT INTO t (v) VALUES ('new')")
    assert rows(db, "SELECT id FROM t WHERE v = 'new'") == [(11,)]
    db.tree.validate()


def test_delete_basics(people):
    r = people.execute("DELETE FROM p WHERE age = 25")
    assert r.rowcount == 2 and r.message == "DELETE 2"
    assert ids(people, "SELECT id FROM p") == [1, 3, 5, 6]
    assert people.execute("DELETE FROM p WHERE id = 999").rowcount == 0
    assert people.execute("DELETE FROM p").rowcount == 4
    assert rows(people, "SELECT * FROM p") == []
    people.tree.validate()


def test_tables_without_a_primary_key(db):
    script(db, "CREATE TABLE u (x INT, y TEXT); INSERT INTO u VALUES (1, 'a'), (1, 'a'), (2, 'b')")
    assert rows(db, "SELECT * FROM u") == [(1, "a"), (1, "a"), (2, "b")]      # duplicates are fine
    assert db.execute("UPDATE u SET y = 'z' WHERE x = 1").rowcount == 2
    assert db.execute("DELETE FROM u WHERE y = 'z'").rowcount == 2
    db.execute("INSERT INTO u VALUES (3, 'c')")
    assert rows(db, "SELECT * FROM u") == [(2, "b"), (3, "c")]
    assert db.table("u").next_rowid == 5                           # row ids are never reused


# ---------- several tables and the catalog ----------

def test_tables_do_not_see_each_other(db):
    script(db, """CREATE TABLE a (id INT PRIMARY KEY, v TEXT); CREATE TABLE b (id INT PRIMARY KEY, v TEXT);
                  CREATE TABLE c (id INT PRIMARY KEY, v TEXT);
                  INSERT INTO a VALUES (1, 'a1'), (2, 'a2'); INSERT INTO b VALUES (1, 'b1'), (3, 'b3');
                  INSERT INTO c VALUES (-5, 'c-5'), (0, 'c0')""")
    assert rows(db, "SELECT v FROM a") == [("a1",), ("a2",)]
    assert rows(db, "SELECT v FROM b WHERE id >= 1") == [("b1",), ("b3",)]
    assert rows(db, "SELECT v FROM c WHERE id < 100") == [("c-5",), ("c0",)]
    assert rows(db, "SELECT v FROM c WHERE id > -100 ORDER BY id DESC") == [("c0",), ("c-5",)]
    db.execute("DELETE FROM a")
    assert one(db, "SELECT count(*) FROM b") == 2 and one(db, "SELECT count(*) FROM c") == 2


def test_drop_table_removes_exactly_its_rows(db):
    script(db, "CREATE TABLE a (id INT PRIMARY KEY); CREATE TABLE b (id INT PRIMARY KEY); CREATE TABLE c (x INT)")
    db.execute("INSERT INTO a VALUES " + ", ".join(f"({i})" for i in range(300)))
    db.execute("INSERT INTO b VALUES " + ", ".join(f"({i})" for i in range(50)))
    db.execute("INSERT INTO c VALUES (1), (2)")
    before = db.tree.count()
    r = db.execute("DROP TABLE a")
    assert r.rowcount == 300 and db.tree.count() == before - 301   # its rows and its catalog entry
    assert sorted(db.tables) == ["b", "c"] and one(db, "SELECT count(*) FROM b") == 50
    db.execute("CREATE TABLE a (id INT PRIMARY KEY, w TEXT)")      # a fresh table is empty, not haunted
    assert rows(db, "SELECT * FROM a") == []
    db.execute("INSERT INTO a VALUES (7, 'new')")
    assert rows(db, "SELECT * FROM a") == [(7, "new")]
    db.tree.validate()


def test_catalog_survives_reopen_with_automatic_keys_intact(tmp_path):
    path = str(tmp_path / "c.db")
    with Database(path) as d:
        script(d, "CREATE TABLE a (id INT PRIMARY KEY, v INT); CREATE TABLE b (x INT)")
        d.execute("INSERT INTO a (v) VALUES (1), (2), (3)")
        d.execute("INSERT INTO b VALUES (1), (2)")
        d.execute("DROP TABLE b")
    with Database(path) as d:
        assert list(d.tables) == ["a"] and d.tables["a"].next_rowid == 4
        d.execute("CREATE TABLE z (q INT)")
        assert d.tables["z"].id == 2                                # the freed id is reused, cleanly
        d.execute("INSERT INTO a (v) VALUES (9)")
        assert rows(d, "SELECT id FROM a ORDER BY id DESC LIMIT 1") == [(4,)]


# ---------- transactions ----------

def test_begin_commit_and_rollback(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT); INSERT INTO t VALUES (1, 'keep')")
    db.execute("BEGIN")
    assert db.in_transaction
    db.execute("INSERT INTO t VALUES (2, 'x')")
    db.execute("UPDATE t SET v = 'changed' WHERE id = 1")
    assert rows(db, "SELECT * FROM t") == [(1, "changed"), (2, "x")]      # you see your own changes
    db.execute("ROLLBACK")
    assert not db.in_transaction
    assert rows(db, "SELECT * FROM t") == [(1, "keep")]
    db.execute("BEGIN")
    db.execute("INSERT INTO t VALUES (2, 'x')")
    db.execute("COMMIT")
    assert rows(db, "SELECT * FROM t") == [(1, "keep"), (2, "x")]
    db.tree.validate()


def test_rollback_undoes_schema_changes_too(db):
    script(db, "CREATE TABLE keep (a INT); INSERT INTO keep VALUES (1)")
    script(db, "BEGIN; CREATE TABLE gone (b INT); INSERT INTO gone VALUES (2); DROP TABLE keep")
    assert sorted(db.tables) == ["gone"]
    db.execute("ROLLBACK")
    assert sorted(db.tables) == ["keep"] and rows(db, "SELECT * FROM keep") == [(1,)]
    fails(db, "SELECT * FROM gone", "no such table")
    db.execute("CREATE TABLE gone (b INT)")                        # the same id is usable again
    assert rows(db, "SELECT * FROM gone") == []
    db.tree.validate()


def test_rollback_restores_the_key_counter(db):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
    script(db, "BEGIN; INSERT INTO t (v) VALUES (1), (2), (3); ROLLBACK")
    db.execute("INSERT INTO t (v) VALUES (9)")
    assert rows(db, "SELECT id FROM t") == [(1,)]


def test_transaction_misuse(db):
    fails(db, "COMMIT", "no transaction is open")
    fails(db, "ROLLBACK", "no transaction is open")
    db.execute("BEGIN")
    fails(db, "BEGIN", "already open")
    db.execute("COMMIT")


def test_close_with_an_open_transaction_rolls_it_back(tmp_path):
    path = str(tmp_path / "x.db")
    d = Database(path)
    script(d, "CREATE TABLE t (id INT PRIMARY KEY); INSERT INTO t VALUES (1); BEGIN; INSERT INTO t VALUES (2)")
    d.close()
    with Database(path) as d2:
        assert rows(d2, "SELECT id FROM t") == [(1,)]
        d2.tree.validate()


def test_unexpected_failure_inside_begin_aborts_the_whole_transaction(db, monkeypatch):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v INT); INSERT INTO t VALUES (1, 1); BEGIN; INSERT INTO t VALUES (2, 2)")
    real = db.tree.insert

    def boom(*a):
        raise RuntimeError("disk on fire")
    monkeypatch.setattr(db.tree, "insert", boom)
    with pytest.raises(TransactionAborted, match="disk on fire"):
        db.execute("INSERT INTO t VALUES (3, 3)")
    monkeypatch.setattr(db.tree, "insert", real)
    assert not db.in_transaction
    assert rows(db, "SELECT id FROM t") == [(1,)]
    db.tree.validate()


def test_unexpected_failure_in_autocommit_undoes_the_statement(db, monkeypatch):
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
    real = db.tree.insert
    calls = []

    def flaky(key, value):
        calls.append(key)
        if len(calls) == 2:
            raise RuntimeError("write failed")
        return real(key, value)
    monkeypatch.setattr(db.tree, "insert", flaky)
    with pytest.raises(RuntimeError):
        db.execute("INSERT INTO t VALUES (1, 1), (2, 2), (3, 3)")
    monkeypatch.setattr(db.tree, "insert", real)
    assert rows(db, "SELECT * FROM t") == [] and db.tables["t"].next_rowid == 1
    db.tree.validate()


def test_failure_after_the_in_memory_key_counter_moved_is_undone(db, monkeypatch):
    """The key counter is bumped in memory before its catalog row is saved; if saving fails, forget the bump."""
    script(db, "CREATE TABLE t (id INT PRIMARY KEY, v INT)")
    real = db.tree.put

    def broken(key, value):
        raise RuntimeError("cannot save the catalog row")
    monkeypatch.setattr(db.tree, "put", broken)
    with pytest.raises(RuntimeError):
        db.execute("INSERT INTO t (v) VALUES (1), (2)")
    monkeypatch.setattr(db.tree, "put", real)
    assert db.tables["t"].next_rowid == 1 and rows(db, "SELECT * FROM t") == []
    db.execute("INSERT INTO t (v) VALUES (7)")
    assert rows(db, "SELECT * FROM t") == [(1, 7)]
    db.tree.validate()


def test_a_whole_script_in_one_transaction_costs_one_fsync(tmp_path):
    events = []
    d = Database(str(tmp_path / "f.db"), pager_options=dict(sync=False, fault_hook=events.append))
    d.execute("CREATE TABLE t (id INT PRIMARY KEY, v INT)")
    events.clear()
    d.execute("BEGIN")
    for i in range(50):
        d.execute(f"INSERT INTO t VALUES ({i}, {i})")
    d.execute("COMMIT")
    assert events.count("wal_synced") == 1
    events.clear()
    for i in range(50, 60):
        d.execute(f"INSERT INTO t VALUES ({i}, {i})")
    assert events.count("wal_synced") == 10
    d.close()


# ---------- crash safety ----------

class Crash(BaseException):
    """A simulated power cut."""


def snapshot(path, dst_dir):
    dst = str(dst_dir / "t.db")
    shutil.copyfile(path, dst)
    if os.path.exists(path + "-wal"):
        shutil.copyfile(path + "-wal", dst + "-wal")
    return dst


def abandon(d):
    """Free a 'crashed' database's file descriptors without any cleanup."""
    for close in (lambda: os.close(d.tree.pager._wal), d.tree.pager._f.close):
        try:
            close()
        except (OSError, ValueError):
            pass


def state(d):
    return {name: d.execute(f"SELECT * FROM {name}").rows for name in sorted(d.tables)}


SETUP = ["CREATE TABLE a (id INT PRIMARY KEY, v INT)", "CREATE TABLE b (id INT PRIMARY KEY, w TEXT)",
         "INSERT INTO a VALUES (1, 10), (2, 20), (3, 30)", "INSERT INTO b VALUES (1, 'x'), (2, 'y')"]
CHANGES = ["INSERT INTO a VALUES (4, 40), (5, 50)", "UPDATE b SET w = upper(w)", "DELETE FROM a WHERE id = 1",
           "CREATE TABLE c (x INT, y TEXT)", "INSERT INTO c VALUES (1, 'one'), (2, 'two')",
           "UPDATE a SET id = id + 10 WHERE id >= 4", "DROP TABLE b"]


def run_with_crash(path, crash_at, body, checkpoint_bytes):
    """Set up the database, then run body(db) and raise Crash at the crash_at-th I/O event (0 = never)."""
    count = [0]
    armed = [False]

    def hook(event):
        if armed[0]:
            count[0] += 1
            if count[0] == crash_at:
                raise Crash(event)

    d = Database(path, pager_options=dict(sync=False, checkpoint_bytes=checkpoint_bytes, fault_hook=hook))
    for s in SETUP:
        d.execute(s)
    armed[0] = True
    try:
        body(d)
    except Crash:
        return count[0], d, True
    return count[0], d, False


@pytest.mark.parametrize("checkpoint_bytes", [3 * FRAME_SIZE, 10 ** 9])
def test_a_transaction_across_tables_is_all_or_nothing_at_every_crash_point(tmp_path, checkpoint_bytes):
    def body(d):
        d.execute("BEGIN")
        for s in CHANGES:
            d.execute(s)
        d.execute("COMMIT")

    (tmp_path / "dry").mkdir()
    total, dry, _ = run_with_crash(str(tmp_path / "dry" / "t.db"), 0, body, checkpoint_bytes)
    after = state(dry)
    dry.close()
    ref = Database(str(tmp_path / "ref.db"))
    for s in SETUP:
        ref.execute(s)
    before = state(ref)
    ref.close()
    assert total >= 3 and before != after
    seen = set()
    for crash_at in range(1, total + 1):
        d_dir = tmp_path / f"r{crash_at}"
        d_dir.mkdir()
        n, crashed, did_crash = run_with_crash(str(d_dir / "t.db"), crash_at, body, checkpoint_bytes)
        assert did_crash and n == crash_at
        s_dir = tmp_path / f"s{crash_at}"
        s_dir.mkdir()
        path = snapshot(str(d_dir / "t.db"), s_dir)
        abandon(crashed)
        with Database(path) as r:
            r.tree.validate()
            got = state(r)
            assert got == before or got == after, f"crash at event {crash_at}: half-applied transaction"
            seen.add("after" if got == after else "before")
            r.execute("INSERT INTO a (v) VALUES (99)") if "a" in r.tables else None
        shutil.rmtree(d_dir)
        shutil.rmtree(s_dir)
    assert seen == {"before", "after"}


@pytest.mark.parametrize("checkpoint_bytes", [3 * FRAME_SIZE, 10 ** 9])
def test_each_autocommitted_statement_is_all_or_nothing_at_every_crash_point(tmp_path, checkpoint_bytes):
    ref = Database(str(tmp_path / "ref.db"))
    for s in SETUP:
        ref.execute(s)
    states = [state(ref)]
    for s in CHANGES:
        ref.execute(s)
        states.append(state(ref))
    ref.close()

    progress = [0]

    def body(d):
        for k, s in enumerate(CHANGES):
            progress[0] = k
            d.execute(s)

    (tmp_path / "dry").mkdir()
    total, dry, _ = run_with_crash(str(tmp_path / "dry" / "t.db"), 0, body, checkpoint_bytes)
    dry.close()
    for crash_at in range(1, total + 1):
        d_dir = tmp_path / f"r{crash_at}"
        d_dir.mkdir()
        n, crashed, did_crash = run_with_crash(str(d_dir / "t.db"), crash_at, body, checkpoint_bytes)
        s_dir = tmp_path / f"s{crash_at}"
        s_dir.mkdir()
        path = snapshot(str(d_dir / "t.db"), s_dir)
        abandon(crashed)
        with Database(path) as r:
            r.tree.validate()
            got = state(r)
            k = progress[0]                                        # the statement that was interrupted
            assert got in (states[k], states[k + 1]), f"crash at event {crash_at} (statement {k})"
        shutil.rmtree(d_dir)
        shutil.rmtree(s_dir)


def test_committed_work_is_in_the_files_without_closing(tmp_path):
    d = Database(str(tmp_path / "live.db"))
    script(d, "CREATE TABLE t (id INT PRIMARY KEY, v TEXT); INSERT INTO t VALUES (1, 'a'), (2, 'b'); BEGIN; "
              "INSERT INTO t VALUES (3, 'uncommitted')")
    (tmp_path / "snap").mkdir()
    with Database(snapshot(str(tmp_path / "live.db"), tmp_path / "snap")) as r:
        assert rows(r, "SELECT * FROM t") == [(1, "a"), (2, "b")]
    d.close()


# ---------- randomized comparison against a plain Python model ----------

def tri_and(a, b):
    return False if a is False or b is False else (None if a is None or b is None else True)


def tri_or(a, b):
    return True if a is True or b is True else (None if a is None or b is None else False)


def tri_not(a):
    return None if a is None else not a


def like_py(s, p):
    return re.fullmatch("".join(".*" if c == "%" else "." if c == "_" else re.escape(c) for c in p), s, re.DOTALL) is not None


def gen_pred(rng, depth=2):
    """A random WHERE condition over rows (id, a, s): returns (sql text, python function)."""
    if depth > 0 and rng.random() < 0.55:
        kind = rng.choice(["and", "or", "not"])
        if kind == "not":
            t, f = gen_pred(rng, depth - 1)
            return f"NOT ({t})", lambda r, f=f: tri_not(f(r))
        (t1, f1), (t2, f2) = gen_pred(rng, depth - 1), gen_pred(rng, depth - 1)
        if kind == "and":
            return f"({t1}) AND ({t2})", lambda r: tri_and(f1(r), f2(r))
        return f"({t1}) OR ({t2})", lambda r: tri_or(f1(r), f2(r))
    k, lo = rng.randrange(0, 60), rng.randrange(0, 60)
    hi = lo + rng.randrange(0, 25)
    c = rng.choice(["bt", "lt", "ge", "eq", "agt", "aeq", "anull", "anotnull", "like", "seq", "ain", "idin"])
    if c == "bt":
        return f"id BETWEEN {lo} AND {hi}", lambda r: lo <= r[0] <= hi
    if c == "lt":
        return f"id < {k}", lambda r: r[0] < k
    if c == "ge":
        return f"{k} <= id", lambda r: r[0] >= k
    if c == "eq":
        return f"id = {k}", lambda r: r[0] == k
    if c == "agt":
        return f"a > {k % 10}", lambda r: None if r[1] is None else r[1] > k % 10
    if c == "aeq":
        return f"a = {k % 10}", lambda r: None if r[1] is None else r[1] == k % 10
    if c == "anull":
        return "a IS NULL", lambda r: r[1] is None
    if c == "anotnull":
        return "a IS NOT NULL", lambda r: r[1] is not None
    if c == "like":
        p = rng.choice(["a%", "%b", "_c%", "%", "x_z"])
        return f"s LIKE '{p}'", lambda r: None if r[2] is None else like_py(r[2], p)
    if c == "seq":
        v = rng.choice(["abc", "xyz", "b", ""])
        return f"s = '{v}'", lambda r: None if r[2] is None else r[2] == v
    if c == "ain":
        return "a IN (1, 3, NULL)", lambda r: None if r[1] is None else (True if r[1] in (1, 3) else None)
    return f"id IN ({k}, {k + 1})", lambda r: r[0] in (k, k + 1)


STRINGS = ["abc", "xyz", "b", "", "abcb", "xcz", "ac", None]


def test_random_operations_agree_with_a_python_model(tmp_path):
    rng = random.Random(2024)
    path = str(tmp_path / "m.db")
    d = Database(path, leaf_capacity=512, max_internal_keys=5)    # small nodes: lots of splits and merges
    d.execute("CREATE TABLE t (id INT PRIMARY KEY, a INT, s TEXT)")
    d.execute("CREATE TABLE u (x INT, y TEXT)")
    model = {}                       # id -> (a, s)
    umodel = []                      # rows (x, y) in insertion order
    saved = None
    in_txn = False
    nops = 0

    def lit(v):
        return "NULL" if v is None else (f"'{v}'" if isinstance(v, str) else str(v))

    def check():
        want = [(k,) + model[k] for k in sorted(model)]
        assert d.execute("SELECT * FROM t").rows == want
        assert d.execute("SELECT * FROM u").rows == umodel
        assert d.execute("SELECT id FROM t ORDER BY id DESC").rows == [(r[0],) for r in reversed(want)]

    for step in range(900):
        op = rng.choice(["insert", "insert", "insert_many", "update_a", "update_s", "update_id", "delete", "select",
                         "select", "u_insert", "u_update", "u_delete", "txn", "reopen"])
        nops += 1
        if op == "insert":
            k, a, s = rng.randrange(0, 70), rng.choice([None, 0, 1, 2, 3, 4, 5, 9]), rng.choice(STRINGS)
            sql = f"INSERT INTO t VALUES ({k}, {lit(a)}, {lit(s)})"
            if k in model:
                fails(d, sql, "UNIQUE")
            else:
                d.execute(sql)
                model[k] = (a, s)
        elif op == "insert_many":
            ks = [rng.randrange(0, 70) for _ in range(rng.randrange(1, 6))]
            vals = [(k, rng.choice([None, 1, 2, 7]), rng.choice(STRINGS)) for k in ks]
            sql = "INSERT INTO t VALUES " + ", ".join(f"({k}, {lit(a)}, {lit(s)})" for k, a, s in vals)
            if any(k in model for k in ks) or len(set(ks)) != len(ks):
                fails(d, sql, "UNIQUE")                           # all or nothing
            else:
                d.execute(sql)
                for k, a, s in vals:
                    model[k] = (a, s)
        elif op in ("update_a", "update_s"):
            text, f = gen_pred(rng)
            if op == "update_a":
                r = d.execute(f"UPDATE t SET a = a + 1 WHERE {text}")
                hit = [k for k in model if f((k,) + model[k]) is True]
                for k in hit:
                    a, s = model[k]
                    model[k] = (None if a is None else a + 1, s)
            else:
                r = d.execute(f"UPDATE t SET s = 'abc' WHERE {text}")
                hit = [k for k in model if f((k,) + model[k]) is True]
                for k in hit:
                    model[k] = (model[k][0], "abc")
            assert r.rowcount == len(hit)
        elif op == "update_id":
            lo = rng.randrange(0, 70)
            hi = lo + rng.randrange(0, 8)
            delta = rng.choice([1, -1, 100, 3])
            hit = [k for k in model if lo <= k <= hi]
            new = {k: k + delta for k in hit}
            moved = {new[k]: model[k] for k in hit}
            clash = len(set(new.values())) != len(new) or any(n in model and n not in new for n in new.values())
            sql = f"UPDATE t SET id = id + {delta} WHERE id BETWEEN {lo} AND {hi}"
            if clash:
                fails(d, sql, "UNIQUE")
            else:
                assert d.execute(sql).rowcount == len(hit)
                for k in hit:
                    del model[k]
                model.update(moved)
        elif op == "delete":
            text, f = gen_pred(rng)
            hit = [k for k in model if f((k,) + model[k]) is True]
            assert d.execute(f"DELETE FROM t WHERE {text}").rowcount == len(hit)
            for k in hit:
                del model[k]
        elif op == "select":
            text, f = gen_pred(rng)
            want = [(k,) + model[k] for k in sorted(model) if f((k,) + model[k]) is True]
            order = rng.choice(["", " ORDER BY id", " ORDER BY id DESC", " ORDER BY a, id", " ORDER BY s DESC, id DESC"])
            got = d.execute(f"SELECT * FROM t WHERE {text}{order}").rows
            if order == " ORDER BY id DESC":
                want.reverse()
            elif order == " ORDER BY a, id":
                want.sort(key=lambda r: (r[1] is not None, r[1] if r[1] is not None else 0, r[0]))
            elif order == " ORDER BY s DESC, id DESC":
                want.sort(key=lambda r: (r[2] is not None, r[2] or "", r[0]), reverse=True)
            assert got == want, (text, order)
            lim = rng.randrange(0, 5)
            off = rng.randrange(0, 4)
            assert d.execute(f"SELECT id FROM t WHERE {text}{order} LIMIT {lim} OFFSET {off}").rows == \
                [(r[0],) for r in want[off:off + lim]]
            assert one(d, f"SELECT count(*) FROM t WHERE {text}") == len(want)
        elif op == "u_insert":
            x, y = rng.choice([None, 1, 2, 3]), rng.choice(STRINGS)
            d.execute(f"INSERT INTO u VALUES ({lit(x)}, {lit(y)})")
            umodel.append((x, y))
        elif op == "u_update":
            k = rng.choice([1, 2, 3])
            d.execute(f"UPDATE u SET y = 'upd' WHERE x = {k}")
            umodel = [(x, "upd" if x == k else y) for x, y in umodel]
        elif op == "u_delete":
            k = rng.choice([1, 2, 3])
            d.execute(f"DELETE FROM u WHERE x = {k}")
            umodel = [(x, y) for x, y in umodel if x != k]
        elif op == "txn":
            if not in_txn:
                d.execute("BEGIN")
                in_txn, saved = True, (dict(model), list(umodel))
            elif rng.random() < 0.5:
                d.execute("COMMIT")
                in_txn = False
            else:
                d.execute("ROLLBACK")
                in_txn = False
                model, umodel = dict(saved[0]), list(saved[1])
        elif op == "reopen":
            d.close()                                             # closing with an open transaction rolls it back
            if in_txn:
                in_txn = False
                model, umodel = dict(saved[0]), list(saved[1])
            d = Database(path, leaf_capacity=512, max_internal_keys=5)
        check()
        if nops % 50 == 0:
            d.tree.validate()
    if in_txn:
        d.execute("COMMIT")
    check()
    d.tree.validate()
    d.close()


# ---------- hostile input ----------

def test_garbage_statements_never_corrupt_or_crash_the_database(tmp_path):
    d = Database(str(tmp_path / "g.db"))
    script(d, "CREATE TABLE t (id INT PRIMARY KEY, a INT, s TEXT); CREATE TABLE u (x INT);"
              "INSERT INTO t VALUES (1, 1, 'a'), (2, NULL, 'b'), (3, 3, NULL); INSERT INTO u VALUES (1), (2)")
    good = rows(d, "SELECT * FROM t")
    seeds = ["SELECT * FROM t WHERE id = 1", "SELECT a + id, upper(s) FROM t ORDER BY a DESC LIMIT 2",
             "SELECT count(*), sum(a) FROM t WHERE a IS NOT NULL", "UPDATE t SET a = a * 2 WHERE id > 1",
             "INSERT INTO t VALUES (9, 9, 'z')", "DELETE FROM t WHERE s LIKE 'z%'",
             "SELECT * FROM t WHERE id BETWEEN 1 AND 2 AND a IN (1, 2, NULL)", "CREATE TABLE q (a INT PRIMARY KEY)",
             "DROP TABLE q", "BEGIN", "ROLLBACK", "SELECT -a, NOT (a = 1) FROM t", "SELECT coalesce(s, 'n') FROM t"]
    alphabet = list("abct 0129'\"();,.*=<>!+-/%_\n") + ["SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "NULL", "id", "a",
                                                          "s", "t", "u", "count(*)", "1/0", "''", "TRUE", "LIKE", "IN", "9999999999999999999"]
    rng = random.Random(3)
    for _ in range(3000):
        text = list(rng.choice(seeds))
        for _ in range(rng.randint(0, 3)):
            i = rng.randrange(len(text) + 1)
            if text and rng.random() < 0.4:
                del text[min(i, len(text) - 1)]
            else:
                text.insert(i, rng.choice(alphabet))
        try:
            d.execute_script("".join(text))
        except (S.SQLSyntaxError, SQLError):
            pass
        if d.in_transaction:
            d.execute("ROLLBACK")
    d.tree.validate()
    for name in d.tables:
        d.execute(f"SELECT * FROM {name}")                        # every surviving table is still readable
    assert d.tables["t"] is not None and good is not None
    d.close()
