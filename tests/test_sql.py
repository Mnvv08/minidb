import operator
import random

import pytest

from sql import (Assignment, Begin, Between, Binary, Column, ColumnDef, Commit, CreateTable, Delete,
                 DropTable, Func, InList, Insert, IsNull, Like, Literal, OrderItem, Rollback, Select,
                 SelectItem, SQLSyntaxError, Star, Unary, Update, dump, parse, parse_script, tokenize)


def kinds(sql):
    return [(t.kind, t.value) for t in tokenize(sql)][:-1]


def where(sql):
    return parse(f"SELECT * FROM t WHERE {sql}").where


def expr(sql):
    return parse(f"SELECT {sql}").items[0].expr


def col(name):
    return Column(name)


def lit(v):
    return Literal(v)


# ---------- tokenizer ----------

def test_tokens_basic():
    assert kinds("select Foo, 12 3.5 'hi' <> <= >=") == [
        ("KW", "SELECT"), ("IDENT", "foo"), ("PUNCT", ","), ("INT", 12), ("FLOAT", 3.5),
        ("STRING", "hi"), ("OP", "!="), ("OP", "<="), ("OP", ">=")]


def test_keywords_are_case_insensitive_and_names_are_lowercased():
    assert kinds("SeLeCt NaMe") == [("KW", "SELECT"), ("IDENT", "name")]


def test_quoted_names_keep_case_and_may_be_keywords():
    assert kinds('"Select" "a ""b"""') == [("IDENT", "Select"), ("IDENT", 'a "b"')]


def test_string_escapes():
    assert kinds("'it''s' '' ''''") == [("STRING", "it's"), ("STRING", ""), ("STRING", "'")]


def test_numbers():
    assert kinds("0 007 1.25 2e3 1E-2 4.5e+1") == [
        ("INT", 0), ("INT", 7), ("FLOAT", 1.25), ("FLOAT", 2000.0), ("FLOAT", 0.01), ("FLOAT", 45.0)]


def test_comments_and_whitespace_are_skipped():
    assert kinds("1 -- ignore me\n 2 --") == [("INT", 1), ("INT", 2)]


def test_minus_minus_is_a_comment_but_spaced_minus_is_not():
    assert kinds("1 - -2") == [("INT", 1), ("OP", "-"), ("OP", "-"), ("INT", 2)]
    assert kinds("1--2") == [("INT", 1)]


@pytest.mark.parametrize("text, message", [
    ("'abc", "unterminated string"),
    ('"abc', "unterminated quoted name"),
    ('""', "cannot be empty"),
    ("12abc", "malformed number"),
    ("1.", "malformed number"),
    ("1e", "malformed number"),
    ("1.5.2", "malformed number"),
    ("1e999", "out of range"),
    ("9" * 5000, "too large"),
    ("a @ b", "unexpected character '@'"),
    ("a ! b", "unexpected character '!'"),
    ("é", "unexpected character"),
])
def test_tokenizer_errors(text, message):
    with pytest.raises(SQLSyntaxError, match=message):
        tokenize(text)


def test_error_position_is_line_and_column():
    with pytest.raises(SQLSyntaxError) as e:
        tokenize("SELECT 1,\n  2,\n   @")
    assert (e.value.line, e.value.col) == (3, 4)
    assert "line 3, column 4" in str(e.value)
    assert str(e.value).splitlines()[-1].rstrip() == "     ^"      # caret under the offending character


# ---------- statements ----------

def test_create_table():
    s = parse("CREATE TABLE users (id INT PRIMARY KEY, name TEXT NOT NULL, score float, ok Boolean NULL)")
    assert s == CreateTable("users", (
        ColumnDef("id", "INT", True, True), ColumnDef("name", "TEXT", False, True),
        ColumnDef("score", "REAL"), ColumnDef("ok", "BOOL")), False)
    assert parse("create table if not exists t (a int)").if_not_exists is True


def test_drop_table():
    assert parse("DROP TABLE t") == DropTable("t", False)
    assert parse("DROP TABLE IF EXISTS t") == DropTable("t", True)


def test_insert():
    assert parse("INSERT INTO t VALUES (1, 'a', NULL)") == Insert("t", None, ((lit(1), lit("a"), lit(None)),))
    s = parse("INSERT INTO t (a, b) VALUES (1, -2), (3, 4 + 5)")
    assert s.columns == ("a", "b")
    assert s.rows == ((lit(1), lit(-2)), (lit(3), Binary("+", lit(4), lit(5))))


def test_select_everything():
    s = parse("SELECT a, b + 1 AS c, * FROM t WHERE a > 1 ORDER BY a DESC, b ASC, c LIMIT 10 OFFSET 20")
    assert s.table == "t"
    assert s.items == (SelectItem(col("a")), SelectItem(Binary("+", col("b"), lit(1)), "c"), SelectItem(Star()))
    assert s.where == Binary(">", col("a"), lit(1))
    assert s.order_by == (OrderItem(col("a"), True), OrderItem(col("b"), False), OrderItem(col("c"), False))
    assert (s.limit, s.offset) == (10, 20)


def test_select_without_from():
    assert parse("SELECT 1 + 1") == Select((SelectItem(Binary("+", lit(1), lit(1))),))


def test_update_and_delete():
    u = parse("UPDATE t SET a = a + 1, b = 'x' WHERE id = 3")
    assert u == Update("t", (Assignment("a", Binary("+", col("a"), lit(1))), Assignment("b", lit("x"))),
                       Binary("=", col("id"), lit(3)))
    assert parse("DELETE FROM t") == Delete("t", None)
    assert parse("DELETE FROM t WHERE a < 1") == Delete("t", Binary("<", col("a"), lit(1)))


def test_transaction_statements():
    assert parse_script("begin; commit; rollback;") == [Begin(), Commit(), Rollback()]


def test_script_splitting():
    stmts = parse_script("SELECT 1;; SELECT 2 ;\nDELETE FROM t;")
    assert len(stmts) == 3
    assert parse_script("") == [] and parse_script(" ; ; -- nothing") == []


def test_parse_wants_exactly_one_statement():
    with pytest.raises(SQLSyntaxError, match="empty"):
        parse("  ")
    with pytest.raises(SQLSyntaxError, match="expected one statement but found 2"):
        parse("SELECT 1; SELECT 2")
    assert parse("SELECT 1;") == parse("SELECT 1")


# ---------- expressions: precedence and associativity ----------

def test_and_binds_tighter_than_or():
    assert where("a OR b AND c") == Binary("OR", col("a"), Binary("AND", col("b"), col("c")))
    assert where("a AND b OR c") == Binary("OR", Binary("AND", col("a"), col("b")), col("c"))


def test_not_binds_looser_than_comparison_but_tighter_than_and():
    assert where("NOT a = 1") == Unary("NOT", Binary("=", col("a"), lit(1)))
    assert where("NOT a AND b") == Binary("AND", Unary("NOT", col("a")), col("b"))
    assert where("NOT NOT a") == Unary("NOT", Unary("NOT", col("a")))


def test_multiplication_binds_tighter_than_addition():
    assert expr("1 + 2 * 3") == Binary("+", lit(1), Binary("*", lit(2), lit(3)))
    assert expr("(1 + 2) * 3") == Binary("*", Binary("+", lit(1), lit(2)), lit(3))


def test_subtraction_and_division_are_left_associative():
    assert expr("10 - 4 - 3") == Binary("-", Binary("-", lit(10), lit(4)), lit(3))
    assert expr("8 / 4 / 2") == Binary("/", Binary("/", lit(8), lit(4)), lit(2))


def test_arithmetic_binds_tighter_than_comparison():
    assert where("a + 1 < b * 2") == Binary("<", Binary("+", col("a"), lit(1)), Binary("*", col("b"), lit(2)))


def test_unary_minus():
    assert expr("-5") == lit(-5)                                    # folded into the literal
    assert expr("- -5") == lit(5)
    assert expr("-a") == Unary("-", col("a"))
    assert expr("-2 * 3") == Binary("*", lit(-2), lit(3))
    assert expr("-(a + 1)") == Unary("-", Binary("+", col("a"), lit(1)))
    assert expr("+5") == lit(5)
    assert expr("1 - -1") == Binary("-", lit(1), lit(-1))
    assert expr("-2.5") == lit(-2.5)


def test_between_does_not_swallow_the_following_and():
    assert where("x BETWEEN 1 AND 5 AND y") == Binary("AND", Between(col("x"), lit(1), lit(5)), col("y"))
    assert where("x NOT BETWEEN 1 + 1 AND 5") == Between(col("x"), Binary("+", lit(1), lit(1)), lit(5), True)


def test_in_like_is_null():
    assert where("x IN (1, 2, 3)") == InList(col("x"), (lit(1), lit(2), lit(3)), False)
    assert where("x NOT IN (1)") == InList(col("x"), (lit(1),), True)
    assert where("x LIKE 'a%'") == Like(col("x"), lit("a%"), False)
    assert where("x NOT LIKE 'a%'") == Like(col("x"), lit("a%"), True)
    assert where("x IS NULL") == IsNull(col("x"), False)
    assert where("x IS NOT NULL") == IsNull(col("x"), True)
    assert where("NOT x IN (1)") == Unary("NOT", InList(col("x"), (lit(1),), False))


def test_in_list_items_can_be_full_expressions():
    assert where("x IN (1 + 1, a OR b)") == InList(
        col("x"), (Binary("+", lit(1), lit(1)), Binary("OR", col("a"), col("b"))), False)


def test_literals():
    assert expr("NULL") == lit(None) and expr("true") == lit(True) and expr("FALSE") == lit(False)
    assert expr("'x'") == lit("x") and expr("1.5") == lit(1.5)


def test_literals_are_strictly_typed():
    assert lit(1) != lit(True) and lit(1) != lit(1.0) and lit(0) != lit(False) and lit(None) != lit(0)
    assert lit(1) == lit(1) and hash(lit(2)) == hash(lit(2))
    assert expr("1") != expr("TRUE") and expr("1") != expr("1.0")


def test_columns_and_functions():
    assert expr("t.a") == Column("a", "t")
    assert expr("count(*)") == Func("count", (), True)
    assert expr("MAX(a)") == Func("max", (col("a"),))
    assert expr("f()") == Func("f", ())
    assert expr("f(1, g(2) + 1)") == Func("f", (lit(1), Binary("+", Func("g", (lit(2),)), lit(1))))


def test_quoted_name_may_be_a_keyword():
    assert parse('SELECT "select" FROM "table"') == Select((SelectItem(col("select")),), "table")


# ---------- syntax errors ----------

@pytest.mark.parametrize("text, message", [
    ("SELECT", "expected an expression but found end of input"),
    ("SELECT 1 +", "expected an expression but found end of input"),
    ("SELECT * FROM", "expected a table name but found end of input"),
    ("SELECT * FROM WHERE", "reserved word 'WHERE'"),
    ("SELECT * FROM t WHERE", "expected an expression"),
    ("SELECT * FROM t garbage", "unexpected 'garbage' after the end of the statement"),
    ("SELECT (1", "expected '\\)' but found end of input"),
    ("SELECT 1)", "unexpected '\\)'"),
    ("SELECT a b", "unexpected 'b'"),
    ("SELECT 1 = 2 = 3", "unexpected '='"),
    ("SELECT a NOT b", "unexpected 'NOT'"),
    ("SELECT x IN ()", "expected an expression"),
    ("SELECT x IN 1", "expected '\\(' but found '1'"),
    ("SELECT x BETWEEN 1", "expected AND"),
    ("SELECT x IS 5", "expected NULL"),
    ("SELECT f(1,)", "expected an expression"),
    ("SELECT * FROM t LIMIT -1", "whole number"),
    ("SELECT * FROM t LIMIT 1.5", "whole number"),
    ("SELECT * FROM t LIMIT x", "whole number"),
    ("SELECT * FROM t ORDER a", "expected BY"),
    ("SELECT 1 AS", "expected an alias"),
    ("INSERT t VALUES (1)", "expected INTO"),
    ("INSERT INTO t VALUES", "expected '\\('"),
    ("INSERT INTO t VALUES (1), (1, 2)", "this row has 2 values but the first row has 1"),
    ("INSERT INTO t (a, b) VALUES (1)", "2 columns listed but 1 values given"),
    ("INSERT INTO t () VALUES ()", "expected a column name"),
    ("UPDATE t a = 1", "expected SET"),
    ("UPDATE t SET a 1", "expected '='"),
    ("UPDATE t SET a == 1", "expected an expression"),
    ("DELETE t", "expected FROM"),
    ("CREATE t (a INT)", "expected TABLE"),
    ("CREATE TABLE t ()", "expected a column name"),
    ("CREATE TABLE t (a)", "expected a column type"),
    ("CREATE TABLE t (a VARCHAR)", "expected a column type"),
    ("CREATE TABLE t (a INT, a TEXT)", "duplicate column 'a'"),
    ("CREATE TABLE t (a INT PRIMARY KEY, b INT PRIMARY KEY)", "only one PRIMARY KEY"),
    ("CREATE TABLE t (a INT PRIMARY)", "expected KEY"),
    ("CREATE TABLE t (a INT", "expected '\\)'"),
    ("DROP t", "expected TABLE"),
    ("FLY ME TO THE MOON", "expected a statement"),
    ("1 + 1", "expected a statement"),
])
def test_syntax_errors_say_what_was_expected(text, message):
    with pytest.raises(SQLSyntaxError, match=message):
        parse(text)


def test_error_points_at_the_offending_token_across_lines():
    with pytest.raises(SQLSyntaxError) as e:
        parse("SELECT a,\n       b\nFROM t\nWHERE a = = 1")
    assert (e.value.line, e.value.col) == (4, 11)
    assert str(e.value).splitlines()[-2:] == ["  WHERE a = = 1", "            ^"]


def test_reserved_word_as_name_suggests_quoting():
    with pytest.raises(SQLSyntaxError, match='double quotes'):
        parse("CREATE TABLE select (a INT)")
    parse('CREATE TABLE "select" (a INT)')


def test_deep_nesting_is_a_syntax_error_not_a_crash():
    for text in ["(" * 10000 + "1" + ")" * 10000, "f(" * 10000 + "1" + ")" * 10000,
                 "x IN (" * 10000 + "1" + ")" * 10000]:
        with pytest.raises(SQLSyntaxError, match="nested too deeply"):
            parse("SELECT " + text)
    assert parse("SELECT " + "(" * 40 + "1" + ")" * 40) == parse("SELECT 1")


def test_endless_chains_are_a_syntax_error_not_a_crash():
    for text in ["NOT " * 100000 + "a", "- " * 100000 + "a", " + ".join(["1"] * 20000),
                 " OR ".join(["a"] * 20000), " AND ".join(["a"] * 20000)]:
        with pytest.raises(SQLSyntaxError, match="too (complex|many)"):
            parse("SELECT " + text)


def test_long_but_reasonable_chains_work_and_roundtrip():
    for text in [" + ".join(["1"] * 200), " OR ".join(f"a = {i}" for i in range(100)), "NOT " * 100 + "a"]:
        ast = parse("SELECT " + text)
        assert parse(ast.sql()) == ast


def test_many_minus_signs_fold_correctly():
    assert expr("- " * 100 + "5") == lit(5) and expr("- " * 101 + "5") == lit(-5)
    assert expr("- " * 3 + "a") == Unary("-", Unary("-", Unary("-", col("a"))))


# ---------- canonical SQL and the AST dump ----------

STATEMENTS = [
    "CREATE TABLE IF NOT EXISTS t (id INT PRIMARY KEY, name TEXT NOT NULL, x REAL, ok BOOL)",
    "DROP TABLE IF EXISTS t", "DROP TABLE t",
    "INSERT INTO t VALUES (1, 'it''s', NULL, TRUE, -2.5)",
    "INSERT INTO t (a, b) VALUES (1, 2 + 3), (4, f(5))",
    "SELECT * FROM t", "SELECT 1 + 1", "SELECT a AS x, count(*) FROM t",
    "SELECT * FROM t WHERE a BETWEEN 1 AND 2 OR b NOT IN (1, 2) AND c IS NOT NULL AND d NOT LIKE 'x%'",
    "SELECT * FROM t ORDER BY a DESC, b LIMIT 3 OFFSET 4",
    "UPDATE t SET a = a + 1, b = 'x' WHERE NOT (a > 1 OR b < 2)",
    "DELETE FROM t WHERE x = -1", "DELETE FROM t", "BEGIN", "COMMIT", "ROLLBACK",
    'SELECT "Weird Name", "select", t."a b" FROM "my table"',
]


@pytest.mark.parametrize("text", STATEMENTS)
def test_sql_roundtrip(text):
    ast = parse(text)
    again = parse(ast.sql())
    assert again == ast
    assert again.sql() == ast.sql()


def test_canonical_form_uses_only_the_parentheses_it_needs():
    assert parse("select  a+1*2  from T where x=1 and y<>2").sql() == \
        "SELECT a + 1 * 2 FROM t WHERE x = 1 AND y != 2"
    for text in ["(a + b) * c", "a - (b - c)", "a / (b * c)", "NOT (a AND b)", "-(a + b)", "a OR (b OR c)",
                 "(a OR b) AND c", "(a = b) = c", "NOT (a = b) = c", "a - -b", "- -a", "(NOT a) = b",
                 "a BETWEEN (x BETWEEN 1 AND 2) AND 3", "(a IS NULL) IS NULL", "(a AND b) IN (1)"]:
        try:
            ast = expr(text)
        except SQLSyntaxError:
            continue                                     # not valid SQL as written; nothing to preserve
        assert expr(ast.sql()) == ast, text
    assert expr("(a + b) * c").sql() == "(a + b) * c"
    assert expr("a - (b - c)").sql() == "a - (b - c)"
    assert expr("a - b - c").sql() == "a - b - c"
    assert expr("NOT (a AND b)").sql() == "NOT (a AND b)"
    assert expr("-(a + b)").sql() == "-(a + b)"
    assert expr("- -a").sql() == "- -a"
    assert expr("(a OR b) AND c").sql() == "(a OR b) AND c"
    assert expr("a OR b AND c").sql() == "a OR b AND c"
    assert expr("a OR (b OR c)").sql() == "a OR (b OR c)"


def test_dump_shows_the_tree():
    out = dump(parse("SELECT a + 1 FROM t WHERE b"))
    assert out.splitlines()[0] == "Select(table='t', order_by=(), limit=None, offset=None)"
    assert "Binary(op='+')" in out and "Literal(value=1)" in out and "Column(name='b', table=None)" in out


# ---------- fuzzing: random valid SQL, random garbage, random mutations ----------

class Gen:
    """Generates random SQL text that follows the grammar, with random spacing, case and comments."""

    COLUMNS = ["a", "b", "id", "Name", '"Weird Name"', '"select"', "t.a", 'x."y z"']

    def __init__(self, seed):
        self.r = random.Random(seed)

    def sp(self):
        return self.r.choice([" ", "  ", "\n", "\t", " -- note\n "])

    def kw(self, w):
        return w.lower() if self.r.random() < 0.3 else w

    def join(self, parts, word):
        out = parts[0]
        for p in parts[1:]:
            out += self.sp() + self.kw(word) + self.sp() + p
        return out

    def or_(self, d):
        return self.join([self.and_(d) for _ in range(self.r.choice([1, 1, 2, 3]))], "OR")

    def and_(self, d):
        return self.join([self.not_(d) for _ in range(self.r.choice([1, 1, 2, 3]))], "AND")

    def not_(self, d):
        return (self.kw("NOT") + self.sp()) * self.r.choice([0, 0, 0, 1, 2]) + self.pred(d)

    def pred(self, d):
        a, c, neg = self.add(d), self.r.randrange(8), self.r.choice(["", self.kw("NOT") + self.sp()])
        s = self.sp()
        if c < 3:
            return a
        if c == 3:
            return a + s + self.r.choice(["=", "!=", "<>", "<", "<=", ">", ">="]) + s + self.add(d)
        if c == 4:
            return a + s + self.kw("IS") + s + neg + self.kw("NULL")
        if c == 5:
            return a + s + neg + self.kw("BETWEEN") + s + self.add(d) + s + self.kw("AND") + s + self.add(d)
        if c == 6:
            items = ("," + s).join(self.or_(d - 1) if d > 0 else self.atom() for _ in range(self.r.randint(1, 3)))
            return a + s + neg + self.kw("IN") + s + "(" + items + ")"
        return a + s + neg + self.kw("LIKE") + s + self.add(d)

    def add(self, d):
        out = self.mul(d)
        for _ in range(self.r.choice([0, 0, 1, 2])):
            out += self.sp() + self.r.choice("+-") + self.sp() + self.mul(d)
        return out

    def mul(self, d):
        out = self.unary(d)
        for _ in range(self.r.choice([0, 0, 1, 2])):
            out += self.sp() + self.r.choice("*/%") + self.sp() + self.unary(d)
        return out

    def unary(self, d):
        if self.r.random() < 0.15:
            return self.r.choice("-+") + self.sp() + self.unary(d)
        return self.primary(d)

    def atom(self):
        c = self.r.randrange(9)
        if c == 0:
            return str(self.r.randrange(10 ** 6))
        if c == 1:
            return self.r.choice(["1.5", "2.50e3", "1E-3", "0.0", "3e+2", "123456789.123"])
        if c == 2:
            return self.r.choice(["'hello'", "'it''s'", "''", "'a -- b'", "'x;y'", "'\"q\"'"])
        if c == 3:
            return self.kw(self.r.choice(["NULL", "TRUE", "FALSE"]))
        return self.r.choice(self.COLUMNS)

    def primary(self, d):
        if d <= 0 or self.r.random() < 0.6:
            return self.atom()
        c = self.r.randrange(3)
        if c == 0:
            return "(" + self.sp() + self.or_(d - 1) + self.sp() + ")"
        if c == 1:
            return self.r.choice(["count", "f", "MAX", '"odd fn"']) + "(" + ("," + self.sp()).join(
                self.or_(d - 1) for _ in range(self.r.randint(0, 3))) + ")"
        return "count(*)"


def test_random_valid_expressions_roundtrip():
    for seed in range(400):
        text = Gen(seed).or_(3 if seed % 20 == 0 else 2)           # a few big ones, many small
        a = parse("SELECT " + text)
        b = parse(a.sql())
        assert b == a, (seed, text)
        assert b.sql() == a.sql(), (seed, text)


def test_random_statements_roundtrip():
    for seed in range(200):
        g = Gen(seed)
        stmts = [f"SELECT {g.or_(1)} AS r{g.sp()}FROM{g.sp()}t{g.sp()}WHERE{g.sp()}{g.or_(1)}"
                 f"{g.sp()}ORDER{g.sp()}BY{g.sp()}{g.add(1)}{g.sp()}DESC{g.sp()}LIMIT 5",
                 f"UPDATE t SET a = {g.or_(1)}, b = {g.add(1)} WHERE {g.or_(1)}",
                 f"DELETE FROM t WHERE {g.or_(1)}",
                 f"INSERT INTO t (a, b) VALUES ({g.or_(1)}, {g.or_(1)}), ({g.or_(1)}, {g.or_(1)})"]
        for text in stmts:
            a = parse(text)
            assert parse(a.sql()) == a, (seed, text)


OPS = {"+": operator.add, "-": operator.sub, "*": operator.mul, "=": operator.eq, "!=": operator.ne,
       "<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge,
       "AND": lambda x, y: x and y, "OR": lambda x, y: x or y}


def evaluate(node):
    if isinstance(node, Literal):
        return node.value
    if isinstance(node, Unary):
        v = evaluate(node.operand)
        return (not v) if node.op == "NOT" else -v
    return OPS[node.op](evaluate(node.left), evaluate(node.right))


def test_precedence_matches_pythons_own_evaluator():
    """Python and SQL agree on the precedence of + - * unary-minus, comparisons, NOT, AND, OR."""
    for seed in range(500):
        r = random.Random(seed)

        def arith():
            out = ("-" + " " if r.random() < 0.2 else "") + str(r.randrange(0, 30))
            for _ in range(r.randrange(0, 4)):
                out += f" {r.choice('+-*')} " + ("- " if r.random() < 0.2 else "") + str(r.randrange(0, 30))
            return out

        def comparison():
            return ("NOT " if r.random() < 0.3 else "") + f"{arith()} {r.choice(['=', '!=', '<', '<=', '>', '>='])} {arith()}"

        text = comparison()
        for _ in range(r.randrange(0, 4)):
            text += f" {r.choice(['AND', 'OR'])} " + comparison()
        py = (text.replace("AND", "and").replace("OR", "or").replace("NOT", "not")
              .replace("!=", "!=").replace(" = ", " == "))
        assert evaluate(where(text)) == eval(py), (seed, text)


def test_arithmetic_precedence_matches_python():
    for seed in range(500):
        r = random.Random(seed)
        text = ("- " if r.random() < 0.3 else "") + str(r.randrange(50))
        for _ in range(r.randrange(1, 7)):
            text += f" {r.choice('+-*')} " + ("- " if r.random() < 0.25 else "") + str(r.randrange(50))
        assert evaluate(expr(text)) == eval(text), (seed, text)


ALPHABET = list("abc 0129'\"();,.*=<>!+-/%_\n\t@#é") + [
    "SELECT", "FROM", "WHERE", "AND", "OR", "NOT", "IN", "BETWEEN", "LIKE", "IS", "NULL", "INSERT INTO",
    "VALUES", "UPDATE", "SET", "DELETE", "CREATE TABLE", "INT", "TEXT", "LIMIT", "ORDER BY", "--", "count(*)"]


def must_parse_or_fail_cleanly(text):
    try:
        stmts = parse_script(text)
    except SQLSyntaxError as e:
        assert 0 <= e.pos <= len(text) and e.line >= 1 and e.col >= 1
        str(e)                                          # rendering the message must not crash either
        return
    for s in stmts:                                     # anything that parses must also round-trip
        assert parse(s.sql()) == s


def test_random_garbage_never_crashes_the_parser():
    r = random.Random(1)
    for _ in range(4000):
        must_parse_or_fail_cleanly("".join(r.choice(ALPHABET) + r.choice(["", " "]) for _ in range(r.randrange(0, 25))))


def test_mutated_valid_statements_never_crash_the_parser():
    r = random.Random(2)
    for text in STATEMENTS * 40:
        t = list(text)
        for _ in range(r.randint(1, 3)):
            i = r.randrange(len(t) + 1)
            kind = r.randrange(3)
            if kind == 0 and t:
                del t[min(i, len(t) - 1)]
            elif kind == 1:
                t.insert(i, r.choice(ALPHABET))
            elif len(t) > 1:
                j = r.randrange(len(t))
                t[min(i, len(t) - 1)], t[j] = t[j], t[min(i, len(t) - 1)]
        must_parse_or_fail_cleanly("".join(t))


@pytest.mark.parametrize("text", STATEMENTS)
def test_every_truncation_of_a_valid_statement_fails_cleanly_or_parses(text):
    for n in range(len(text) + 1):
        must_parse_or_fail_cleanly(text[:n])
