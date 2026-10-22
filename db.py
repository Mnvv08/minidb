"""minidb: tables, a catalog and a query executor on top of the B+ tree.

    db = Database("my.db")
    result = db.execute("SELECT name FROM users WHERE id BETWEEN 10 AND 50")
    result.columns, result.rows, result.rowcount, result.message, result.plan

How the pieces fit
------------------
* ONE B+ tree file holds everything. A key is  (table id << 40) + (primary key + 2**39) , so each
  table owns a contiguous slice of the key space and a range scan over a slice is a table scan.
  Table id 0 is the catalog: one JSON row per table (name, columns, next automatic key).
* A row is the table's non-key columns packed into bytes: a null bitmap, then INT/REAL as 8 bytes,
  BOOL as 1 byte, TEXT as a 2-byte length plus UTF-8. The primary key is not stored in the value;
  it is recovered from the key. A table without a PRIMARY KEY gets a hidden, ever-growing row id.
* Every statement is one atomic transaction (one log write, one fsync). BEGIN ... COMMIT / ROLLBACK
  groups several statements. Each statement checks everything BEFORE it writes anything, so a
  failing statement changes nothing, even inside a BEGIN block.
* The planner looks at the WHERE clause. Conditions on the primary key (=, <, <=, >, >=, BETWEEN,
  combined with AND) become a point lookup or a range scan instead of a full table scan.

Limits (honest ones): a row, and a table definition, must fit in one B+ tree value (1012 bytes);
an INT PRIMARY KEY must be within +-2**39; there are no joins, GROUP BY, secondary indexes or
ALTER TABLE.
"""
import json
import math
import struct
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal
from itertools import islice

import sql as S
from btree import BTree

INT_MIN, INT_MAX = -(2 ** 63), 2 ** 63 - 1
PK_BITS = 40
PK_BIAS = 1 << 39
PK_MIN, PK_MAX = -PK_BIAS, PK_BIAS - 1
MAX_TABLES = 1 << 22
CATALOG_ID = 0
AGG_NAMES = ("count", "sum", "avg", "min", "max")


class SQLError(Exception):
    """A statement that is valid SQL but cannot be executed (unknown table, bad type, ...)."""


class TransactionAborted(SQLError):
    """Something unexpected failed inside BEGIN; the whole transaction was rolled back."""


def pack_key(table_id, pk):
    return (table_id << PK_BITS) + pk + PK_BIAS


def key_pk(key):
    return (key & ((1 << PK_BITS) - 1)) - PK_BIAS


def table_range(table_id):
    return table_id << PK_BITS, (table_id << PK_BITS) + (1 << PK_BITS) - 1


def type_name(v):
    if v is None:
        return "NULL"
    if isinstance(v, bool):
        return "BOOL"
    if isinstance(v, int):
        return "INT"
    return "REAL" if isinstance(v, float) else "TEXT"


def kind(v):
    if isinstance(v, bool):
        return "bool"
    return "num" if isinstance(v, (int, float)) else "str"


# ------------------------------------------------------------------ tables and rows

class Table:
    def __init__(self, table_id, name, columns, next_rowid=1):
        self.id, self.name, self.columns, self.next_rowid = table_id, name, tuple(columns), next_rowid
        self.names = [c.name for c in self.columns]
        self.types = [c.type for c in self.columns]
        self.index = {n: i for i, n in enumerate(self.names)}
        self.pk = next((i for i, c in enumerate(self.columns) if c.primary_key), None)
        self.stored = [i for i in range(len(self.columns)) if i != self.pk]

    def to_json(self):
        return json.dumps({"id": self.id, "name": self.name, "next": self.next_rowid,
                           "cols": [[c.name, c.type, int(c.primary_key), int(c.not_null)] for c in self.columns]},
                          separators=(",", ":")).encode()

    @staticmethod
    def from_json(data):
        d = json.loads(data.decode())
        cols = [S.ColumnDef(n, t, bool(pk), bool(nn)) for n, t, pk, nn in d["cols"]]
        return Table(d["id"], d["name"], cols, d["next"])

    def encode(self, row):
        bitmap = bytearray((len(self.stored) + 7) // 8)
        parts = []
        for n, i in enumerate(self.stored):
            v = row[i]
            if v is None:
                bitmap[n // 8] |= 1 << (n % 8)
                continue
            t = self.types[i]
            if t == "INT":
                parts.append(struct.pack("<q", v))
            elif t == "REAL":
                parts.append(struct.pack("<d", v))
            elif t == "BOOL":
                parts.append(b"\x01" if v else b"\x00")
            else:
                try:
                    b = v.encode("utf-8")
                except UnicodeEncodeError:
                    raise SQLError(f"column '{self.names[i]}': text is not valid unicode") from None
                if len(b) > 65535:
                    raise SQLError(f"column '{self.names[i]}': text is too long")
                parts.append(struct.pack("<H", len(b)) + b)
        return bytes(bitmap) + b"".join(parts)

    def decode(self, key, data):
        row = [None] * len(self.columns)
        if self.pk is not None:
            row[self.pk] = key_pk(key)
        try:
            pos = (len(self.stored) + 7) // 8
            for n, i in enumerate(self.stored):
                if data[n // 8] >> (n % 8) & 1:
                    continue
                t = self.types[i]
                if t == "INT":
                    row[i] = struct.unpack_from("<q", data, pos)[0]
                    pos += 8
                elif t == "REAL":
                    row[i] = struct.unpack_from("<d", data, pos)[0]
                    pos += 8
                elif t == "BOOL":
                    row[i] = data[pos] == 1
                    pos += 1
                else:
                    n_bytes = struct.unpack_from("<H", data, pos)[0]
                    row[i] = data[pos + 2:pos + 2 + n_bytes].decode("utf-8")
                    pos += 2 + n_bytes
            if pos != len(data):
                raise IndexError("unexpected bytes after the last column")
        except (struct.error, IndexError, UnicodeDecodeError):
            raise SQLError(f"table '{self.name}' contains a corrupt row (key {key})") from None
        return tuple(row)


# ------------------------------------------------------------------ expressions

def check_int(n):
    if not INT_MIN <= n <= INT_MAX:
        raise SQLError("integer overflow")
    return n


def check_float(x):
    if not math.isfinite(x):
        raise SQLError("numeric overflow")
    return x


def need_num(x, what):
    if kind(x) != "num":
        raise SQLError(f"{what} needs a number, got {type_name(x)}")


def need_bool(x, what):
    if x is None or x is True or x is False:
        return x
    raise SQLError(f"{what} needs boolean operands, got {type_name(x)}")


def compare(x, y):
    if kind(x) != kind(y):
        raise SQLError(f"cannot compare {type_name(x)} with {type_name(y)}")
    return (x > y) - (x < y)


def three_and(a, b):
    if a is False or b is False:
        return False
    return None if a is None or b is None else True


def arith(op, x, y):
    if x is None or y is None:
        return None
    if kind(x) != "num" or kind(y) != "num":
        raise SQLError(f"cannot apply '{op}' to {type_name(x)} and {type_name(y)}")
    both_int = isinstance(x, int) and isinstance(y, int)
    try:
        if op == "+":
            r = x + y
        elif op == "-":
            r = x - y
        elif op == "*":
            r = x * y
        elif y == 0:
            raise SQLError("division by zero")
        elif op == "/":
            if both_int:                                    # SQL integer division truncates toward zero
                r = abs(x) // abs(y)
                r = -r if (x < 0) != (y < 0) else r
            else:
                r = x / y
        else:
            if both_int:
                q = abs(x) // abs(y)
                q = -q if (x < 0) != (y < 0) else q
                r = x - y * q
            else:
                r = math.fmod(x, y)
    except OverflowError:
        raise SQLError("numeric overflow") from None
    return check_int(r) if isinstance(r, int) else check_float(r)


def like(s, p):
    """SQL LIKE: % matches any run of characters, _ matches one. Linear-ish, no regex backtracking."""
    i = j = 0
    star, mark = -1, 0
    while i < len(s):
        if j < len(p) and p[j] == "%":
            star, mark = j, i
            j += 1
        elif j < len(p) and (p[j] == "_" or p[j] == s[i]):
            i += 1
            j += 1
        elif star != -1:
            j = star + 1
            mark += 1
            i = mark
        else:
            return False
    while j < len(p) and p[j] == "%":
        j += 1
    return j == len(p)


def sql_round(x, digits=0):
    if x is None or digits is None:
        return None
    need_num(x, "round()")
    if isinstance(digits, bool) or not isinstance(digits, int) or not 0 <= digits <= 15:
        raise SQLError("round() digits must be a whole number from 0 to 15")
    if isinstance(x, int):
        return x
    q = Decimal(repr(x)).quantize(Decimal(1).scaleb(-digits), rounding=ROUND_HALF_UP)
    return check_float(float(q))


def f_abs(x):
    if x is None:
        return None
    need_num(x, "abs()")
    return check_int(abs(x)) if isinstance(x, int) else abs(x)


def text_fn(name, fn):
    def run(x):
        if x is None:
            return None
        if not isinstance(x, str):
            raise SQLError(f"{name}() needs TEXT, got {type_name(x)}")
        return fn(x)
    return run


# name -> (min args, max args, python function over evaluated args; None = NULL in, NULL out)
SCALARS = {
    "abs": (1, 1, f_abs),
    "length": (1, 1, text_fn("length", len)),
    "lower": (1, 1, text_fn("lower", str.lower)),
    "upper": (1, 1, text_fn("upper", str.upper)),
    "round": (1, 2, sql_round),
}


def walk(node):
    """Yield every AST node under (and including) node, without recursion."""
    stack = [node]
    while stack:
        n = stack.pop()
        yield n
        if hasattr(n, "__dataclass_fields__"):
            for name in n.__dataclass_fields__:
                v = getattr(n, name)
                if hasattr(v, "__dataclass_fields__"):
                    stack.append(v)
                elif isinstance(v, tuple):
                    stack.extend(x for x in v if hasattr(x, "__dataclass_fields__"))


def contains_aggregate(node):
    return any(isinstance(n, S.Func) and n.name in AGG_NAMES for n in walk(node))


class Compiler:
    """Turns an AST into a Python closure. mode "row": closures take a row tuple, bare columns are
    fine and aggregates are not. mode "agg": closures take a dict of finished aggregate values."""

    def __init__(self, table=None, mode="row"):
        self.table, self.mode, self.aggs = table, mode, {}

    def compile(self, node):
        method = getattr(self, "c_" + type(node).__name__, None)
        if method is None:
            raise SQLError("'*' is only allowed as SELECT * or count(*)")
        return method(node)

    def c_Literal(self, node):
        v = node.value
        if isinstance(v, int) and not isinstance(v, bool):
            check_int(v)
        return lambda row: v

    def c_Column(self, node):
        if self.mode == "agg":
            raise SQLError(f"column '{node.name}' must appear inside an aggregate function")
        t = self.table
        if t is None:
            raise SQLError(f"no such column: {node.name}")
        if node.table is not None and node.table != t.name:
            raise SQLError(f"no such table: {node.table}")
        if node.name not in t.index:
            raise SQLError(f"no such column: {node.name}")
        i = t.index[node.name]
        return lambda row: row[i]

    def c_Unary(self, node):
        f = self.compile(node.operand)
        if node.op == "NOT":
            def run_not(row):
                x = need_bool(f(row), "NOT")
                return None if x is None else not x
            return run_not

        def run_neg(row):
            x = f(row)
            if x is None:
                return None
            need_num(x, "unary minus")
            return check_int(-x) if isinstance(x, int) else -x
        return run_neg

    def c_Binary(self, node):
        op, a, b = node.op, self.compile(node.left), self.compile(node.right)
        if op == "AND":
            def run_and(row):
                x = need_bool(a(row), "AND")
                if x is False:
                    return False
                y = need_bool(b(row), "AND")
                return three_and(x, y)
            return run_and
        if op == "OR":
            def run_or(row):
                x = need_bool(a(row), "OR")
                if x is True:
                    return True
                y = need_bool(b(row), "OR")
                if y is True:
                    return True
                return None if x is None or y is None else False
            return run_or
        if op in S.COMPARE:
            test = {"=": lambda c: c == 0, "!=": lambda c: c != 0, "<": lambda c: c < 0,
                    "<=": lambda c: c <= 0, ">": lambda c: c > 0, ">=": lambda c: c >= 0}[op]

            def run_cmp(row):
                x, y = a(row), b(row)
                return None if x is None or y is None else test(compare(x, y))
            return run_cmp
        return lambda row: arith(op, a(row), b(row))

    def c_Between(self, node):
        e, lo, hi, neg = self.compile(node.expr), self.compile(node.low), self.compile(node.high), node.negated

        def run(row):
            x, l, h = e(row), lo(row), hi(row)
            r1 = None if x is None or l is None else compare(x, l) >= 0
            r2 = None if x is None or h is None else compare(x, h) <= 0
            r = three_and(r1, r2)
            return r if r is None or not neg else not r
        return run

    def c_InList(self, node):
        e, items, neg = self.compile(node.expr), [self.compile(i) for i in node.items], node.negated

        def run(row):
            x = e(row)
            if x is None:
                return None
            saw_null = False
            for item in items:
                v = item(row)
                if v is None:
                    saw_null = True
                elif compare(x, v) == 0:
                    return not neg
            return None if saw_null else neg
        return run

    def c_Like(self, node):
        e, p, neg = self.compile(node.expr), self.compile(node.pattern), node.negated

        def run(row):
            x, pat = e(row), p(row)
            if x is None or pat is None:
                return None
            if not isinstance(x, str) or not isinstance(pat, str):
                raise SQLError("LIKE needs TEXT on both sides")
            r = like(x, pat)
            return not r if neg else r
        return run

    def c_IsNull(self, node):
        e, neg = self.compile(node.expr), node.negated
        return lambda row: (e(row) is not None) if neg else (e(row) is None)

    def c_Func(self, node):
        name = node.name
        if name in AGG_NAMES:
            return self.aggregate(node)
        if name == "coalesce" and not node.star:
            args = [self.compile(a) for a in node.args]
            if not args:
                raise SQLError("coalesce() needs at least one argument")

            def run_coalesce(row):
                for a in args:
                    v = a(row)
                    if v is not None:
                        return v
                return None
            return run_coalesce
        if name not in SCALARS:
            raise SQLError(f"no such function: {name}")
        low, high, fn = SCALARS[name]
        if node.star or not low <= len(node.args) <= high:
            raise SQLError(f"{name}() takes {low if low == high else f'{low} to {high}'} argument(s)")
        args = [self.compile(a) for a in node.args]
        return lambda row: fn(*[a(row) for a in args])

    def aggregate(self, node):
        if self.mode != "agg":
            raise SQLError(f"aggregate function {node.name}() is not allowed here")
        if node.star and node.name != "count":
            raise SQLError(f"{node.name}(*) is not valid")
        if not node.star and len(node.args) != 1:
            raise SQLError(f"{node.name}() takes exactly 1 argument")
        if node not in self.aggs:
            arg = None if node.star else Compiler(self.table, "row").compile(node.args[0])
            self.aggs[node] = (node.name, arg)
        return lambda ctx: ctx[node]


def run_aggregate(name, arg, rows):
    if name == "count":
        return len(rows) if arg is None else sum(1 for r in rows if arg(r) is not None)
    values = [v for v in map(arg, rows) if v is not None]
    if not values:
        return None
    if name in ("sum", "avg"):
        for v in values:
            need_num(v, f"{name}()")
        total = sum(values)
        if name == "sum":
            return check_int(total) if isinstance(total, int) else check_float(total)
        return check_float(total / len(values))
    if len({kind(v) for v in values}) > 1:
        raise SQLError(f"{name}() cannot mix {' and '.join(sorted({type_name(v) for v in values}))}")
    return min(values) if name == "min" else max(values)


# ------------------------------------------------------------------ planning

def null_first(v):
    """Sort key: NULL sorts before every value (so it comes last in a descending sort)."""
    return (0,) if v is None else (1, v)


def conjuncts(node):
    out, stack = [], [node]
    while stack:
        n = stack.pop()
        if isinstance(n, S.Binary) and n.op == "AND":
            stack.extend((n.right, n.left))
        else:
            out.append(n)
    return out


FLIP = {"=": "=", "<": ">", "<=": ">=", ">": "<", ">=": "<="}


def constant(node):
    """The value of an expression that mentions no columns, or NOCONST."""
    if any(isinstance(n, (S.Column, S.Star)) for n in walk(node)):
        return NOCONST
    try:
        return Compiler(None).compile(node)(())
    except SQLError:
        return NOCONST


NOCONST = object()


def pk_bounds(where, table):
    """Inclusive bounds (lo, hi) on the primary key implied by WHERE; None for 'no restriction'.

    Either side may be None (unbounded). lo > hi means no row can match. Only top-level AND
    conditions are used, and the full WHERE is still applied to every row that is fetched, so
    this can only ever make a query faster, never change its answer.
    """
    if where is None or table.pk is None:
        return None
    pk_name = table.names[table.pk]

    def is_pk(n):
        return isinstance(n, S.Column) and n.name == pk_name and n.table in (None, table.name)

    lo = hi = None

    def tighten_lo(v):
        nonlocal lo
        lo = v if lo is None else max(lo, v)

    def tighten_hi(v):
        nonlocal hi
        hi = v if hi is None else min(hi, v)

    for c in conjuncts(where):
        constraints = []                               # (operator, constant) pairs about the key
        if isinstance(c, S.Binary) and c.op in FLIP:
            if is_pk(c.left):
                constraints.append((c.op, constant(c.right)))
            elif is_pk(c.right):
                constraints.append((FLIP[c.op], constant(c.left)))
        elif isinstance(c, S.Between) and not c.negated and is_pk(c.expr):
            constraints += [(">=", constant(c.low)), ("<=", constant(c.high))]
        for op, v in constraints:
            if v is NOCONST:
                continue
            if v is None:                              # comparing with NULL is never true
                return (1, 0)
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue                               # a type error will surface when rows are evaluated
            if op == "=":
                if v != math.floor(v):
                    return (1, 0)
                tighten_lo(math.floor(v))
                tighten_hi(math.floor(v))
            elif op == "<":
                tighten_hi(math.ceil(v) - 1)
            elif op == "<=":
                tighten_hi(math.floor(v))
            elif op == ">":
                tighten_lo(math.floor(v) + 1)
            else:
                tighten_lo(math.ceil(v))
    if lo is None and hi is None:
        return None
    if lo is not None:
        lo = max(lo, PK_MIN)
    if hi is not None:
        hi = min(hi, PK_MAX)
    return lo, hi


# ------------------------------------------------------------------ results

@dataclass
class Result:
    columns: tuple = ()
    rows: list = None
    rowcount: int = 0
    message: str = ""
    plan: str = ""

    def __post_init__(self):
        if self.rows is None:
            self.rows = []


def format_value(v):
    if v is None:
        return "NULL"
    if v is True:
        return "TRUE"
    if v is False:
        return "FALSE"
    return str(v)


def format_result(result):
    """Render a Result as a text table (or its message for statements without rows)."""
    if not result.columns:
        return result.message
    cells = [[format_value(v) for v in row] for row in result.rows]
    widths = [max([len(c)] + [len(r[i]) for r in cells]) for i, c in enumerate(result.columns)]
    line = "+" + "+".join("-" * (w + 2) for w in widths) + "+"
    out = [line, "| " + " | ".join(c.ljust(w) for c, w in zip(result.columns, widths)) + " |", line]
    for r in cells:
        out.append("| " + " | ".join(v.ljust(w) for v, w in zip(r, widths)) + " |")
    out.append(line)
    out.append(f"({len(cells)} row{'s' if len(cells) != 1 else ''})")
    return "\n".join(out)


# ------------------------------------------------------------------ the database

class Database:
    def __init__(self, path, **btree_options):
        self.tree = BTree(path, **btree_options)
        self.tables = {}
        self._txn = None
        try:
            self._load_catalog()
        except Exception:
            self.tree.close()
            raise

    # ----- lifecycle and catalog

    def close(self):
        if self._txn is not None:
            self._rollback_txn()
        self.tree.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    @property
    def in_transaction(self):
        return self._txn is not None

    def _load_catalog(self):
        self.tables = {}
        lo, hi = table_range(CATALOG_ID)
        for _, data in self.tree.scan(lo, hi):
            t = Table.from_json(data)
            self.tables[t.name] = t

    def _save_table(self, t):
        blob = t.to_json()
        if len(blob) > self.tree.max_value:
            raise SQLError(f"table definition is too large ({len(blob)} bytes, limit {self.tree.max_value})")
        self.tree.put(pack_key(CATALOG_ID, t.id), blob)

    def table(self, name):
        t = self.tables.get(name)
        if t is None:
            raise SQLError(f"no such table: {name}")
        return t

    def schema(self, name=None):
        """CREATE TABLE text for one table, or for all of them."""
        names = [name] if name else sorted(self.tables)
        return "\n".join(S.CreateTable(n, self.table(n).columns).sql() for n in names)

    # ----- running statements

    def execute(self, text):
        return self._run(S.parse(text))

    def execute_script(self, text):
        return [self._run(s) for s in S.parse_script(text)]

    def _run(self, st):
        if isinstance(st, S.Begin):
            return self._begin()
        if isinstance(st, S.Commit):
            return self._commit()
        if isinstance(st, S.Rollback):
            return self._rollback()
        if isinstance(st, S.Select):
            return self._select(st)
        handler = {S.CreateTable: self._create_table, S.DropTable: self._drop_table, S.Insert: self._insert,
                   S.Update: self._update, S.Delete: self._delete}[type(st)]
        return self._mutate(lambda: handler(st))

    def _mutate(self, fn):
        if self._txn is not None:                       # inside BEGIN: statements join the transaction
            try:
                return fn()
            except SQLError:
                raise                                   # every check runs before the first write
            except BaseException as e:
                self._rollback_txn()
                raise TransactionAborted(f"transaction rolled back after an unexpected error: {e!r}") from e
        try:
            with self.tree.transaction():
                return fn()
        except BaseException:
            self._load_catalog()                        # forget any in-memory change that was undone
            raise

    def _begin(self):
        if self._txn is not None:
            raise SQLError("a transaction is already open")
        cm = self.tree.transaction()
        cm.__enter__()
        self._txn = cm
        return Result(message="BEGIN")

    def _commit(self):
        if self._txn is None:
            raise SQLError("no transaction is open")
        cm, self._txn = self._txn, None
        try:
            cm.__exit__(None, None, None)
        except BaseException:
            self._load_catalog()
            raise
        return Result(message="COMMIT")

    def _rollback(self):
        if self._txn is None:
            raise SQLError("no transaction is open")
        self._rollback_txn()
        return Result(message="ROLLBACK")

    def _rollback_txn(self):
        cm, self._txn = self._txn, None

        class Abort(Exception):
            pass
        err = Abort()
        try:
            cm.__exit__(Abort, err, None)
        except Abort:
            pass
        self._load_catalog()

    # ----- CREATE / DROP

    def _create_table(self, st):
        if st.name in self.tables:
            if st.if_not_exists:
                return Result(message=f"table {st.name} already exists")
            raise SQLError(f"table '{st.name}' already exists")
        for c in st.columns:
            if c.primary_key and c.type != "INT":
                raise SQLError(f"PRIMARY KEY column '{c.name}' must be INT")
        tid = max((t.id for t in self.tables.values()), default=0) + 1
        if tid >= MAX_TABLES:
            raise SQLError("too many tables")
        t = Table(tid, st.name, st.columns)
        self._save_table(t)
        self.tables[t.name] = t
        return Result(message=f"CREATE TABLE {t.name}")

    def _drop_table(self, st):
        t = self.tables.get(st.name)
        if t is None:
            if st.if_exists:
                return Result(message=f"no such table: {st.name}")
            raise SQLError(f"no such table: {st.name}")
        keys = [k for k, _ in self.tree.scan(*table_range(t.id))]
        for k in keys:
            self.tree.delete(k)
        self.tree.delete(pack_key(CATALOG_ID, t.id))
        del self.tables[t.name]
        return Result(message=f"DROP TABLE {t.name}", rowcount=len(keys))

    # ----- row helpers

    def _coerce(self, t, i, v):
        if v is None:
            return None
        want, name = t.types[i], t.names[i]
        ok = ((want == "INT" and type(v) is int) or (want == "REAL" and type(v) in (int, float))
              or (want == "TEXT" and type(v) is str) or (want == "BOOL" and type(v) is bool))
        if not ok:
            raise SQLError(f"column '{name}' expects {want} but got {type_name(v)} {v!r}")
        if want == "INT":
            return check_int(v)
        if want == "REAL":
            return check_float(float(v))
        return v

    def _finish_row(self, t, row):
        """NOT NULL checks and encoding (with the size limit) for a fully built row."""
        for i, c in enumerate(t.columns):
            if row[i] is None and c.not_null:
                raise SQLError(f"NOT NULL constraint failed: {t.name}.{c.name}")
        blob = t.encode(row)
        if len(blob) > self.tree.max_value:
            raise SQLError(f"row is too large ({len(blob)} bytes, limit {self.tree.max_value})")
        return blob

    def _fetch(self, t, bounds):
        """Yield (key, row) for the rows whose primary key is within bounds (None = whole table)."""
        if bounds is None:
            lo, hi = table_range(t.id)
        else:
            lo_pk = PK_MIN if bounds[0] is None else bounds[0]
            hi_pk = PK_MAX if bounds[1] is None else bounds[1]
            if lo_pk > hi_pk:
                return
            lo, hi = pack_key(t.id, lo_pk), pack_key(t.id, hi_pk)
            if lo == hi:                                 # point lookup: one root-to-leaf descent
                data = self.tree.get(lo)
                if data is not None:
                    yield lo, t.decode(lo, data)
                return
        for key, data in self.tree.scan(lo, hi):
            yield key, t.decode(key, data)

    def _describe_plan(self, t, bounds):
        if t.pk is None or bounds is None:
            return f"SCAN {t.name}"
        name = t.names[t.pk]
        lo, hi = bounds
        if lo is not None and hi is not None:
            if lo > hi:
                return f"SEARCH {t.name} USING PRIMARY KEY (no matching range)"
            if lo == hi:
                return f"SEARCH {t.name} USING PRIMARY KEY ({name} = {lo})"
            return f"SEARCH {t.name} USING PRIMARY KEY ({name} BETWEEN {lo} AND {hi})"
        return f"SEARCH {t.name} USING PRIMARY KEY ({name} {'>=' if hi is None else '<='} {lo if hi is None else hi})"

    def _compile_where(self, t, where):
        if where is None:
            return None
        if contains_aggregate(where):
            raise SQLError("aggregate functions are not allowed in WHERE")
        fn = Compiler(t).compile(where)

        def test(row):
            v = fn(row)
            if v is None or v is False:
                return False
            if v is True:
                return True
            raise SQLError(f"WHERE must be a boolean expression, got {type_name(v)}")
        return test

    def _matching(self, t, where):
        """Plan and run a WHERE: returns (list of (key, row), plan text). Materialised, so that
        the caller can safely modify the tree afterwards."""
        test = self._compile_where(t, where)
        bounds = pk_bounds(where, t)
        rows = [(k, r) for k, r in self._fetch(t, bounds) if test is None or test(r)]
        return rows, self._describe_plan(t, bounds)

    # ----- INSERT

    def _insert(self, st):
        t = self.table(st.table)
        if st.columns is None:
            targets = list(range(len(t.columns)))
            if len(st.rows[0]) != len(targets):
                raise SQLError(f"table {t.name} has {len(targets)} columns but {len(st.rows[0])} values were supplied")
        else:
            targets = []
            for name in st.columns:
                if name not in t.index:
                    raise SQLError(f"table {t.name} has no column named {name}")
                if t.index[name] in targets:
                    raise SQLError(f"column {name} specified twice")
                targets.append(t.index[name])
        value_fns = [[Compiler(None).compile(e) for e in row] for row in st.rows]
        nxt, seen, planned = t.next_rowid, set(), []
        for fns in value_fns:
            row = [None] * len(t.columns)
            for i, fn in zip(targets, fns):
                row[i] = self._coerce(t, i, fn(()))
            if t.pk is None or t.pk not in targets:
                pk = nxt                                  # hidden row id, or an automatic primary key
                if t.pk is not None:
                    row[t.pk] = pk
            else:
                pk = row[t.pk]
                if pk is None:
                    raise SQLError(f"NOT NULL constraint failed: {t.name}.{t.names[t.pk]}")
            if not PK_MIN <= pk <= PK_MAX:
                raise SQLError(f"key {pk} is out of range (a primary key must be within +-2**39)")
            nxt = max(nxt, pk + 1)
            key = pack_key(t.id, pk)
            if key in seen or self.tree.get(key) is not None:
                raise SQLError(f"UNIQUE constraint failed: {t.name}.{t.names[t.pk]} ({pk})"
                               if t.pk is not None else "row id collision")
            seen.add(key)
            planned.append((key, self._finish_row(t, row)))
        for key, blob in planned:
            self.tree.insert(key, blob)
        if nxt != t.next_rowid:
            t.next_rowid = nxt
            self._save_table(t)
        return Result(rowcount=len(planned), message=f"INSERT {len(planned)}")

    # ----- UPDATE / DELETE

    def _update(self, st):
        t = self.table(st.table)
        compiler = Compiler(t)
        assigns, seen_cols = [], set()
        for a in st.assignments:
            if a.column not in t.index:
                raise SQLError(f"table {t.name} has no column named {a.column}")
            if a.column in seen_cols:
                raise SQLError(f"column {a.column} assigned twice")
            seen_cols.add(a.column)
            if contains_aggregate(a.value):
                raise SQLError("aggregate functions are not allowed in SET")
            assigns.append((t.index[a.column], compiler.compile(a.value)))
        matched, plan = self._matching(t, st.where)
        old_keys = {k for k, _ in matched}
        changes, new_keys = [], set()
        for key, row in matched:
            new = list(row)
            for i, fn in assigns:
                new[i] = self._coerce(t, i, fn(row))      # every right-hand side sees the OLD row
            new_key = key
            if t.pk is not None and new[t.pk] != row[t.pk]:
                if new[t.pk] is None:
                    raise SQLError(f"NOT NULL constraint failed: {t.name}.{t.names[t.pk]}")
                if not PK_MIN <= new[t.pk] <= PK_MAX:
                    raise SQLError(f"key {new[t.pk]} is out of range (a primary key must be within +-2**39)")
                new_key = pack_key(t.id, new[t.pk])
            if new_key in new_keys:
                raise SQLError(f"UNIQUE constraint failed: {t.name}.{t.names[t.pk]}")
            new_keys.add(new_key)
            if new_key != key and new_key not in old_keys and self.tree.get(new_key) is not None:
                raise SQLError(f"UNIQUE constraint failed: {t.name}.{t.names[t.pk]} ({new[t.pk]})")
            changes.append((key, new_key, self._finish_row(t, new), new))
        for key, new_key, _, _ in changes:
            if key != new_key:
                self.tree.delete(key)                    # move: first remove all the old keys ...
        for _, new_key, blob, _ in changes:
            self.tree.put(new_key, blob)                 # ... then write all the new rows
        grew = False
        for _, _, _, new in changes:
            if t.pk is not None and new[t.pk] >= t.next_rowid:
                t.next_rowid, grew = new[t.pk] + 1, True
        if grew:
            self._save_table(t)
        return Result(rowcount=len(changes), message=f"UPDATE {len(changes)}", plan=plan)

    def _delete(self, st):
        t = self.table(st.table)
        matched, plan = self._matching(t, st.where)
        for key, _ in matched:
            self.tree.delete(key)
        return Result(rowcount=len(matched), message=f"DELETE {len(matched)}", plan=plan)

    # ----- SELECT

    def _select(self, st):
        t = self.table(st.table) if st.table is not None else None
        items = []                                        # (header, expression)
        for it in st.items:
            if isinstance(it.expr, S.Star):
                if t is None:
                    raise SQLError("SELECT * needs a FROM clause")
                items += [(n, S.Column(n)) for n in t.names]
            else:
                head = it.alias or (it.expr.name if isinstance(it.expr, S.Column) else it.expr.sql())
                items.append((head, it.expr))
        columns = tuple(h for h, _ in items)
        is_agg = any(contains_aggregate(e) for _, e in items)
        if is_agg and any(isinstance(it.expr, S.Star) for it in st.items):
            raise SQLError("cannot mix * with aggregate functions")

        if t is None:
            bounds, plan = None, "no table"
            source = iter([()])
            test = self._compile_where(None, st.where)
        else:
            test = self._compile_where(t, st.where)
            bounds = pk_bounds(st.where, t)
            plan = self._describe_plan(t, bounds)
            source = (r for _, r in self._fetch(t, bounds))
        rows_iter = (r for r in source if test is None or test(r))
        limit, offset = st.limit, st.offset or 0

        if is_agg:
            if st.order_by:
                raise SQLError("ORDER BY is not supported together with aggregate functions")
            comp = Compiler(t, "agg")
            fns = [comp.compile(e) for _, e in items]
            rows = list(rows_iter)
            ctx = {node: run_aggregate(name, arg, rows) for node, (name, arg) in comp.aggs.items()}
            out = [tuple(f(ctx) for f in fns)]
            out = out[offset:offset + limit] if limit is not None else out[offset:]
            return Result(columns, out, len(out), plan=plan)

        compiler = Compiler(t)
        fns = [compiler.compile(e) for _, e in items]
        aliases = {h: e for h, e in items}
        order, resolved = [], []                          # (compiled key, descending) and the expressions
        for o in st.order_by:
            expr = o.expr
            if isinstance(expr, S.Column) and expr.table is None and expr.name in aliases:
                expr = aliases[expr.name]                 # ORDER BY an output name means that output
            resolved.append(expr)
            order.append((compiler.compile(expr), o.desc))
        presorted = (t is not None and t.pk is not None and len(resolved) == 1 and
                     isinstance(resolved[0], S.Column) and resolved[0].name == t.names[t.pk] and
                     resolved[0].table in (None, t.name))  # rows already come out in key order
        desc_only = presorted and order[0][1]

        if not order or (presorted and not desc_only):
            stop = None if limit is None else offset + limit
            picked = islice(rows_iter, offset, stop)       # stops reading as soon as enough rows are found
            out = [tuple(f(r) for f in fns) for r in picked]
        else:
            rows = list(rows_iter)
            if desc_only:
                rows.reverse()
            else:
                try:
                    for fn, desc in reversed(order):      # stable sorts, last key first
                        rows.sort(key=lambda r, fn=fn: null_first(fn(r)), reverse=desc)
                except TypeError:
                    raise SQLError("ORDER BY cannot compare values of different types") from None
            rows = rows[offset:offset + limit] if limit is not None else rows[offset:]
            out = [tuple(f(r) for f in fns) for r in rows]
        return Result(columns, out, len(out), plan=plan)


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print('usage: python db.py FILE.db "SQL; SQL; ..."')
        sys.exit(2)
    with Database(sys.argv[1]) as database:
        try:
            for res in database.execute_script(" ".join(sys.argv[2:])):
                print(format_result(res))
                if res.plan:
                    print("plan:", res.plan)
        except (S.SQLSyntaxError, SQLError) as e:
            print("error:", e)
            sys.exit(1)
