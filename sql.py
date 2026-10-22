"""SQL tokenizer and recursive-descent parser.

    tokenize(text)     -> list of Token
    parse(text)        -> one statement (an AST node)
    parse_script(text) -> list of statements separated by ';'
    node.sql()         -> canonical SQL text for any node (parse(node.sql()) == node)
    dump(node)         -> indented tree, for looking at what the parser built

Supported statements:
    CREATE TABLE [IF NOT EXISTS] t (col TYPE [PRIMARY KEY] [NOT NULL], ...)   TYPE: INT TEXT REAL BOOL
    DROP TABLE [IF EXISTS] t
    INSERT INTO t [(col, ...)] VALUES (expr, ...), (expr, ...)
    SELECT item, ... [FROM t] [WHERE e] [ORDER BY e [ASC|DESC], ...] [LIMIT n [OFFSET m]]
    UPDATE t SET col = expr, ... [WHERE e]
    DELETE FROM t [WHERE e]
    BEGIN / COMMIT / ROLLBACK

Expressions, from loosest to tightest binding:
    OR  <  AND  <  NOT  <  = != <> < <= > >= IS [NOT] NULL [NOT] BETWEEN [NOT] IN [NOT] LIKE
        <  + -  <  * / %  <  unary - +  <  literals, columns, t.col, f(...), ( ... )

Every syntax error is an SQLSyntaxError that points at the exact place in the text.
"""
import math
import re
from dataclasses import dataclass, fields, is_dataclass

KEYWORDS = frozenset("""
    SELECT FROM WHERE INSERT INTO VALUES UPDATE SET DELETE CREATE DROP TABLE IF NOT EXISTS
    PRIMARY KEY NULL AND OR IS IN BETWEEN LIKE ORDER BY ASC DESC LIMIT OFFSET AS TRUE FALSE
    BEGIN COMMIT ROLLBACK
""".split())

TYPES = {"int": "INT", "integer": "INT", "text": "TEXT", "real": "REAL",
         "float": "REAL", "bool": "BOOL", "boolean": "BOOL"}

COMPARE = ("=", "!=", "<", "<=", ">", ">=")
MAX_NESTING = 50          # parentheses / function calls / IN lists inside each other
MAX_AST_DEPTH = 250       # depth of the finished tree (a chain of N operators is N levels deep)


class SQLSyntaxError(Exception):
    """A syntax error with its position: .message, .pos (offset), .line, .col (both from 1)."""

    def __init__(self, message, sql="", pos=0):
        self.message, self.sql, self.pos = message, sql, pos
        before = sql[:pos]
        self.line = before.count("\n") + 1
        self.col = pos - (before.rfind("\n") + 1) + 1
        super().__init__(self._render())

    def _render(self):
        if not self.sql:
            return self.message
        text = self.sql.split("\n")[self.line - 1].replace("\t", " ")
        return f"{self.message} (line {self.line}, column {self.col})\n  {text}\n  {' ' * (self.col - 1)}^"


# ------------------------------------------------------------------ tokenizer

@dataclass(frozen=True)
class Token:
    kind: str        # KW IDENT INT FLOAT STRING OP PUNCT EOF
    value: object    # KW: upper-case word; IDENT: name; INT/FLOAT/STRING: the python value; OP/PUNCT: text
    pos: int
    end: int


_NUMBER = re.compile(r"[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?")
_WORD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_SIMPLE_NAME = re.compile(r"[a-z_][a-z0-9_]*\Z")


def tokenize(sql):
    toks, i, n = [], 0, len(sql)
    while i < n:
        c = sql[i]
        if c.isspace():
            i += 1
        elif sql.startswith("--", i):                          # comment to end of line
            j = sql.find("\n", i)
            i = n if j < 0 else j + 1
        elif c in "0123456789":
            m = _NUMBER.match(sql, i)
            j = m.end()
            if j < n and (sql[j].isalpha() or sql[j] in "_."):
                raise SQLSyntaxError("malformed number", sql, i)
            try:
                if m.group(1) or m.group(2):
                    value = float(m.group())
                    if not math.isfinite(value):
                        raise SQLSyntaxError("number out of range", sql, i)
                    toks.append(Token("FLOAT", value, i, j))
                else:
                    toks.append(Token("INT", int(m.group()), i, j))
            except ValueError:
                raise SQLSyntaxError("number too large", sql, i) from None
            i = j
        elif c in "'\"":
            j, parts = i + 1, []
            while True:
                k = sql.find(c, j)
                if k < 0:
                    raise SQLSyntaxError("unterminated string" if c == "'" else "unterminated quoted name", sql, i)
                parts.append(sql[j:k])
                if sql.startswith(c * 2, k):                   # doubled quote = one literal quote
                    parts.append(c)
                    j = k + 2
                else:
                    j = k + 1
                    break
            text = "".join(parts)
            if c == "'":
                toks.append(Token("STRING", text, i, j))
            elif text == "":
                raise SQLSyntaxError("a quoted name cannot be empty", sql, i)
            else:
                toks.append(Token("IDENT", text, i, j))
            i = j
        elif c.isascii() and (c.isalpha() or c == "_"):
            m = _WORD.match(sql, i)
            word = m.group()
            if word.upper() in KEYWORDS:
                toks.append(Token("KW", word.upper(), i, m.end()))
            else:
                toks.append(Token("IDENT", word.lower(), i, m.end()))
            i = m.end()
        elif sql.startswith(("<=", ">=", "!=", "<>"), i):
            op = sql[i:i + 2]
            toks.append(Token("OP", "!=" if op == "<>" else op, i, i + 2))
            i += 2
        elif c in "=<>+-*/%":
            toks.append(Token("OP", c, i, i + 1))
            i += 1
        elif c in "(),;.":
            toks.append(Token("PUNCT", c, i, i + 1))
            i += 1
        else:
            raise SQLSyntaxError(f"unexpected character {c!r}", sql, i)
    toks.append(Token("EOF", None, n, n))
    return toks


# ------------------------------------------------------------------ AST

def _p(node, need):
    """SQL for an operand, in parentheses only if it binds looser than the position needs."""
    text = node.sql()
    return f"({text})" if node.PREC < need else text


BINARY_PREC = {"OR": 1, "AND": 2, "=": 4, "!=": 4, "<": 4, "<=": 4, ">": 4, ">=": 4,
               "+": 5, "-": 5, "*": 6, "/": 6, "%": 6}


def quote_ident(name):
    if _SIMPLE_NAME.match(name) and name.upper() not in KEYWORDS:
        return name
    return '"' + name.replace('"', '""') + '"'


@dataclass(frozen=True, eq=False)
class Literal:
    value: object                       # int, float, str, bool or None
    PREC = 8

    def __eq__(self, other):            # strict: Literal(1) must not equal Literal(True) or Literal(1.0)
        return isinstance(other, Literal) and type(self.value) is type(other.value) and self.value == other.value

    def __hash__(self):
        return hash((type(self.value), self.value))

    def sql(self):
        v = self.value
        if v is None:
            return "NULL"
        if v is True:
            return "TRUE"
        if v is False:
            return "FALSE"
        if isinstance(v, str):
            return "'" + v.replace("'", "''") + "'"
        return repr(v)


@dataclass(frozen=True)
class Column:
    name: str
    table: str = None
    PREC = 8

    def sql(self):
        return (quote_ident(self.table) + "." if self.table else "") + quote_ident(self.name)


@dataclass(frozen=True)
class Star:
    PREC = 8

    def sql(self):
        return "*"


@dataclass(frozen=True)
class Unary:
    op: str                             # "-" or "NOT"
    operand: object

    @property
    def PREC(self):
        return 3 if self.op == "NOT" else 7

    def sql(self):
        if self.op == "NOT":
            return f"NOT {_p(self.operand, 3)}"
        inner = _p(self.operand, 7)
        return "-" + (" " if inner.startswith("-") else "") + inner     # never "--": that starts a comment


@dataclass(frozen=True)
class Binary:
    op: str                             # OR AND = != < <= > >= + - * / %
    left: object
    right: object

    @property
    def PREC(self):
        return BINARY_PREC[self.op]

    def sql(self):
        p = self.PREC
        if p == 4:                                  # comparisons do not chain: both sides must be arithmetic
            return f"{_p(self.left, 5)} {self.op} {_p(self.right, 5)}"
        return f"{_p(self.left, p)} {self.op} {_p(self.right, p + 1)}"     # left-associative


@dataclass(frozen=True)
class Between:
    expr: object
    low: object
    high: object
    negated: bool = False
    PREC = 4

    def sql(self):
        return (f"{_p(self.expr, 5)} {'NOT ' if self.negated else ''}BETWEEN "
                f"{_p(self.low, 5)} AND {_p(self.high, 5)}")


@dataclass(frozen=True)
class InList:
    expr: object
    items: tuple
    negated: bool = False
    PREC = 4

    def sql(self):
        return f"{_p(self.expr, 5)} {'NOT ' if self.negated else ''}IN ({', '.join(i.sql() for i in self.items)})"


@dataclass(frozen=True)
class Like:
    expr: object
    pattern: object
    negated: bool = False
    PREC = 4

    def sql(self):
        return f"{_p(self.expr, 5)} {'NOT ' if self.negated else ''}LIKE {_p(self.pattern, 5)}"


@dataclass(frozen=True)
class IsNull:
    expr: object
    negated: bool = False
    PREC = 4

    def sql(self):
        return f"{_p(self.expr, 5)} IS {'NOT ' if self.negated else ''}NULL"


@dataclass(frozen=True)
class Func:
    name: str
    args: tuple = ()
    star: bool = False                  # count(*)
    PREC = 8

    def sql(self):
        inner = "*" if self.star else ", ".join(a.sql() for a in self.args)
        return f"{quote_ident(self.name)}({inner})"


@dataclass(frozen=True)
class SelectItem:
    expr: object
    alias: str = None

    def sql(self):
        return self.expr.sql() + (f" AS {quote_ident(self.alias)}" if self.alias else "")


@dataclass(frozen=True)
class OrderItem:
    expr: object
    desc: bool = False

    def sql(self):
        return self.expr.sql() + (" DESC" if self.desc else "")


@dataclass(frozen=True)
class Assignment:
    column: str
    value: object

    def sql(self):
        return f"{quote_ident(self.column)} = {self.value.sql()}"


@dataclass(frozen=True)
class ColumnDef:
    name: str
    type: str                           # INT TEXT REAL BOOL
    primary_key: bool = False
    not_null: bool = False              # always True for a primary key

    def sql(self):
        tail = " PRIMARY KEY" if self.primary_key else " NOT NULL" if self.not_null else ""
        return f"{quote_ident(self.name)} {self.type}{tail}"


@dataclass(frozen=True)
class CreateTable:
    name: str
    columns: tuple
    if_not_exists: bool = False

    def sql(self):
        ine = "IF NOT EXISTS " if self.if_not_exists else ""
        return f"CREATE TABLE {ine}{quote_ident(self.name)} ({', '.join(c.sql() for c in self.columns)})"


@dataclass(frozen=True)
class DropTable:
    name: str
    if_exists: bool = False

    def sql(self):
        return f"DROP TABLE {'IF EXISTS ' if self.if_exists else ''}{quote_ident(self.name)}"


@dataclass(frozen=True)
class Insert:
    table: str
    columns: tuple                      # None = all columns, in table order
    rows: tuple                         # tuple of tuples of expressions

    def sql(self):
        cols = f" ({', '.join(quote_ident(c) for c in self.columns)})" if self.columns is not None else ""
        rows = ", ".join("(" + ", ".join(e.sql() for e in row) + ")" for row in self.rows)
        return f"INSERT INTO {quote_ident(self.table)}{cols} VALUES {rows}"


@dataclass(frozen=True)
class Select:
    items: tuple
    table: str = None
    where: object = None
    order_by: tuple = ()
    limit: int = None
    offset: int = None

    def sql(self):
        out = "SELECT " + ", ".join(i.sql() for i in self.items)
        if self.table is not None:
            out += " FROM " + quote_ident(self.table)
        if self.where is not None:
            out += " WHERE " + self.where.sql()
        if self.order_by:
            out += " ORDER BY " + ", ".join(o.sql() for o in self.order_by)
        if self.limit is not None:
            out += f" LIMIT {self.limit}"
        if self.offset is not None:
            out += f" OFFSET {self.offset}"
        return out


@dataclass(frozen=True)
class Update:
    table: str
    assignments: tuple
    where: object = None

    def sql(self):
        out = f"UPDATE {quote_ident(self.table)} SET " + ", ".join(a.sql() for a in self.assignments)
        return out + (f" WHERE {self.where.sql()}" if self.where is not None else "")


@dataclass(frozen=True)
class Delete:
    table: str
    where: object = None

    def sql(self):
        return f"DELETE FROM {quote_ident(self.table)}" + (f" WHERE {self.where.sql()}" if self.where is not None else "")


@dataclass(frozen=True)
class Begin:
    def sql(self):
        return "BEGIN"


@dataclass(frozen=True)
class Commit:
    def sql(self):
        return "COMMIT"


@dataclass(frozen=True)
class Rollback:
    def sql(self):
        return "ROLLBACK"


def dump(node):
    """Indented tree view of any node, for humans."""
    return "\n".join(_dump(node, 0))


def _is_complex(x):
    return is_dataclass(x) or (isinstance(x, tuple) and any(is_dataclass(i) or isinstance(i, tuple) for i in x))


def _dump(v, ind):
    pad = "  " * ind
    if is_dataclass(v):
        simple = [(f.name, getattr(v, f.name)) for f in fields(v) if not _is_complex(getattr(v, f.name))]
        head = pad + type(v).__name__
        if simple:
            head += "(" + ", ".join(f"{n}={x!r}" for n, x in simple) + ")"
        out = [head]
        for f in fields(v):
            x = getattr(v, f.name)
            if _is_complex(x):
                out.append(f"{pad}  {f.name}:")
                out.extend(_dump(x, ind + 2))
        return out
    out = []                                                   # a tuple
    for i, x in enumerate(v):
        if isinstance(x, tuple):
            out.append(f"{pad}[{i}]")
            out.extend(_dump(x, ind + 1))
        else:
            out.extend(_dump(x, ind))
    return out


def too_deep(root):
    """True if the tree is deeper than MAX_AST_DEPTH. Iterative, so it cannot overflow the stack itself."""
    stack = [(root, 1)]
    while stack:
        node, depth = stack.pop()
        if depth > MAX_AST_DEPTH:
            return True
        if is_dataclass(node):
            stack.extend((getattr(node, f.name), depth + 1) for f in fields(node)
                         if is_dataclass(getattr(node, f.name)) or isinstance(getattr(node, f.name), tuple))
        elif isinstance(node, tuple):
            stack.extend((x, depth + 1) for x in node if is_dataclass(x) or isinstance(x, tuple))
    return False


# ------------------------------------------------------------------ parser

def nested(method):
    """Count nesting so a hostile input like '((((...' is a syntax error, not a RecursionError."""
    def wrapper(self, *args):
        self.depth += 1
        if self.depth > MAX_NESTING:
            self.error("expression is nested too deeply")
        try:
            return method(self, *args)
        finally:
            self.depth -= 1
    return wrapper


class Parser:
    def __init__(self, sql):
        self.sql = sql
        self.toks = tokenize(sql)
        self.i = 0
        self.depth = 0

    # ----- token helpers

    @property
    def tok(self):
        return self.toks[self.i]

    def advance(self):
        t = self.toks[self.i]
        if t.kind != "EOF":
            self.i += 1
        return t

    def describe(self, t):
        return "end of input" if t.kind == "EOF" else f"'{self.sql[t.pos:t.end]}'"

    def error(self, message, tok=None):
        raise SQLSyntaxError(message, self.sql, (tok or self.tok).pos)

    def is_kw(self, *names):
        return self.tok.kind == "KW" and self.tok.value in names

    def accept_kw(self, *names):
        if self.is_kw(*names):
            return self.advance().value
        return None

    def expect_kw(self, name):
        if not self.is_kw(name):
            self.error(f"expected {name} but found {self.describe(self.tok)}")
        self.advance()

    def is_punct(self, ch):
        return self.tok.kind == "PUNCT" and self.tok.value == ch

    def accept_punct(self, ch):
        if self.is_punct(ch):
            self.advance()
            return True
        return False

    def expect_punct(self, ch):
        if not self.accept_punct(ch):
            self.error(f"expected '{ch}' but found {self.describe(self.tok)}")

    def is_op(self, *ops):
        return self.tok.kind == "OP" and self.tok.value in ops

    def name(self, what="a name"):
        t = self.tok
        if t.kind == "IDENT":
            self.advance()
            return t.value
        if t.kind == "KW":
            self.error(f"expected {what} but found reserved word '{self.sql[t.pos:t.end]}' "
                       f"(put it in double quotes to use it as a name)")
        self.error(f"expected {what} but found {self.describe(t)}")

    def number(self, what):
        t = self.tok
        if t.kind != "INT":
            self.error(f"expected {what} (a whole number) but found {self.describe(t)}")
        self.advance()
        return t.value

    # ----- statements

    def script(self):
        stmts = []
        while True:
            while self.accept_punct(";"):
                pass
            if self.tok.kind == "EOF":
                return stmts
            start = self.tok
            stmt = self.statement()
            if too_deep(stmt):
                self.error(f"expression is too complex (more than {MAX_AST_DEPTH} levels deep)", start)
            stmts.append(stmt)
            if self.tok.kind != "EOF" and not self.is_punct(";"):
                self.error(f"unexpected {self.describe(self.tok)} after the end of the statement")

    def statement(self):
        t = self.tok
        handler = {"SELECT": self.select, "INSERT": self.insert, "UPDATE": self.update,
                   "DELETE": self.delete, "CREATE": self.create_table, "DROP": self.drop_table,
                   "BEGIN": self.simple(Begin), "COMMIT": self.simple(Commit),
                   "ROLLBACK": self.simple(Rollback)}.get(t.value if t.kind == "KW" else None)
        if handler is None:
            self.error("expected a statement (SELECT, INSERT, UPDATE, DELETE, CREATE, DROP, BEGIN, "
                       f"COMMIT or ROLLBACK) but found {self.describe(t)}")
        return handler()

    def simple(self, cls):
        def run():
            self.advance()
            return cls()
        return run

    def create_table(self):
        self.advance()
        self.expect_kw("TABLE")
        if_not_exists = False
        if self.accept_kw("IF"):
            self.expect_kw("NOT")
            self.expect_kw("EXISTS")
            if_not_exists = True
        name = self.name("a table name")
        self.expect_punct("(")
        cols, seen, pk_seen = [], set(), False
        while True:
            start = self.tok
            col = self.column_def()
            if col.name in seen:
                self.error(f"duplicate column '{col.name}'", start)
            if col.primary_key and pk_seen:
                self.error("a table can have only one PRIMARY KEY", start)
            seen.add(col.name)
            pk_seen = pk_seen or col.primary_key
            cols.append(col)
            if not self.accept_punct(","):
                break
        self.expect_punct(")")
        return CreateTable(name, tuple(cols), if_not_exists)

    def column_def(self):
        name = self.name("a column name")
        t = self.tok
        if t.kind != "IDENT" or t.value not in TYPES:
            self.error(f"expected a column type (INT, TEXT, REAL or BOOL) but found {self.describe(t)}")
        self.advance()
        pk = not_null = False
        while True:
            if self.accept_kw("PRIMARY"):
                self.expect_kw("KEY")
                pk = not_null = True
            elif self.accept_kw("NOT"):
                self.expect_kw("NULL")
                not_null = True
            elif self.accept_kw("NULL"):
                pass
            else:
                return ColumnDef(name, TYPES[t.value], pk, not_null)

    def drop_table(self):
        self.advance()
        self.expect_kw("TABLE")
        if_exists = False
        if self.accept_kw("IF"):
            self.expect_kw("EXISTS")
            if_exists = True
        return DropTable(self.name("a table name"), if_exists)

    def insert(self):
        self.advance()
        self.expect_kw("INTO")
        table = self.name("a table name")
        columns = None
        if self.accept_punct("("):
            columns = [self.name("a column name")]
            while self.accept_punct(","):
                columns.append(self.name("a column name"))
            self.expect_punct(")")
            columns = tuple(columns)
        self.expect_kw("VALUES")
        rows = []
        while True:
            start = self.tok
            self.expect_punct("(")
            row = [self.expr()]
            while self.accept_punct(","):
                row.append(self.expr())
            self.expect_punct(")")
            if rows and len(row) != len(rows[0]):
                self.error(f"this row has {len(row)} values but the first row has {len(rows[0])}", start)
            rows.append(tuple(row))
            if not self.accept_punct(","):
                break
        if columns is not None and len(rows[0]) != len(columns):
            self.error(f"{len(columns)} columns listed but {len(rows[0])} values given")
        return Insert(table, columns, tuple(rows))

    def select(self):
        self.advance()
        items = [self.select_item()]
        while self.accept_punct(","):
            items.append(self.select_item())
        table = where = limit = offset = None
        order_by = ()
        if self.accept_kw("FROM"):
            table = self.name("a table name")
            if self.accept_kw("WHERE"):
                where = self.expr()
            if self.accept_kw("ORDER"):
                self.expect_kw("BY")
                order = [self.order_item()]
                while self.accept_punct(","):
                    order.append(self.order_item())
                order_by = tuple(order)
            if self.accept_kw("LIMIT"):
                limit = self.number("a limit")
                if self.accept_kw("OFFSET"):
                    offset = self.number("an offset")
        return Select(tuple(items), table, where, order_by, limit, offset)

    def select_item(self):
        if self.is_op("*"):
            self.advance()
            return SelectItem(Star())
        expr = self.expr()
        alias = self.name("an alias") if self.accept_kw("AS") else None
        return SelectItem(expr, alias)

    def order_item(self):
        expr = self.expr()
        desc = self.accept_kw("ASC", "DESC") == "DESC"
        return OrderItem(expr, desc)

    def update(self):
        self.advance()
        table = self.name("a table name")
        self.expect_kw("SET")
        assigns = []
        while True:
            col = self.name("a column name")
            if not self.is_op("="):
                self.error(f"expected '=' but found {self.describe(self.tok)}")
            self.advance()
            assigns.append(Assignment(col, self.expr()))
            if not self.accept_punct(","):
                break
        where = self.expr() if self.accept_kw("WHERE") else None
        return Update(table, tuple(assigns), where)

    def delete(self):
        self.advance()
        self.expect_kw("FROM")
        table = self.name("a table name")
        where = self.expr() if self.accept_kw("WHERE") else None
        return Delete(table, where)

    # ----- expressions (one method per precedence level)

    def expr(self):
        return self.parse_or()

    @nested
    def parse_or(self):
        left = self.parse_and()
        while self.accept_kw("OR"):
            left = Binary("OR", left, self.parse_and())
        return left

    def parse_and(self):
        left = self.parse_not()
        while self.accept_kw("AND"):
            left = Binary("AND", left, self.parse_not())
        return left

    def parse_not(self):
        count = 0
        while self.accept_kw("NOT"):                   # a loop, not recursion: NOT NOT NOT ... is cheap
            count += 1
            if count > MAX_AST_DEPTH:
                self.error("too many NOTs in a row")
        node = self.parse_predicate()
        for _ in range(count):
            node = Unary("NOT", node)
        return node

    def parse_predicate(self):
        left = self.parse_add()
        t = self.tok
        if t.kind == "OP" and t.value in COMPARE:
            self.advance()
            return Binary(t.value, left, self.parse_add())
        if self.accept_kw("IS"):
            negated = bool(self.accept_kw("NOT"))
            self.expect_kw("NULL")
            return IsNull(left, negated)
        negated = False
        nxt = self.toks[min(self.i + 1, len(self.toks) - 1)]      # the last token is always EOF
        if self.is_kw("NOT") and nxt.kind == "KW" and nxt.value in ("BETWEEN", "IN", "LIKE"):
            self.advance()
            negated = True
        if self.accept_kw("BETWEEN"):
            low = self.parse_add()
            self.expect_kw("AND")
            return Between(left, low, self.parse_add(), negated)
        if self.accept_kw("IN"):
            self.expect_punct("(")
            items = [self.expr()]
            while self.accept_punct(","):
                items.append(self.expr())
            self.expect_punct(")")
            return InList(left, tuple(items), negated)
        if self.accept_kw("LIKE"):
            return Like(left, self.parse_add(), negated)
        return left

    def parse_add(self):
        left = self.parse_mul()
        while self.is_op("+", "-"):
            op = self.advance().value
            left = Binary(op, left, self.parse_mul())
        return left

    def parse_mul(self):
        left = self.parse_unary()
        while self.is_op("*", "/", "%"):
            op = self.advance().value
            left = Binary(op, left, self.parse_unary())
        return left

    def parse_unary(self):
        signs = []
        while self.is_op("-", "+"):
            signs.append(self.advance().value)
            if len(signs) > MAX_AST_DEPTH:
                self.error("too many signs in a row")
        node = self.parse_primary()
        for sign in reversed(signs):
            if sign == "-":
                if type(node) is Literal and type(node.value) in (int, float):
                    node = Literal(-node.value)               # fold: -5 is the literal -5
                else:
                    node = Unary("-", node)
        return node

    def parse_primary(self):
        t = self.tok
        if t.kind in ("INT", "FLOAT", "STRING"):
            self.advance()
            return Literal(t.value)
        if t.kind == "KW" and t.value in ("NULL", "TRUE", "FALSE"):
            self.advance()
            return Literal({"NULL": None, "TRUE": True, "FALSE": False}[t.value])
        if self.accept_punct("("):
            inner = self.expr()
            self.expect_punct(")")
            return inner
        if t.kind == "IDENT":
            self.advance()
            if self.accept_punct("("):
                if self.is_op("*"):
                    self.advance()
                    self.expect_punct(")")
                    return Func(t.value, (), True)
                args = []
                if not self.is_punct(")"):
                    args.append(self.expr())
                    while self.accept_punct(","):
                        args.append(self.expr())
                self.expect_punct(")")
                return Func(t.value, tuple(args))
            if self.accept_punct("."):
                return Column(self.name("a column name"), t.value)
            return Column(t.value)
        self.error(f"expected an expression but found {self.describe(t)}")


def parse_script(sql):
    return Parser(sql).script()


def parse(sql):
    stmts = Parser(sql).script()
    if not stmts:
        raise SQLSyntaxError("empty statement", sql, 0)
    if len(stmts) > 1:
        raise SQLSyntaxError(f"expected one statement but found {len(stmts)}", sql, 0)
    return stmts[0]


if __name__ == "__main__":
    import sys
    text = " ".join(sys.argv[1:]) or sys.stdin.read()
    try:
        for stmt in parse_script(text):
            print(dump(stmt))
            print("  =>", stmt.sql())
    except SQLSyntaxError as e:
        print(e)
        sys.exit(1)
