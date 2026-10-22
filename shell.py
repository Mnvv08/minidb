"""minidb shell: an interactive SQL prompt.

    python shell.py my.db                 interactive
    python shell.py my.db -c "SELECT 1"   run statements and exit
    python shell.py my.db < script.sql    run a script from a pipe

Statements end with ';' and may span several lines. Lines starting with '.' are shell commands
(type .help). Ctrl-C abandons the statement being typed; Ctrl-D exits.
"""
import os
import sys
import time

import sql as S
from db import Database, SQLError, TransactionAborted, format_result

HELP = """\
.help               show this help
.tables             list tables
.schema [TABLE]     show CREATE TABLE text
.plan on|off        show how each statement finds its rows
.timer on|off       show how long each statement took
.stats              show file statistics (rows, tree height, pages)
.read FILE          run the statements in FILE
.quit               leave (also .exit or Ctrl-D)"""


def is_complete(text):
    """True if text ends with a ';' (ignoring strings and comments). An unterminated string means
    'keep typing'; any other lexical error counts as complete so the user sees it at once."""
    try:
        toks = S.tokenize(text)
    except S.SQLSyntaxError as e:
        return "unterminated" not in e.message
    return len(toks) >= 2 and toks[-2].kind == "PUNCT" and toks[-2].value == ";"


class Shell:
    def __init__(self, db, out=None, plan=False, timer=False):
        self.db, self.out = db, out or sys.stdout
        self.show_plan, self.show_timer = plan, timer
        self.buffer = []
        self.errors = 0
        self.done = False

    def say(self, text=""):
        print(text, file=self.out)

    @property
    def prompt(self):
        if self.buffer:
            return "   ...> "
        return "minidb*> " if self.db.in_transaction else "minidb> "

    def feed(self, line):
        """Handle one line of input."""
        if not self.buffer:
            stripped = line.strip()
            if not stripped:
                return
            if stripped.startswith("."):
                self.command(stripped)
                return
        self.buffer.append(line)
        text = "\n".join(self.buffer)
        if is_complete(text):
            self.buffer = []
            self.run_sql(text)

    def cancel(self):
        self.buffer = []

    def finish(self):
        """End of input: a half-typed statement is an error."""
        if self.buffer:
            self.buffer = []
            self.error("incomplete statement (missing ';')")
        if self.db.in_transaction:
            self.db.execute("ROLLBACK")
            self.say("note: the open transaction was rolled back")

    def error(self, message):
        self.errors += 1
        self.say(f"error: {message}")

    def run_sql(self, text):
        start = time.perf_counter()
        try:
            results = self.db.execute_script(text)
        except (S.SQLSyntaxError, SQLError) as e:
            self.error(e)
            if isinstance(e, TransactionAborted):
                self.say("(the transaction was rolled back)")
            return
        elapsed = time.perf_counter() - start
        for r in results:
            self.say(format_result(r))
            if self.show_plan and r.plan:
                self.say(f"plan: {r.plan}")
        if self.show_timer:
            self.say(f"time: {elapsed * 1000:.2f} ms")

    def command(self, line):
        parts = line.split(None, 1)
        name, arg = parts[0].lower(), (parts[1].strip() if len(parts) > 1 else "")
        if name in (".quit", ".exit"):
            self.done = True
        elif name == ".help":
            self.say(HELP)
        elif name == ".tables":
            for t in sorted(self.db.tables):
                self.say(t)
        elif name == ".schema":
            try:
                text = self.db.schema(arg or None)
            except SQLError as e:
                self.error(e)
                return
            if text:
                self.say(text)
        elif name in (".plan", ".timer"):
            if arg.lower() not in ("on", "off"):
                self.error(f"usage: {name} on|off")
                return
            setattr(self, "show_plan" if name == ".plan" else "show_timer", arg.lower() == "on")
            self.say(f"{name[1:]} is {arg.lower()}")
        elif name == ".stats":
            st = self.db.tree.stats()
            self.say(f"tables: {len(self.db.tables)}")
            for k, v in st.items():
                self.say(f"{k.replace('_', ' ')}: {v}")
        elif name == ".read":
            self.read_file(arg)
        else:
            self.error(f"unknown command {name} (try .help)")

    def read_file(self, path):
        if not path:
            self.error("usage: .read FILE")
            return
        try:
            with open(path, encoding="utf-8") as f:
                lines = f.read().splitlines()
        except (OSError, UnicodeDecodeError) as e:
            self.error(f"cannot read {path}: {e}")
            return
        for line in lines:
            self.feed(line)
            if self.done:
                break
        if self.buffer:
            self.buffer = []
            self.error(f"{path}: incomplete statement (missing ';')")

    def run_lines(self, lines):
        for line in lines:
            self.feed(line.rstrip("\n"))
            if self.done:
                break
        self.finish()


def interactive(shell):
    try:
        import readline
        history = os.path.expanduser("~/.minidb_history")
        try:
            readline.read_history_file(history)
        except OSError:
            pass
        import atexit
        atexit.register(lambda: _save_history(readline, history))
    except ImportError:
        pass
    shell.say("minidb shell. Statements end with ';'. Type .help for commands, .quit to leave.")
    while not shell.done:
        try:
            line = input(shell.prompt)
        except EOFError:
            shell.say()
            break
        except KeyboardInterrupt:
            shell.cancel()
            shell.say("^C")
            continue
        shell.feed(line)
    shell.finish()


def _save_history(readline, path):
    try:
        readline.write_history_file(path)
    except OSError:
        pass


def main(argv):
    args = list(argv)
    command = None
    if "-c" in args:
        i = args.index("-c")
        if i + 1 >= len(args):
            print("usage: python shell.py FILE.db [-c \"SQL\"]")
            return 2
        command = args[i + 1]
        del args[i:i + 2]
    if len(args) != 1:
        print("usage: python shell.py FILE.db [-c \"SQL\"]")
        return 2
    try:
        db = Database(args[0])
    except Exception as e:
        print(f"error: cannot open {args[0]}: {e}")
        return 1
    with db:
        shell = Shell(db)
        if command is not None:
            if not is_complete(command):
                command += ";"                      # -c "SELECT 1" needs no trailing ';'
            shell.run_lines(command.splitlines() or [""])
        elif sys.stdin.isatty():
            interactive(shell)
        else:
            shell.run_lines(sys.stdin)
    return 1 if shell.errors and not sys.stdin.isatty() else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
