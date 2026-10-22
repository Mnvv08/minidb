import io
import subprocess
import sys
from pathlib import Path

import pytest

from db import Database
from shell import Shell, is_complete

ROOT = Path(__file__).resolve().parent.parent


@pytest.fixture
def sh(tmp_path):
    d = Database(str(tmp_path / "s.db"))
    out = io.StringIO()
    yield Shell(d, out), out
    d.close()


def run(sh, text):
    shell, out = sh
    out.truncate(0)
    out.seek(0)
    shell.run_lines(text.splitlines())
    return out.getvalue()


@pytest.mark.parametrize("text, complete", [
    ("SELECT 1;", True), ("SELECT 1", False), ("SELECT 1;  ", True), ("SELECT 1; -- done", True),
    ("SELECT ';'", False), ("SELECT ';';", True), ("SELECT 'abc", False), ("SELECT 'a''b';", True),
    ("SELECT \"a;", False), ("SELECT 1 -- ;", False), ("", False), (";", True), ("SELECT @;", True),
    ("SELECT 1;\nSELECT 2", False), ("SELECT 1;\nSELECT 2;", True),
])
def test_is_complete(text, complete):
    assert is_complete(text) is complete


def test_statements_can_span_lines_and_share_a_line(sh):
    out = run(sh, "CREATE TABLE t (id INT PRIMARY KEY,\n   v TEXT);\nINSERT INTO t VALUES (1, 'a'); INSERT INTO t VALUES (2, 'b');\n"
                  "SELECT *\nFROM t;")
    assert "CREATE TABLE t" in out and out.count("INSERT 1") == 2 and "(2 rows)" in out
    assert "error" not in out


def test_semicolon_inside_a_string_does_not_end_the_statement(sh):
    out = run(sh, "CREATE TABLE t (v TEXT);\nINSERT INTO t VALUES ('a;\nb');\nSELECT length(v) FROM t;")
    assert "error" not in out and "| 4" in out


def test_errors_are_reported_and_the_shell_carries_on(sh):
    shell, _ = sh
    out = run(sh, "SELECT nope;\nSELEC 1;\nSELECT 1 +;\nSELECT 42;")
    assert out.count("error:") == 3 and "42" in out and shell.errors == 3
    assert "^" in out                                              # syntax errors show a caret


def test_dot_commands(sh):
    run(sh, "CREATE TABLE a (id INT PRIMARY KEY, v TEXT NOT NULL);\nCREATE TABLE b (x REAL);")
    assert run(sh, ".tables").split() == ["a", "b"]
    assert run(sh, ".schema a").strip() == "CREATE TABLE a (id INT PRIMARY KEY, v TEXT NOT NULL)"
    assert len(run(sh, ".schema").splitlines()) == 2
    assert "no such table" in run(sh, ".schema nope")
    assert ".tables" in run(sh, ".help")
    assert "unknown command" in run(sh, ".nonsense")
    assert "usage" in run(sh, ".plan maybe")
    assert "keys:" in run(sh, ".stats")


def test_plan_and_timer_switches(sh):
    run(sh, "CREATE TABLE t (id INT PRIMARY KEY);\nINSERT INTO t VALUES (1);")
    assert "plan:" not in run(sh, "SELECT * FROM t WHERE id = 1;")
    run(sh, ".plan on")
    assert "plan: SEARCH t USING PRIMARY KEY (id = 1)" in run(sh, "SELECT * FROM t WHERE id = 1;")
    run(sh, ".timer on")
    assert "time:" in run(sh, "SELECT 1;")
    run(sh, ".timer off\n.plan off")
    assert "time:" not in run(sh, "SELECT 1;")


def test_dot_lines_inside_a_statement_are_part_of_it(sh):
    out = run(sh, "SELECT 1,\n.5;")
    assert "error" in out                                          # '.5' is not valid SQL; not run as a command


def test_read_file(sh, tmp_path):
    f = tmp_path / "s.sql"
    f.write_text("CREATE TABLE t (id INT PRIMARY KEY);\nINSERT INTO t VALUES (1), (2), (3);\n")
    assert "INSERT 3" in run(sh, f".read {f}")
    assert "| 3" in run(sh, "SELECT count(*) FROM t;")
    assert "cannot read" in run(sh, ".read /no/such/file.sql")
    assert "usage" in run(sh, ".read")
    g = tmp_path / "bad.sql"
    g.write_text("SELECT 1")
    assert "incomplete" in run(sh, f".read {g}")


def test_incomplete_statement_at_end_of_input_is_an_error(sh):
    assert "incomplete statement" in run(sh, "SELECT 1")


def test_quit_stops_reading(sh):
    out = run(sh, "SELECT 1;\n.quit\nSELECT 2;")
    assert "| 1" in out and "| 2" not in out


def test_open_transaction_at_exit_is_rolled_back_with_a_note(sh, tmp_path):
    shell, out = sh
    out = run(sh, "CREATE TABLE t (id INT PRIMARY KEY);\nBEGIN;\nINSERT INTO t VALUES (1);")
    assert "rolled back" in out
    assert shell.prompt == "minidb> "
    assert shell.db.execute("SELECT count(*) FROM t").rows == [(0,)]


def test_prompt_shows_state(sh):
    shell, _ = sh
    assert shell.prompt == "minidb> "
    shell.feed("BEGIN;")
    assert shell.prompt == "minidb*> "
    shell.feed("SELECT")
    assert shell.prompt == "   ...> "
    shell.cancel()
    assert shell.prompt == "minidb*> "


def cli(tmp_path, *args, stdin=None):
    return subprocess.run([sys.executable, str(ROOT / "shell.py"), str(tmp_path / "c.db"), *args], input=stdin,
                          capture_output=True, text=True, cwd=ROOT, timeout=60)


def test_command_line_dash_c(tmp_path):
    r = cli(tmp_path, "-c", "CREATE TABLE t (id INT PRIMARY KEY); INSERT INTO t VALUES (7)")
    assert r.returncode == 0, r.stderr
    r = cli(tmp_path, "-c", "SELECT id FROM t")                    # no trailing ';' needed
    assert r.returncode == 0 and "| 7" in r.stdout


def test_piped_script_and_exit_status(tmp_path):
    r = cli(tmp_path, stdin="CREATE TABLE t (id INT PRIMARY KEY);\nINSERT INTO t VALUES (1);\nSELECT * FROM t;\n")
    assert r.returncode == 0 and "(1 row)" in r.stdout
    r = cli(tmp_path, stdin="SELECT nope;\n")
    assert r.returncode == 1 and "error: no such column" in r.stdout


def test_cli_usage_and_unopenable_file(tmp_path):
    r = subprocess.run([sys.executable, str(ROOT / "shell.py")], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 2 and "usage" in r.stdout
    r = subprocess.run([sys.executable, str(ROOT / "shell.py"), str(tmp_path)], capture_output=True, text=True, cwd=ROOT)
    assert r.returncode == 1 and "cannot open" in r.stdout


def test_data_survives_between_shell_sessions(tmp_path):
    cli(tmp_path, "-c", "CREATE TABLE t (id INT PRIMARY KEY, v TEXT); INSERT INTO t VALUES (1, 'kept')")
    r = cli(tmp_path, "-c", "SELECT v FROM t")
    assert "kept" in r.stdout
