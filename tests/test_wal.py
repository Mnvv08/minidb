import os
import random
import shutil

import pytest

from btree import BTree
from pager import PAGE_SIZE, Pager, PagerError
from wal import FRAME_SIZE, TXN_HEADER, TXN_TRAILER, WALPager, encode_txn, parse_wal


class Crash(BaseException):
    """A simulated power cut. BaseException, so nothing in the code under test can swallow it."""


def page(n):
    return bytes([n]) * PAGE_SIZE


def snapshot(src, dst_dir):
    """Copy the files exactly as the OS holds them right now, as a crash would leave them.

    Data a process still has in its own user-space buffers is NOT in these files, just as it
    would be lost in a real crash.
    """
    dst = str(dst_dir / "t.db")
    shutil.copyfile(src, dst)
    if os.path.exists(str(src) + "-wal"):
        shutil.copyfile(str(src) + "-wal", dst + "-wal")
    return dst


def open_pager(path, **kw):
    return WALPager(str(path), sync=False, **kw)


def abandon(t):
    """Release a 'crashed' tree's file descriptors without any cleanup.

    A dead process is simply gone; the test has to free its descriptors itself, or ~400
    simulated crashes run out of them (macOS allows only 256 per process by default).
    Call it only AFTER the snapshot has been taken.
    """
    if t is None:
        return
    for close in (lambda: os.close(t.pager._wal), t.pager._f.close):
        try:
            close()
        except (OSError, ValueError):
            pass


# ---------- log record format ----------

def test_encode_parse_roundtrip():
    data = encode_txn(1, {3: page(3), 5: page(5)}) + encode_txn(2, {3: page(9)})
    txns, good = parse_wal(data)
    assert [t for t, _ in txns] == [1, 2]
    assert txns[0][1] == [(3, page(3)), (5, page(5))]
    assert txns[1][1] == [(3, page(9))]
    assert good == len(data)


def test_parse_stops_at_a_torn_tail():
    a, b = encode_txn(1, {1: page(1)}), encode_txn(2, {2: page(2)})
    for cut in (1, 11, 12, len(b) // 2, len(b) - 9, len(b) - 1):
        txns, good = parse_wal(a + b[:cut])
        assert [t for t, _ in txns] == [1]
        assert good == len(a)


def test_parse_rejects_corruption_and_ignores_everything_after_it():
    a, b, c = (encode_txn(i, {i: page(i)}) for i in (1, 2, 3))
    flipped = bytearray(b)
    flipped[len(flipped) // 2] ^= 0xFF
    txns, good = parse_wal(a + bytes(flipped) + c)
    assert [t for t, _ in txns] == [1]
    assert good == len(a)
    bad_marker = bytearray(b)
    bad_marker[-1] ^= 0xFF
    assert [t for t, _ in parse_wal(a + bytes(bad_marker) + c)[0]] == [1]


def test_parse_handles_garbage_and_empty_input():
    assert parse_wal(b"") == ([], 0)
    assert parse_wal(b"garbage" * 1000) == ([], 0)
    assert parse_wal(b"WTXN" + bytes(100)) == ([], 0)


# ---------- the WAL pager ----------

def test_commit_visible_and_rollback_discards(tmp_path):
    p = open_pager(tmp_path / "t.db")
    p.begin()
    n = p.allocate()
    p.write_page(n, page(7))
    assert p.read_page(n) == page(7)
    p.commit()
    assert p.num_pages == 2
    assert p.read_page(n) == page(7)

    p.begin()
    p.allocate()
    p.write_page(n, page(8))
    p.rollback()
    assert p.num_pages == 2
    assert p.read_page(n) == page(7)
    p.close()


def test_committed_data_survives_a_crash_before_checkpoint(tmp_path):
    p = open_pager(tmp_path / "a.db")
    p.begin()
    n = p.allocate()
    p.write_page(n, page(7))
    p.commit()
    (tmp_path / "snap").mkdir()
    path = snapshot(tmp_path / "a.db", tmp_path / "snap")
    assert os.path.getsize(path) == PAGE_SIZE          # the database file does not have it yet...
    q = open_pager(path)
    assert q.recovered_txns == 1                       # ...but the log does
    assert q.num_pages == 2
    assert q.read_page(n) == page(7)
    q.close()
    p.close()


def test_uncommitted_transaction_is_lost_in_a_crash(tmp_path):
    p = open_pager(tmp_path / "a.db")
    p.begin()
    n = p.allocate()
    p.write_page(n, page(7))
    (tmp_path / "snap").mkdir()
    path = snapshot(tmp_path / "a.db", tmp_path / "snap")
    q = open_pager(path)
    assert q.recovered_txns == 0
    assert q.num_pages == 1
    q.close()
    p.rollback()
    p.close()


def test_writes_outside_a_transaction_are_autocommitted(tmp_path):
    p = open_pager(tmp_path / "a.db")
    n = p.allocate()
    p.write_page(n, page(4))
    (tmp_path / "snap").mkdir()
    q = open_pager(snapshot(tmp_path / "a.db", tmp_path / "snap"))
    assert q.recovered_txns == 2
    assert q.read_page(n) == page(4)
    q.close()
    p.close()


def test_checkpoint_moves_data_into_database_file_and_empties_log(tmp_path):
    db = tmp_path / "a.db"
    p = open_pager(db)
    p.begin()
    n = p.allocate()
    p.write_page(n, page(7))
    p.commit()
    assert os.path.getsize(str(db) + "-wal") > 0
    p.checkpoint()
    assert os.path.getsize(str(db) + "-wal") == 0
    assert p.checkpoints == 1
    with Pager(str(db)) as plain:                      # a plain pager sees it: it is really in the file
        assert plain.num_pages == 2
        assert plain.read_page(n) == page(7)
    p.close()


def test_log_is_checkpointed_automatically_when_it_gets_big(tmp_path):
    p = open_pager(tmp_path / "a.db", checkpoint_bytes=2 * FRAME_SIZE)
    for i in range(10):
        p.begin()
        n = p.allocate()
        p.write_page(n, page(i))
        p.commit()
    assert p.checkpoints >= 3
    assert all(p.read_page(i + 1) == page(i) for i in range(10))
    p.close()


def test_clean_close_leaves_no_log_and_keeps_data(tmp_path):
    db = tmp_path / "a.db"
    p = open_pager(db)
    p.begin()
    n = p.allocate()
    p.write_page(n, page(7))
    p.commit()
    p.close()
    assert not os.path.exists(str(db) + "-wal")
    q = open_pager(db)
    assert q.recovered_txns == 0
    assert q.read_page(n) == page(7)
    q.close()


def test_transaction_misuse_raises(tmp_path):
    p = open_pager(tmp_path / "a.db")
    with pytest.raises(PagerError):
        p.commit()
    with pytest.raises(PagerError):
        p.rollback()
    p.begin()
    with pytest.raises(PagerError):
        p.begin()
    with pytest.raises(PagerError):
        p.checkpoint()
    p.rollback()
    p.close()


def test_log_without_database_file_is_discarded(tmp_path):
    db = tmp_path / "a.db"
    (tmp_path / "a.db-wal").write_bytes(encode_txn(1, {1: page(1)}))
    p = open_pager(db)
    assert p.recovered_txns == 0
    assert p.num_pages == 1
    p.close()


def test_recovery_discards_a_torn_tail_and_empties_the_log(tmp_path):
    p = open_pager(tmp_path / "a.db")
    p.begin()
    n = p.allocate()
    p.write_page(n, page(7))
    p.commit()
    (tmp_path / "snap").mkdir()
    path = snapshot(tmp_path / "a.db", tmp_path / "snap")
    with open(path + "-wal", "ab") as f:
        f.write(b"WTXN" + os.urandom(5000))             # a half-written record after the good one
    q = open_pager(path)
    assert q.recovered_txns == 1
    assert q.read_page(n) == page(7)
    assert os.path.getsize(path + "-wal") == 0
    q.close()
    p.close()


def test_crash_during_recovery_is_harmless(tmp_path):
    p = open_pager(tmp_path / "a.db")
    p.begin()
    n = p.allocate()
    p.write_page(n, page(7))
    p.commit()
    (tmp_path / "snap").mkdir()
    path = snapshot(tmp_path / "a.db", tmp_path / "snap")

    def hook(event):
        if event == "recover_applied":
            raise Crash(event)

    with pytest.raises(Crash):
        open_pager(path, fault_hook=hook)               # dies after applying, before emptying the log
    q = open_pager(path)                                # recovery simply runs again
    assert q.read_page(n) == page(7)
    assert q.num_pages == 2
    q.close()
    p.close()


# ---------- B+ tree transactions ----------

def small(path, **opts):
    return BTree(str(path), leaf_capacity=100, max_internal_keys=4,
                 pager_options=dict(sync=False, **opts))


def test_block_of_operations_commits_together(tmp_path):
    t = small(tmp_path / "a.db")
    with t.transaction():
        for k in range(100):
            t.insert(k, b"v")
    (tmp_path / "snap").mkdir()
    with small(snapshot(tmp_path / "a.db", tmp_path / "snap")) as r:
        assert r.validate() == 100
    t.close()


def test_exception_rolls_back_the_whole_block(tmp_path):
    with small(tmp_path / "a.db") as t:
        for k in range(20):
            t.insert(k, b"v")
        before, root, pages = list(t.items()), t.root, t.pager.num_pages
        with pytest.raises(RuntimeError):
            with t.transaction():
                for k in range(100, 200):
                    t.insert(k, b"x")
                assert t.count() == 120
                raise RuntimeError("boom")
        assert list(t.items()) == before
        assert t.root == root
        assert t.pager.num_pages == pages
        assert t.validate() == 20
        t.insert(500, b"still works")
        assert t.validate() == 21


def test_duplicate_key_inside_a_block_undoes_the_block(tmp_path):
    with small(tmp_path / "a.db") as t:
        t.insert(1, b"a")
        with pytest.raises(Exception):
            with t.transaction():
                t.insert(100, b"new")
                t.insert(1, b"dup")
        assert t.get(100) is None
        assert t.validate() == 1


def test_put_is_atomic(tmp_path):
    with small(tmp_path / "a.db") as t:
        for k in range(30):
            t.insert(k, b"old")

        def boom(key, value):
            raise RuntimeError("crash between the delete and the insert")

        t._insert_checked = boom
        with pytest.raises(RuntimeError):
            t.put(5, b"new")
        del t._insert_checked
        assert t.get(5) == b"old"                       # the delete half was undone
        assert t.validate() == 30


def test_every_single_operation_is_durable_without_closing(tmp_path):
    t = small(tmp_path / "a.db")
    for k in range(50):
        t.insert(k, b"v%d" % k)
    t.delete(10)
    t.put(20, b"changed")
    (tmp_path / "snap").mkdir()
    with small(snapshot(tmp_path / "a.db", tmp_path / "snap")) as r:
        assert r.validate() == 49
        assert r.get(10) is None
        assert r.get(20) == b"changed"
    t.close()


def test_a_block_costs_one_fsync_and_single_calls_one_each(tmp_path):
    events = []
    t = small(tmp_path / "a.db", fault_hook=events.append)
    events.clear()
    with t.transaction():
        for k in range(100):
            t.insert(k, b"v")
    assert events.count("wal_synced") == 1
    events.clear()
    for k in range(100, 150):
        t.insert(k, b"v")
    assert events.count("wal_synced") == 50
    t.close()


def test_operation_that_changes_nothing_writes_nothing(tmp_path):
    events = []
    with small(tmp_path / "a.db", fault_hook=events.append) as t:
        t.insert(1, b"a")
        events.clear()
        assert t.delete(99) is False
        assert "wal_write" not in events


# ---------- crash testing ----------

def make_ops(seed, n):
    rng = random.Random(seed)
    ops = []
    for _ in range(n):
        k = rng.randrange(40)
        if rng.random() < 0.35:
            ops.append(("delete", k))
        else:
            ops.append(("put", k, rng.randbytes(rng.randrange(0, 14))))
    return ops


def apply_model(model, op):
    if op[0] == "put":
        model[op[1]] = op[2]
    else:
        model.pop(op[1], None)


def apply_tree(t, op):
    if op[0] == "put":
        t.put(op[1], op[2])
    else:
        t.delete(op[1])


def run_until_crash(path, ops, crash_at, checkpoint_bytes):
    """Run the workload; raise a simulated power cut at the crash_at-th I/O event (0 = never).

    Returns (events seen, state before the interrupted operation, state after it, tree).
    The tree is returned only to keep it alive: a crashed process cannot clean up anything.
    """
    count = [0]

    def hook(event):
        count[0] += 1
        if count[0] == crash_at:
            raise Crash(event)

    try:
        t = BTree(str(path), leaf_capacity=100, max_internal_keys=4,
                  pager_options=dict(sync=False, checkpoint_bytes=checkpoint_bytes, fault_hook=hook))
    except Crash:
        return count[0], {}, {}, None                   # died while creating the database
    model = {}
    for op in ops:
        before = dict(model)
        apply_model(model, op)
        try:
            apply_tree(t, op)
        except Crash:
            return count[0], before, dict(model), t
    return count[0], dict(model), dict(model), t


@pytest.mark.parametrize("checkpoint_bytes", [2 * FRAME_SIZE, 6 * FRAME_SIZE, 10 ** 9])
def test_power_cut_at_every_single_io_event(tmp_path, checkpoint_bytes):
    ops = make_ops(seed=3, n=60)
    (tmp_path / "dry").mkdir()
    total, final, _, t = run_until_crash(tmp_path / "dry" / "t.db", ops, 0, checkpoint_bytes)
    t.close()
    assert total > 100                                  # the workload really does a lot of I/O
    seen = set()
    for crash_at in range(1, total + 1):
        run_dir = tmp_path / f"run{crash_at}"
        run_dir.mkdir()
        n, before, after, t = run_until_crash(run_dir / "t.db", ops, crash_at, checkpoint_bytes)
        assert n == crash_at                            # the crash really happened here
        snap_dir = tmp_path / f"snap{crash_at}"
        snap_dir.mkdir()
        path = snapshot(run_dir / "t.db", snap_dir)
        abandon(t)
        with BTree(path, leaf_capacity=100, max_internal_keys=4,
                   pager_options=dict(sync=False)) as r:
            r.validate()
            got = dict(r.items())
            assert got == before or got == after, f"crash at event {crash_at}: state is neither before nor after"
            seen.add("after" if got == after and got != before else "before")
            r.put(999, b"post-crash")
            assert r.validate() == len(got) + 1
        shutil.rmtree(run_dir)
        shutil.rmtree(snap_dir)
    assert seen == {"before", "after"}                  # both outcomes occurred: the test is not trivial


def test_log_cut_at_any_byte_gives_a_consistent_earlier_state(tmp_path):
    ops = make_ops(seed=8, n=14)
    (tmp_path / "w").mkdir()
    db = tmp_path / "w" / "t.db"
    _, final, _, t = run_until_crash(db, ops, 0, 10 ** 9)   # never checkpoints: everything is in the log
    states, model = [{}], {}
    for op in ops:
        apply_model(model, op)
        states.append(dict(model))
    wal = (tmp_path / "w" / "t.db-wal").read_bytes()
    base = db.read_bytes()                              # the database file as creation left it

    boundaries, pos = [], 0
    while pos < len(wal):
        n = TXN_HEADER.unpack_from(wal, pos)[2]
        pos += TXN_HEADER.size + n * FRAME_SIZE + TXN_TRAILER.size
        boundaries.append(pos)
    assert pos == len(wal)
    cuts = set(range(0, len(wal), 997)) | {len(wal)}
    for b in boundaries:
        cuts |= {b - 1, b, b + 1}
    cuts = sorted(c for c in cuts if 0 <= c <= len(wal))

    last_index = -1
    for cut in cuts:
        d = tmp_path / f"cut{cut}"
        d.mkdir()
        (d / "t.db").write_bytes(base)
        (d / "t.db-wal").write_bytes(wal[:cut])
        with small(d / "t.db") as r:
            r.validate()
            got = dict(r.items())
        assert got in states, f"cut at byte {cut} gave a state that never existed"
        index = states.index(got)
        assert index >= last_index                      # a longer log never gives an older state
        last_index = index
        shutil.rmtree(d)
    assert last_index == len(ops) or states[last_index] == final
    assert got == final                                 # the complete log recovers everything
    t.close()


def test_corruption_in_the_middle_of_the_log_stops_recovery_there(tmp_path):
    ops = make_ops(seed=21, n=14)
    (tmp_path / "w").mkdir()
    db = tmp_path / "w" / "t.db"
    _, final, _, t = run_until_crash(db, ops, 0, 10 ** 9)
    states, model = [{}], {}
    for op in ops:
        apply_model(model, op)
        states.append(dict(model))
    wal = (tmp_path / "w" / "t.db-wal").read_bytes()
    base = db.read_bytes()
    for offset in range(100, len(wal), 4999):
        damaged = bytearray(wal)
        damaged[offset] ^= 0xFF
        d = tmp_path / f"bad{offset}"
        d.mkdir()
        (d / "t.db").write_bytes(base)
        (d / "t.db-wal").write_bytes(bytes(damaged))
        with small(d / "t.db") as r:
            r.validate()
            assert dict(r.items()) in states
        shutil.rmtree(d)
    t.close()
