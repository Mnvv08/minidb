import os
import random

import pytest

from btree import KEY_MAX, KEY_MIN, BTree, CorruptTreeError, DuplicateKeyError
from pager import PAGE_SIZE


def small(path):
    return BTree(str(path), leaf_capacity=100, max_internal_keys=4)


def val(k):
    return b"v" + str(k).encode()


def build(path, n=200):
    t = small(path)
    for k in range(n):
        t.insert(k, val(k))
    return t


# ---------- range scans ----------

def test_scan_inclusive_bounds(tmp_path):
    with build(tmp_path / "t.db") as t:
        assert [k for k, _ in t.scan(10, 20)] == list(range(10, 21))
        assert [k for k, _ in t.scan(10, 10)] == [10]
        assert list(t.scan(10, 20))[0] == (10, val(10))


def test_scan_open_ended(tmp_path):
    with build(tmp_path / "t.db") as t:
        assert [k for k, _ in t.scan(lo=190)] == list(range(190, 200))
        assert [k for k, _ in t.scan(hi=4)] == list(range(5))
        assert [k for k, _ in t.scan()] == list(range(200))


def test_scan_bounds_between_keys_and_outside(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in range(0, 100, 10):
            t.insert(k, val(k))
        assert [k for k, _ in t.scan(15, 45)] == [20, 30, 40]
        assert list(t.scan(91, 99)) == []
        assert list(t.scan(-50, -1)) == []
        assert [k for k, _ in t.scan(-50, 5)] == [0]
        assert [k for k, _ in t.scan(85, 500)] == [90]


def test_scan_lo_greater_than_hi_is_empty(tmp_path):
    with build(tmp_path / "t.db") as t:
        assert list(t.scan(20, 10)) == []


def test_scan_empty_tree(tmp_path):
    with small(tmp_path / "t.db") as t:
        assert list(t.scan()) == []
        assert list(t.scan(1, 5)) == []


def test_scan_crosses_many_leaves_and_reads_few_pages(tmp_path):
    with build(tmp_path / "t.db", 300) as t:
        t.page_reads = 0
        got = [k for k, _ in t.scan(100, 120)]
        assert got == list(range(100, 121))
        reads = t.page_reads
        t.page_reads = 0
        assert t.count() == 300
        assert reads < t.page_reads / 3          # a narrow range touches far fewer pages than a full scan


def test_scan_extreme_keys(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in (KEY_MIN, -1, 0, 1, KEY_MAX):
            t.insert(k, b"x")
        assert [k for k, _ in t.scan(KEY_MIN, KEY_MAX)] == [KEY_MIN, -1, 0, 1, KEY_MAX]
        assert [k for k, _ in t.scan(KEY_MAX, KEY_MAX)] == [KEY_MAX]


def test_scan_validates_arguments(tmp_path):
    with small(tmp_path / "t.db") as t:
        with pytest.raises(TypeError):
            list(t.scan("a", 5))
        with pytest.raises(ValueError):
            list(t.scan(0, KEY_MAX + 1))


# ---------- delete ----------

def test_delete_existing_and_missing(tmp_path):
    with small(tmp_path / "t.db") as t:
        t.insert(1, b"a")
        t.insert(2, b"b")
        assert t.delete(1) is True
        assert t.delete(1) is False
        assert t.delete(99) is False
        assert t.get(1) is None
        assert t.get(2) == b"b"
        assert t.validate() == 1


def test_delete_from_empty_tree(tmp_path):
    with small(tmp_path / "t.db") as t:
        assert t.delete(5) is False
        assert t.validate() == 0


def test_delete_validates_key(tmp_path):
    with small(tmp_path / "t.db") as t:
        with pytest.raises(TypeError):
            t.delete("1")
        with pytest.raises(ValueError):
            t.delete(KEY_MAX + 1)


def test_delete_all_ascending_shrinks_tree_to_one_leaf(tmp_path):
    with build(tmp_path / "t.db", 300) as t:
        assert t.height() >= 4
        for k in range(300):
            assert t.delete(k) is True
            if k % 10 == 0:
                t.validate()
        assert t.count() == 0
        assert t.height() == 1
        assert t.validate() == 0


def test_delete_all_descending(tmp_path):
    with build(tmp_path / "t.db", 300) as t:
        for k in range(299, -1, -1):
            assert t.delete(k) is True
            if k % 10 == 0:
                t.validate()
        assert t.height() == 1
        assert t.validate() == 0


def test_delete_all_in_random_order_validating_every_step(tmp_path):
    rng = random.Random(5)
    keys = list(range(200))
    rng.shuffle(keys)
    with build(tmp_path / "t.db", 200) as t:
        remaining = set(keys)
        for k in keys:
            assert t.delete(k) is True
            remaining.discard(k)
            assert t.validate() == len(remaining)
        assert t.height() == 1


def test_remaining_keys_are_intact_after_deleting_half(tmp_path):
    with build(tmp_path / "t.db", 300) as t:
        for k in range(0, 300, 2):
            t.delete(k)
        assert t.validate() == 150
        assert [k for k, _ in t.items()] == list(range(1, 300, 2))
        assert all(t.get(k) == val(k) for k in range(1, 300, 2))
        assert all(t.get(k) is None for k in range(0, 300, 2))


def test_deleting_from_the_middle_keeps_scans_correct(tmp_path):
    with build(tmp_path / "t.db", 300) as t:
        for k in range(100, 200):
            t.delete(k)
        assert t.validate() == 200
        assert [k for k, _ in t.scan(95, 205)] == list(range(95, 100)) + list(range(200, 206))


# ---------- page reuse (free list) ----------

def test_freed_pages_are_reused(tmp_path):
    path = tmp_path / "t.db"
    with build(path, 300) as t:
        pages_when_full = t.pager.num_pages
        for k in range(300):
            t.delete(k)
        s = t.stats()
        assert s["tree_pages"] == 1
        assert s["free_pages"] == pages_when_full - 3       # all but header, meta and the one root leaf
        for k in range(300):
            t.insert(k, val(k))
        assert t.pager.num_pages <= pages_when_full         # the file did not grow
        assert t.validate() == 300


def test_file_does_not_grow_during_insert_delete_churn(tmp_path):
    rng = random.Random(11)
    with small(tmp_path / "t.db") as t:
        live = set()
        for _ in range(1500):
            k = rng.randrange(150)
            if k in live:
                t.delete(k)
                live.discard(k)
            else:
                t.insert(k, val(k))
                live.add(k)
        t.validate()
        assert t.pager.num_pages < 150     # bounded: 150 keys can never need that many pages


def test_free_list_survives_reopen(tmp_path):
    path = tmp_path / "t.db"
    with build(path, 200) as t:
        for k in range(150):
            t.delete(k)
        free_before = t.stats()["free_pages"]
        assert free_before > 0
    with small(path) as t:
        assert t.stats()["free_pages"] == free_before
        assert t.validate() == 50
        size = t.pager.num_pages
        for k in range(1000, 1100):
            t.insert(k, val(k))
        assert t.validate() == 150
        assert t.pager.num_pages <= size + 100


def test_reading_a_freed_page_is_corruption(tmp_path):
    with build(tmp_path / "t.db", 100) as t:
        for k in range(100):
            t.delete(k)
        freed = t._free_pages()[0]
        with pytest.raises(CorruptTreeError, match="node type"):
            t._read_node(freed)


def test_validate_detects_leaked_page(tmp_path):
    with build(tmp_path / "t.db", 100) as t:
        for k in range(100):
            t.delete(k)
        assert t.free_head != 0
        t.free_head = 0                       # forget the free list: those pages are now leaked
        with pytest.raises(CorruptTreeError, match="leaked"):
            t.validate()


def test_validate_detects_free_list_cycle(tmp_path):
    with build(tmp_path / "t.db", 100) as t:
        for k in range(100):
            t.delete(k)
        pages = t._free_pages()
        assert len(pages) >= 2
        from btree import FREE, NODE_HEADER
        import struct
        buf = bytearray(t.pager.read_page(pages[-1]))
        NODE_HEADER.pack_into(buf, 0, FREE, 0, pages[0])      # last page points back to the first
        t.pager.write_page(pages[-1], bytes(buf))
        with pytest.raises(CorruptTreeError, match="cycle"):
            t.validate()


def test_validate_detects_underfull_leaf(tmp_path):
    with build(tmp_path / "t.db", 100) as t:
        leaf_page = t.path_to(50)[-1]
        node = t._read_node(leaf_page)
        node.keys, node.values = node.keys[:1], node.values[:1]
        t._write_node(leaf_page, node)
        with pytest.raises(CorruptTreeError):
            t.validate()


# ---------- put (replace) ----------

def test_put_replaces_and_can_shrink_without_breaking_balance(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in range(100):
            t.insert(k, b"x" * 13)
        for k in range(100):
            t.put(k, b"y")                    # every value shrinks a lot
        assert t.validate() == 100
        assert all(t.get(k) == b"y" for k in range(100))


def test_put_with_invalid_value_does_not_lose_the_old_one(tmp_path):
    with small(tmp_path / "t.db") as t:
        t.insert(1, b"keep")
        with pytest.raises(ValueError):
            t.put(1, b"x" * (t.max_value + 1))
        with pytest.raises(TypeError):
            t.put(1, "text")
        assert t.get(1) == b"keep"


def test_insert_duplicate_after_delete_works(tmp_path):
    with small(tmp_path / "t.db") as t:
        t.insert(1, b"a")
        with pytest.raises(DuplicateKeyError):
            t.insert(1, b"b")
        t.delete(1)
        t.insert(1, b"b")
        assert t.get(1) == b"b"


# ---------- randomized model tests ----------

def run_model(t, rng, steps, key_range, max_value_len):
    model = {}
    for i in range(steps):
        op = rng.choice(["insert", "insert", "put", "delete", "delete", "scan", "get"])
        k = rng.randrange(-key_range, key_range)
        if op == "insert":
            v = rng.randbytes(rng.randrange(0, max_value_len + 1))
            if k in model:
                with pytest.raises(DuplicateKeyError):
                    t.insert(k, v)
            else:
                t.insert(k, v)
                model[k] = v
        elif op == "put":
            v = rng.randbytes(rng.randrange(0, max_value_len + 1))
            t.put(k, v)
            model[k] = v
        elif op == "delete":
            assert t.delete(k) == (k in model)
            model.pop(k, None)
        elif op == "get":
            assert t.get(k) == model.get(k)
        else:
            lo, hi = sorted((k, rng.randrange(-key_range, key_range)))
            expected = [(a, model[a]) for a in sorted(model) if lo <= a <= hi]
            assert list(t.scan(lo, hi)) == expected
        if i % 50 == 0:
            assert t.validate() == len(model)
    assert t.validate() == len(model)
    assert list(t.items()) == sorted(model.items())


def test_random_mixed_operations_small_nodes(tmp_path):
    with small(tmp_path / "t.db") as t:
        run_model(t, random.Random(2024), steps=4000, key_range=300, max_value_len=13)


def test_random_mixed_operations_other_shapes(tmp_path):
    shapes = [(64, 3), (80, 3), (200, 5), (300, 8)]
    for n, (cap, internal) in enumerate(shapes):
        with BTree(str(tmp_path / f"s{n}.db"), leaf_capacity=cap, max_internal_keys=internal) as t:
            run_model(t, random.Random(100 + n), steps=2500, key_range=250,
                      max_value_len=t.max_value)


def test_random_mixed_operations_default_size_with_big_values(tmp_path):
    with BTree(str(tmp_path / "t.db")) as t:
        run_model(t, random.Random(77), steps=2500, key_range=400, max_value_len=t.max_value)


def test_random_operations_survive_reopen(tmp_path):
    path = tmp_path / "t.db"
    rng = random.Random(9)
    model = {}
    for _ in range(5):                         # five sessions: work, close, reopen, verify
        with small(path) as t:
            assert dict(t.items()) == model
            for _ in range(400):
                k = rng.randrange(200)
                if rng.random() < 0.5:
                    v = rng.randbytes(rng.randrange(0, 14))
                    t.put(k, v)
                    model[k] = v
                else:
                    t.delete(k)
                    model.pop(k, None)
            assert t.validate() == len(model)
