import random

import pytest

from btree import (
    KEY_MAX, KEY_MIN, MAX_INTERNAL_KEYS, BTree, CorruptTreeError, DuplicateKeyError,
)
from pager import PAGE_SIZE, Pager


def small(path):
    """A tree with tiny nodes, so a few dozen keys already build several levels."""
    return BTree(str(path), leaf_capacity=100, max_internal_keys=4)


def val(k):
    return b"v" + str(k).encode()


def test_new_tree_is_empty(tmp_path):
    with small(tmp_path / "t.db") as t:
        assert t.get(1) is None
        assert list(t.items()) == []
        assert t.height() == 1
        assert t.validate() == 0


def test_insert_and_get(tmp_path):
    with small(tmp_path / "t.db") as t:
        t.insert(5, b"five")
        t.insert(1, b"one")
        t.insert(9, b"nine")
        assert t.get(1) == b"one"
        assert t.get(5) == b"five"
        assert t.get(9) == b"nine"
        assert list(t.items()) == [(1, b"one"), (5, b"five"), (9, b"nine")]


def test_get_missing_and_contains(tmp_path):
    with small(tmp_path / "t.db") as t:
        t.insert(5, b"five")
        assert 5 in t
        assert 6 not in t
        assert t.get(6) is None


def test_duplicate_insert_rejected_and_put_replaces(tmp_path):
    with small(tmp_path / "t.db") as t:
        t.insert(1, b"a")
        with pytest.raises(DuplicateKeyError):
            t.insert(1, b"b")
        assert t.get(1) == b"a"
        t.put(1, b"b")
        assert t.get(1) == b"b"
        t.put(2, b"c")
        assert t.count() == 2


def test_root_split_grows_height(tmp_path):
    with small(tmp_path / "t.db") as t:
        old_root = t.root
        for k in range(1, 8):
            t.insert(k, val(k))
        assert t.height() == 1
        assert t.root == old_root
        t.insert(8, val(8))
        assert t.height() == 2
        assert t.root != old_root
        assert t.validate() == 8


def test_sequential_inserts_build_deep_tree(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in range(300):
            t.insert(k, val(k))
        assert t.validate() == 300
        assert t.height() >= 4
        assert [k for k, _ in t.items()] == list(range(300))
        assert all(t.get(k) == val(k) for k in range(300))


def test_reverse_inserts(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in range(299, -1, -1):
            t.insert(k, val(k))
        assert t.validate() == 300
        assert [k for k, _ in t.items()] == list(range(300))


def test_random_inserts_match_model(tmp_path):
    rng = random.Random(42)
    model = {}
    with small(tmp_path / "t.db") as t:
        for i in range(2000):
            k = rng.randrange(-5000, 5000)
            v = rng.randbytes(rng.randrange(0, 14))
            if k in model and rng.random() < 0.5:
                with pytest.raises(DuplicateKeyError):
                    t.insert(k, v)
            else:
                t.put(k, v)
                model[k] = v
            if i % 100 == 0:
                assert t.validate() == len(model)
        assert t.validate() == len(model)
        assert list(t.items()) == sorted(model.items())
        for k, v in model.items():
            assert t.get(k) == v


def test_replace_with_larger_value_can_split(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in range(1, 8):
            t.insert(k, val(k))
        assert t.height() == 1
        t.put(4, b"x" * 13)           # leaf grows from 91 to 102 bytes, over its 100-byte limit
        assert t.height() == 2
        assert t.get(4) == b"x" * 13
        assert t.validate() == 7


def test_default_page_size_bulk(tmp_path):
    with BTree(str(tmp_path / "t.db")) as t:
        for k in range(3000):
            t.insert(k, bytes([k % 256]) * 100)
        assert t.validate() == 3000
        assert t.height() >= 2
        assert t.get(1234) == bytes([1234 % 256]) * 100
        assert t.get(3000) is None


def test_max_size_values_split_correctly(tmp_path):
    rng = random.Random(7)
    keys = list(range(60))
    rng.shuffle(keys)
    with BTree(str(tmp_path / "t.db")) as t:
        big = b"x" * t.max_value
        for k in keys:
            t.insert(k, big)
        assert t.validate() == 60
        assert t.height() >= 2
        assert all(t.get(k) == big for k in range(60))


def test_persistence_across_reopen(tmp_path):
    path = tmp_path / "t.db"
    with small(path) as t:
        for k in range(200):
            t.insert(k, val(k))
    with small(path) as t:
        assert t.validate() == 200
        assert t.get(150) == val(150)
        t.insert(1000, b"new")
        assert t.validate() == 201
    with small(path) as t:
        assert t.get(1000) == b"new"


def test_extreme_keys(tmp_path):
    keys = [KEY_MAX, 0, KEY_MIN, 1, -1]
    with small(tmp_path / "t.db") as t:
        for i, k in enumerate(keys):
            t.insert(k, bytes([i]))
        assert [k for k, _ in t.items()] == sorted(keys)
        for i, k in enumerate(keys):
            assert t.get(k) == bytes([i])


def test_key_and_value_validation(tmp_path):
    with small(tmp_path / "t.db") as t:
        with pytest.raises(TypeError):
            t.insert("1", b"x")
        with pytest.raises(TypeError):
            t.insert(True, b"x")
        with pytest.raises(TypeError):
            t.insert(1.5, b"x")
        with pytest.raises(ValueError):
            t.insert(KEY_MAX + 1, b"x")
        with pytest.raises(ValueError):
            t.insert(KEY_MIN - 1, b"x")
        with pytest.raises(TypeError):
            t.insert(1, "text")
        with pytest.raises(ValueError):
            t.insert(1, b"x" * (t.max_value + 1))
        t.insert(1, b"x" * t.max_value)
        with pytest.raises(TypeError):
            t.get("1")
        assert t.count() == 1


def test_constructor_validation(tmp_path):
    with pytest.raises(ValueError):
        BTree(str(tmp_path / "a.db"), leaf_capacity=10)
    with pytest.raises(ValueError):
        BTree(str(tmp_path / "b.db"), leaf_capacity=PAGE_SIZE + 1)
    with pytest.raises(ValueError):
        BTree(str(tmp_path / "c.db"), max_internal_keys=2)
    with pytest.raises(ValueError):
        BTree(str(tmp_path / "d.db"), max_internal_keys=MAX_INTERNAL_KEYS + 1)
    assert not (tmp_path / "a.db").exists()


def test_corrupt_meta_rejected(tmp_path):
    path = tmp_path / "t.db"
    with small(path) as t:
        t.insert(1, b"a")
    with Pager(str(path)) as p:
        p.write_page(1, bytes(PAGE_SIZE))
    with pytest.raises(CorruptTreeError, match="magic"):
        small(path)


def test_corrupt_node_rejected(tmp_path):
    path = tmp_path / "t.db"
    with small(path) as t:
        t.insert(1, b"a")
        root = t.root
    with Pager(str(path)) as p:
        p.write_page(root, bytes(PAGE_SIZE))
    with small(path) as t:
        with pytest.raises(CorruptTreeError, match="node type"):
            t.get(1)


def test_path_to_length_equals_height(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in range(100):
            t.insert(k, val(k))
        h = t.height()
        assert h >= 3
        for k in (0, 50, 99, 12345):
            path = t.path_to(k)
            assert len(path) == h
            assert path[0] == t.root


def test_max_value_for_default_capacity(tmp_path):
    with BTree(str(tmp_path / "t.db")) as t:
        assert t.max_value == 1012


def test_validate_detects_a_broken_tree(tmp_path):
    with small(tmp_path / "t.db") as t:
        for k in range(8):
            t.insert(k, val(k))
        assert t.validate() == 8
        leaf_page = t.path_to(0)[-1]
        node = t._read_node(leaf_page)
        node.keys.reverse()
        node.values.reverse()
        t._write_node(leaf_page, node)
        with pytest.raises(CorruptTreeError):
            t.validate()
