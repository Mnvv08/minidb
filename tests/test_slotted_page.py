import random
import struct

import pytest

from pager import PAGE_SIZE, Pager
from slotted_page import (
    HEADER_SIZE, MAX_RECORD, SLOT_FMT, SLOT_SIZE,
    CorruptPageError, PageFullError, SlotError, SlottedPage,
)


def fill(page, size=100):
    """Insert size-byte records until the page is full; return their slot numbers."""
    slots = []
    while True:
        try:
            slots.append(page.insert(bytes([len(slots) % 251]) * size))
        except PageFullError:
            return slots


def test_empty_page_layout():
    page = SlottedPage()
    assert page.num_slots == 0
    assert page.free_space() == PAGE_SIZE - HEADER_SIZE
    assert page.total_free() == PAGE_SIZE - HEADER_SIZE
    assert len(page.to_bytes()) == PAGE_SIZE
    assert list(page.records()) == []


def test_insert_get_roundtrip_and_sequential_slots():
    page = SlottedPage()
    assert page.insert(b"alice") == 0
    assert page.insert(b"bob") == 1
    assert page.insert(b"carol") == 2
    assert page.get(0) == b"alice"
    assert page.get(1) == b"bob"
    assert page.get(2) == b"carol"
    assert page.num_records() == 3


def test_get_deleted_returns_none_and_bad_slot_raises():
    page = SlottedPage()
    page.insert(b"x")
    page.delete(0)
    assert page.get(0) is None
    with pytest.raises(SlotError):
        page.get(1)
    with pytest.raises(SlotError):
        page.get(-1)


def test_delete_returns_true_then_false():
    page = SlottedPage()
    page.insert(b"x")
    assert page.delete(0) is True
    assert page.delete(0) is False
    with pytest.raises(SlotError):
        page.delete(5)


def test_insert_reuses_deleted_slot():
    page = SlottedPage()
    for r in (b"a", b"b", b"c"):
        page.insert(r)
    page.delete(1)
    assert page.insert(b"new") == 1
    assert page.get(0) == b"a"
    assert page.get(2) == b"c"
    assert page.num_slots == 3


def test_page_full_raises_and_leaves_page_unchanged():
    page = SlottedPage()
    slots = fill(page, 100)
    assert len(slots) == (PAGE_SIZE - HEADER_SIZE) // (100 + SLOT_SIZE)
    before = page.to_bytes()
    with pytest.raises(PageFullError):
        page.insert(b"y" * 100)
    assert page.to_bytes() == before


def test_space_is_reused_after_delete():
    page = SlottedPage()
    slots = fill(page, 100)
    page.delete(slots[3])
    assert page.free_space() < 100           # no contiguous room...
    assert page.total_free() >= 100          # ...but the hole counts
    assert page.insert(b"z" * 100) == slots[3]   # compaction makes it fit
    assert page.get(slots[3]) == b"z" * 100
    for s in slots:
        if s != slots[3]:
            assert page.get(s) == bytes([s % 251]) * 100


def test_compact_preserves_ids_and_reclaims_space():
    page = SlottedPage()
    page.insert(b"a" * 100)
    page.insert(b"b" * 200)
    page.insert(b"c" * 300)
    page.delete(1)
    before = page.free_space()
    page.compact()
    assert page.free_space() == before + 200
    assert page.get(0) == b"a" * 100
    assert page.get(1) is None
    assert page.get(2) == b"c" * 300


def test_update_shrink_in_place():
    page = SlottedPage()
    page.insert(b"aaaa")
    page.update(0, b"bb")
    assert page.get(0) == b"bb"
    assert page.total_free() == PAGE_SIZE - HEADER_SIZE - SLOT_SIZE - 2


def test_update_grow_keeps_slot_id():
    page = SlottedPage()
    page.insert(b"first")
    page.insert(b"second")
    page.update(0, b"x" * 500)
    assert page.get(0) == b"x" * 500
    assert page.get(1) == b"second"
    assert page.num_slots == 2


def test_update_too_big_fails_cleanly():
    page = SlottedPage()
    slots = fill(page, 100)
    before = page.to_bytes()
    with pytest.raises(PageFullError):
        page.update(slots[0], b"x" * 200)
    assert page.to_bytes() == before


def test_update_that_needs_compaction():
    page = SlottedPage()
    slots = fill(page, 100)
    page.delete(slots[5])
    page.update(slots[0], b"x" * 150)
    assert page.get(slots[0]) == b"x" * 150
    for s in slots[1:]:
        if s != slots[5]:
            assert page.get(s) == bytes([s % 251]) * 100


def test_update_deleted_or_missing_slot_raises():
    page = SlottedPage()
    page.insert(b"a")
    page.delete(0)
    with pytest.raises(SlotError):
        page.update(0, b"b")
    with pytest.raises(SlotError):
        page.update(9, b"b")


def test_max_record_size():
    page = SlottedPage()
    assert page.insert(b"x" * MAX_RECORD) == 0
    assert page.free_space() == 0
    assert page.get(0) == b"x" * MAX_RECORD
    with pytest.raises(PageFullError):
        SlottedPage().insert(b"x" * (MAX_RECORD + 1))


def test_zero_length_record_is_not_deleted():
    page = SlottedPage()
    assert page.insert(b"") == 0
    assert page.get(0) == b""
    assert page.get(0) is not None
    page.delete(0)
    assert page.get(0) is None


def test_insert_rejects_non_bytes():
    with pytest.raises(TypeError):
        SlottedPage().insert("text")


def test_serialization_roundtrip():
    page = SlottedPage()
    page.insert(b"alice")
    page.insert(b"bob")
    page.insert(b"carol")
    page.delete(1)
    copy = SlottedPage(page.to_bytes())
    assert list(copy.records()) == [(0, b"alice"), (2, b"carol")]
    assert copy.insert(b"dave") == 1


def test_uninitialized_zero_page_rejected():
    with pytest.raises(CorruptPageError):
        SlottedPage(bytes(PAGE_SIZE))


def test_corrupt_slot_rejected():
    page = SlottedPage()
    page.insert(b"hello")
    raw = bytearray(page.to_bytes())
    struct.pack_into(SLOT_FMT, raw, HEADER_SIZE, 5000, 10)   # points past the page
    with pytest.raises(CorruptPageError):
        SlottedPage(bytes(raw))


def test_wrong_size_rejected():
    with pytest.raises(CorruptPageError):
        SlottedPage(b"short")


def test_reserved_field_survives_compaction_and_serialization():
    page = SlottedPage()
    page.reserved = 42
    page.insert(b"a" * 10)
    page.insert(b"b" * 10)
    page.delete(0)
    page.compact()
    assert page.reserved == 42
    assert SlottedPage(page.to_bytes()).reserved == 42


def test_stored_through_pager(tmp_path):
    path = str(tmp_path / "t.db")
    page = SlottedPage()
    page.insert(b"hello")
    page.insert(b"world")
    with Pager(path) as p:
        n = p.allocate()
        p.write_page(n, page.to_bytes())
    with Pager(path) as p:
        loaded = SlottedPage(p.read_page(n))
    assert list(loaded.records()) == [(0, b"hello"), (1, b"world")]


def test_random_operations_match_model():
    rng = random.Random(1234)
    page = SlottedPage()
    model = {}
    for _ in range(3000):
        op = rng.choice(["insert", "insert", "delete", "update"])
        if op == "insert":
            record = rng.randbytes(rng.randrange(0, 300))
            try:
                slot = page.insert(record)
            except PageFullError:
                continue
            assert slot not in model
            model[slot] = record
        elif op == "delete" and model:
            slot = rng.choice(list(model))
            assert page.delete(slot) is True
            del model[slot]
        elif op == "update" and model:
            slot = rng.choice(list(model))
            record = rng.randbytes(rng.randrange(0, 400))
            try:
                page.update(slot, record)
            except PageFullError:
                continue
            model[slot] = record
        assert page.free_space() >= 0
        assert dict(page.records()) == model
        # the raw bytes must always be a valid page that holds the same data
        assert dict(SlottedPage(page.to_bytes()).records()) == model
