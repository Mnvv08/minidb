import os

import pytest

from pager import PAGE_SIZE, Pager, PagerError


def test_new_file_has_only_header_page(tmp_path):
    path = tmp_path / "t.db"
    with Pager(str(path)) as p:
        assert p.num_pages == 1
    assert os.path.getsize(path) == PAGE_SIZE


def test_allocate_returns_sequential_pages(tmp_path):
    with Pager(str(tmp_path / "t.db")) as p:
        assert p.allocate() == 1
        assert p.allocate() == 2
        assert p.allocate() == 3
        assert p.num_pages == 4


def test_write_read_roundtrip(tmp_path):
    with Pager(str(tmp_path / "t.db")) as p:
        n = p.allocate()
        data = b"hello pager".ljust(PAGE_SIZE, b"\x00")
        p.write_page(n, data)
        assert p.read_page(n) == data


def test_new_page_is_zeroed(tmp_path):
    with Pager(str(tmp_path / "t.db")) as p:
        n = p.allocate()
        assert p.read_page(n) == bytes(PAGE_SIZE)


def test_data_persists_after_reopen(tmp_path):
    path = str(tmp_path / "t.db")
    data = b"persist me".ljust(PAGE_SIZE, b"\x00")
    with Pager(path) as p:
        n = p.allocate()
        p.write_page(n, data)
    with Pager(path) as p:
        assert p.num_pages == 2
        assert p.read_page(n) == data


def test_bad_magic_rejected(tmp_path):
    path = tmp_path / "t.db"
    path.write_bytes(b"X" * PAGE_SIZE)
    with pytest.raises(PagerError, match="bad magic"):
        Pager(str(path))


def test_file_smaller_than_one_page_rejected(tmp_path):
    path = tmp_path / "t.db"
    path.write_bytes(b"short")
    with pytest.raises(PagerError, match="too small"):
        Pager(str(path))


def test_header_claiming_missing_pages_rejected(tmp_path):
    path = tmp_path / "t.db"
    with Pager(str(path)) as p:
        p.allocate()
    os.truncate(path, PAGE_SIZE)       # header says 2 pages, file only has 1
    with pytest.raises(PagerError, match="truncated"):
        Pager(str(path))


def test_write_wrong_size_rejected(tmp_path):
    with Pager(str(tmp_path / "t.db")) as p:
        n = p.allocate()
        with pytest.raises(PagerError):
            p.write_page(n, b"too short")


def test_out_of_range_pages_rejected(tmp_path):
    with Pager(str(tmp_path / "t.db")) as p:
        p.allocate()
        with pytest.raises(PagerError):
            p.read_page(0)                      # page 0 is the reserved header
        with pytest.raises(PagerError):
            p.read_page(5)
        with pytest.raises(PagerError):
            p.write_page(0, bytes(PAGE_SIZE))


def test_context_manager_closes_file(tmp_path):
    with Pager(str(tmp_path / "t.db")) as p:
        pass
    assert p._f.closed
