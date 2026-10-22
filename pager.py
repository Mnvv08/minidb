import os
import struct

PAGE_SIZE = 4096
MAGIC = b"MINIDB01"
HEADER_FMT = "<8sII"   # magic (8 bytes), page size (u32), number of pages (u32)


class PagerError(Exception):
    """Raised for corrupt files and invalid page operations."""


class Pager:
    """Reads and writes fixed-size pages in a single database file.

    Page 0 is the header. Data pages are numbered 1..num_pages-1.
    """

    def __init__(self, path: str):
        self.path = path
        is_new = not os.path.exists(path) or os.path.getsize(path) == 0
        self._f = open(path, "w+b" if is_new else "r+b")
        try:
            if is_new:
                self.num_pages = 1
                self._write_header()
                self.flush()
            else:
                self._load_header()
        except Exception:
            self._f.close()
            raise

    # ---------- header ----------

    def _write_header(self):
        header = struct.pack(HEADER_FMT, MAGIC, PAGE_SIZE, self.num_pages)
        self._f.seek(0)
        self._f.write(header.ljust(PAGE_SIZE, b"\x00"))

    def _load_header(self):
        size = os.fstat(self._f.fileno()).st_size
        if size < PAGE_SIZE:
            raise PagerError("file too small to be a minidb database")
        self._f.seek(0)
        magic, page_size, num_pages = struct.unpack_from(HEADER_FMT, self._f.read(PAGE_SIZE))
        if magic != MAGIC:
            raise PagerError("not a minidb file (bad magic number)")
        if page_size != PAGE_SIZE:
            raise PagerError(f"unsupported page size {page_size}")
        if num_pages < 1 or size < num_pages * PAGE_SIZE:
            raise PagerError("file is truncated: header claims more pages than exist")
        self.num_pages = num_pages

    # ---------- pages ----------

    def _check_page(self, page_no: int):
        if not 1 <= page_no < self.num_pages:
            raise PagerError(f"page {page_no} out of range (valid: 1..{self.num_pages - 1})")

    def read_page(self, page_no: int) -> bytes:
        self._check_page(page_no)
        self._f.seek(page_no * PAGE_SIZE)
        return self._f.read(PAGE_SIZE)

    def write_page(self, page_no: int, data: bytes):
        self._check_page(page_no)
        if len(data) != PAGE_SIZE:
            raise PagerError(f"page data must be exactly {PAGE_SIZE} bytes, got {len(data)}")
        self._f.seek(page_no * PAGE_SIZE)
        self._f.write(data)

    def allocate(self) -> int:
        """Add a new zeroed page at the end of the file and return its number."""
        page_no = self.num_pages
        self.num_pages += 1
        self.write_page(page_no, bytes(PAGE_SIZE))
        self._write_header()
        return page_no

    # ---------- lifecycle ----------

    def flush(self):
        """Push everything to disk."""
        self._f.flush()
        os.fsync(self._f.fileno())

    def close(self):
        if not self._f.closed:
            self.flush()
            self._f.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
