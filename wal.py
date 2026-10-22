"""Write-ahead logging for the Pager.

Every change happens inside a transaction. While a transaction is open, modified pages are
only kept in memory. On commit, all of them are appended to the log file as ONE record and
the log is fsynced; only then is the commit acknowledged. The database file itself is updated
later, at a checkpoint, and the log is emptied afterwards.

If the process dies at any moment, the next start replays every complete record in the log.
A record is complete only if its length, its CRC-32 and its commit marker are all intact,
so a half-written record is simply ignored: a transaction is either fully there or not at all.

Log record:   header(magic "WTXN", txn id, page count) | (page no, 4096 bytes)... | crc32, "CMIT"
"""
import os
import struct
import zlib

from pager import HEADER_FMT, MAGIC, PAGE_SIZE, Pager, PagerError

WAL_SUFFIX = "-wal"
TXN_MAGIC = b"WTXN"
COMMIT_MAGIC = b"CMIT"
TXN_HEADER = struct.Struct("<4sII")      # magic, transaction id, number of pages
PAGE_NO = struct.Struct("<I")
TXN_TRAILER = struct.Struct("<I4s")      # crc32 of everything before it, commit marker
FRAME_SIZE = PAGE_NO.size + PAGE_SIZE


def encode_txn(txn_id, pages):
    """Serialize {page_no: bytes} into one self-checking log record."""
    body = TXN_HEADER.pack(TXN_MAGIC, txn_id, len(pages))
    body += b"".join(PAGE_NO.pack(n) + data for n, data in sorted(pages.items()))
    return body + TXN_TRAILER.pack(zlib.crc32(body), COMMIT_MAGIC)


def parse_wal(data):
    """Return (transactions, good_end).

    transactions is a list of (txn_id, [(page_no, bytes), ...]) for every complete record,
    in order. Parsing stops at the first record that is incomplete or fails its checks.
    """
    txns, pos = [], 0
    while pos + TXN_HEADER.size <= len(data):
        magic, txn_id, n = TXN_HEADER.unpack_from(data, pos)
        if magic != TXN_MAGIC or n == 0:
            break
        body_len = TXN_HEADER.size + n * FRAME_SIZE
        end = pos + body_len + TXN_TRAILER.size
        if end > len(data):
            break                                       # torn tail: the record was cut short
        crc, commit = TXN_TRAILER.unpack_from(data, pos + body_len)
        if commit != COMMIT_MAGIC or crc != zlib.crc32(data[pos:pos + body_len]):
            break
        pages, off = [], pos + TXN_HEADER.size
        for _ in range(n):
            page_no = PAGE_NO.unpack_from(data, off)[0]
            pages.append((page_no, data[off + PAGE_NO.size:off + FRAME_SIZE]))
            off += FRAME_SIZE
        txns.append((txn_id, pages))
        pos = end
    return txns, pos


class WALPager(Pager):
    """A Pager whose writes are transactional and crash-safe.

    Writes outside begin()/commit() are applied as tiny transactions of their own.
    Page 0 (the file header) is managed here: it is logged whenever the page count changes.
    """

    DEFAULT_SYNC = True        # tests switch this off to run faster

    def __init__(self, path, sync=None, checkpoint_bytes=1 << 20, fault_hook=None):
        self.sync = self.DEFAULT_SYNC if sync is None else sync
        self.checkpoint_bytes = checkpoint_bytes
        self.fault_hook = fault_hook        # test hook: called with an event name, may raise
        self._wal_path = path + WAL_SUFFIX
        self._wal = None                    # file descriptor of the log
        self._wal_size = 0
        self._wal_pages = {}                # committed, not yet in the database file
        self._txn_pages = None              # None = no open transaction
        self._txn_start_pages = 0
        self._next_txn = 1
        self.recovered_txns = 0
        self.checkpoints = 0
        self._recover(path)
        super().__init__(path)
        try:
            self._wal = os.open(self._wal_path, os.O_RDWR | os.O_CREAT | os.O_APPEND, 0o644)
            self._wal_size = os.fstat(self._wal).st_size
        except Exception:
            Pager.close(self)
            raise

    def _fault(self, event):
        if self.fault_hook is not None:
            self.fault_hook(event)

    def _fsync(self, fd):
        if self.sync:
            os.fsync(fd)

    # ---------- recovery ----------

    def _recover(self, path):
        """Replay committed transactions from a log left behind by a crash, then empty the log."""
        if not os.path.exists(self._wal_path):
            return
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            os.remove(self._wal_path)       # a log without its database file is meaningless
            return
        with open(self._wal_path, "rb") as f:
            data = f.read()
        txns, _ = parse_wal(data)
        if txns:
            final = {}
            for _, pages in txns:
                for page_no, content in pages:
                    final[page_no] = content        # later transactions win
            with open(path, "r+b") as db:
                for page_no in sorted(final):
                    db.seek(page_no * PAGE_SIZE)
                    db.write(final[page_no])
                db.flush()
                os.fsync(db.fileno())
            self.recovered_txns = len(txns)
        self._fault("recover_applied")
        fd = os.open(self._wal_path, os.O_WRONLY)   # anything after the last good record is garbage
        try:
            os.ftruncate(fd, 0)
            os.fsync(fd)
        finally:
            os.close(fd)

    # ---------- transactions ----------

    def begin(self):
        if self._txn_pages is not None:
            raise PagerError("a transaction is already open")
        self._txn_pages = {}
        self._txn_start_pages = self.num_pages

    def rollback(self):
        """Throw away everything done since begin()."""
        if self._txn_pages is None:
            raise PagerError("no open transaction")
        self._txn_pages = None
        self.num_pages = self._txn_start_pages

    def commit(self):
        """Make the transaction durable: one log record, one fsync."""
        if self._txn_pages is None:
            raise PagerError("no open transaction")
        pages = self._txn_pages
        if pages:
            if self.num_pages != self._txn_start_pages:
                pages[0] = self._header_page()
            record = encode_txn(self._next_txn, pages)
            start = self._wal_size
            self._fault("wal_write")
            try:
                self._append(record)
            except OSError:
                self._undo_append(start)
                raise
            self._fault("wal_written")
            try:
                self._fsync(self._wal)
            except OSError:
                self._undo_append(start)
                raise
            self._fault("wal_synced")
            self._wal_size = start + len(record)
            self._next_txn += 1
            self._wal_pages.update(pages)
        self._txn_pages = None
        if self._wal_size >= self.checkpoint_bytes:
            self.checkpoint()

    def _append(self, record):
        view = memoryview(record)
        while view:
            view = view[os.write(self._wal, view):]

    def _undo_append(self, size):
        """After a failed write, cut the log back so later records stay reachable."""
        try:
            os.ftruncate(self._wal, size)
        except OSError:
            pass

    def _header_page(self):
        return struct.pack(HEADER_FMT, MAGIC, PAGE_SIZE, self.num_pages).ljust(PAGE_SIZE, b"\x00")

    def _autocommit(self, action):
        self.begin()
        try:
            result = action()
            self.commit()
        except BaseException:
            if self._txn_pages is not None:
                self.rollback()
            raise
        return result

    # ---------- pages ----------

    def read_page(self, page_no):
        self._check_page(page_no)
        if self._txn_pages is not None and page_no in self._txn_pages:
            return self._txn_pages[page_no]
        if page_no in self._wal_pages:
            return self._wal_pages[page_no]
        return super().read_page(page_no)

    def write_page(self, page_no, data):
        self._check_page(page_no)
        if len(data) != PAGE_SIZE:
            raise PagerError(f"page data must be exactly {PAGE_SIZE} bytes, got {len(data)}")
        if self._txn_pages is None:
            self._autocommit(lambda: self.write_page(page_no, data))
        else:
            self._txn_pages[page_no] = bytes(data)

    def allocate(self):
        if self._txn_pages is None:
            return self._autocommit(self.allocate)
        page_no = self.num_pages
        self.num_pages += 1
        self._txn_pages[page_no] = bytes(PAGE_SIZE)
        return page_no

    # ---------- checkpoint and lifecycle ----------

    def checkpoint(self):
        """Copy committed pages into the database file, fsync it, then empty the log."""
        if self._txn_pages is not None:
            raise PagerError("cannot checkpoint while a transaction is open")
        if self._wal is None or (not self._wal_pages and self._wal_size == 0):
            return
        for page_no in sorted(self._wal_pages):
            self._fault("db_write")
            self._f.seek(page_no * PAGE_SIZE)
            self._f.write(self._wal_pages[page_no])
        self._f.flush()
        self._fsync(self._f.fileno())
        self._fault("db_synced")
        os.ftruncate(self._wal, 0)
        self._fsync(self._wal)
        self._fault("wal_truncated")
        self._wal_pages.clear()
        self._wal_size = 0
        self.checkpoints += 1

    def flush(self):
        if self._wal is not None and self._txn_pages is None:
            self.checkpoint()
        super().flush()

    def close(self):
        if self._f.closed:
            return
        if self._txn_pages is not None:
            self.rollback()
        if self._wal is not None:
            self.checkpoint()
            os.close(self._wal)
            self._wal = None
            try:
                os.remove(self._wal_path)       # everything is in the database file now
            except FileNotFoundError:
                pass
        super().close()
