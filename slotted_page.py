import struct

from pager import PAGE_SIZE

HEADER_FMT = "<HHI"                          # num_slots, free_end, reserved
HEADER_SIZE = struct.calcsize(HEADER_FMT)    # 8 bytes
SLOT_FMT = "<HH"                             # record offset, record length
SLOT_SIZE = struct.calcsize(SLOT_FMT)        # 4 bytes
MAX_RECORD = PAGE_SIZE - HEADER_SIZE - SLOT_SIZE


class PageError(Exception):
    """Base class for slotted-page errors."""


class PageFullError(PageError):
    """The record does not fit in this page."""


class SlotError(PageError):
    """Invalid slot number, or the slot has been deleted."""


class CorruptPageError(PageError):
    """The bytes do not form a valid slotted page."""


class SlottedPage:
    """One 4 KB page holding variable-length records addressed by slot number.

    Layout:  [header][slot array ->] ...free space... [<- record data]
    A slot with offset 0 is a tombstone (deleted); its number can be reused.
    """

    def __init__(self, data=None):
        if data is None:
            self._buf = bytearray(PAGE_SIZE)
            self._write_header(0, PAGE_SIZE)
        else:
            if len(data) != PAGE_SIZE:
                raise CorruptPageError(f"page must be {PAGE_SIZE} bytes, got {len(data)}")
            self._buf = bytearray(data)
            self._validate()

    # ---------- raw header / slot access ----------

    def _header(self):
        return struct.unpack_from(HEADER_FMT, self._buf, 0)

    def _write_header(self, num_slots, free_end):
        reserved = self._header()[2]
        struct.pack_into(HEADER_FMT, self._buf, 0, num_slots, free_end, reserved)

    @property
    def num_slots(self):
        return self._header()[0]

    @property
    def free_end(self):
        return self._header()[1]

    @property
    def reserved(self):
        """A 32-bit field for the layer above (page type, next-page pointer...)."""
        return self._header()[2]

    @reserved.setter
    def reserved(self, value):
        n, free_end, _ = self._header()
        struct.pack_into(HEADER_FMT, self._buf, 0, n, free_end, value)

    def _slot(self, i):
        return struct.unpack_from(SLOT_FMT, self._buf, HEADER_SIZE + i * SLOT_SIZE)

    def _set_slot(self, i, offset, length):
        struct.pack_into(SLOT_FMT, self._buf, HEADER_SIZE + i * SLOT_SIZE, offset, length)

    def _check_slot(self, i):
        if not 0 <= i < self.num_slots:
            raise SlotError(f"slot {i} out of range (page has {self.num_slots} slots)")

    def _validate(self):
        n, free_end, _ = self._header()
        if not (HEADER_SIZE + n * SLOT_SIZE <= free_end <= PAGE_SIZE):
            raise CorruptPageError("invalid header (not an initialized slotted page)")
        for i in range(n):
            offset, length = self._slot(i)
            if offset == 0:
                continue
            if offset < free_end or offset + length > PAGE_SIZE:
                raise CorruptPageError(f"slot {i} points outside the record area")

    # ---------- space accounting ----------

    def free_space(self) -> int:
        """Contiguous bytes between the slot array and the record area."""
        n, free_end, _ = self._header()
        return free_end - (HEADER_SIZE + n * SLOT_SIZE)

    def total_free(self) -> int:
        """Contiguous free space plus dead bytes that compaction would reclaim."""
        n, free_end, _ = self._header()
        live_bytes = 0
        for i in range(n):
            offset, length = self._slot(i)
            if offset:
                live_bytes += length
        dead = (PAGE_SIZE - free_end) - live_bytes
        return self.free_space() + dead

    # ---------- records ----------

    def _first_free_slot(self):
        for i in range(self.num_slots):
            if self._slot(i)[0] == 0:
                return i
        return None

    def _place(self, slot, record):
        end = self.free_end - len(record)
        self._buf[end:end + len(record)] = record
        self._set_slot(slot, end, len(record))
        self._write_header(self.num_slots, end)

    def insert(self, record) -> int:
        """Store a record and return its slot number."""
        if not isinstance(record, (bytes, bytearray)):
            raise TypeError("record must be bytes")
        slot = self._first_free_slot()
        need = len(record) + (0 if slot is not None else SLOT_SIZE)
        if need > self.free_space():
            if need > self.total_free():
                raise PageFullError(f"record of {len(record)} bytes does not fit")
            self.compact()      # only runs when it is guaranteed to make room
        if slot is None:
            slot = self.num_slots
            self._write_header(slot + 1, self.free_end)
            self._set_slot(slot, 0, 0)
        self._place(slot, record)
        return slot

    def get(self, slot):
        """Return the record's bytes, or None if the slot was deleted."""
        self._check_slot(slot)
        offset, length = self._slot(slot)
        if offset == 0:
            return None
        return bytes(self._buf[offset:offset + length])

    def delete(self, slot) -> bool:
        """Delete a record. Returns False if it was already deleted."""
        self._check_slot(slot)
        if self._slot(slot)[0] == 0:
            return False
        self._set_slot(slot, 0, 0)
        return True

    def update(self, slot, record):
        """Replace a record, keeping its slot number. Page is unchanged on failure."""
        if not isinstance(record, (bytes, bytearray)):
            raise TypeError("record must be bytes")
        self._check_slot(slot)
        offset, old_len = self._slot(slot)
        if offset == 0:
            raise SlotError(f"slot {slot} is deleted")
        if len(record) <= old_len:                     # fits where it is
            self._buf[offset:offset + len(record)] = record
            self._set_slot(slot, offset, len(record))
            return
        if len(record) - old_len > self.total_free():
            raise PageFullError("updated record does not fit")
        self._set_slot(slot, 0, 0)                     # free the old copy first
        if len(record) > self.free_space():
            self.compact()
        self._place(slot, record)

    def compact(self):
        """Squeeze out dead bytes. Slot numbers do not change."""
        n, _, reserved = self._header()
        new = bytearray(PAGE_SIZE)
        end = PAGE_SIZE
        for i in range(n):
            record = self.get(i)
            if record is None:
                continue                               # stays a tombstone (0, 0)
            end -= len(record)
            new[end:end + len(record)] = record
            struct.pack_into(SLOT_FMT, new, HEADER_SIZE + i * SLOT_SIZE, end, len(record))
        struct.pack_into(HEADER_FMT, new, 0, n, end, reserved)
        self._buf = new

    def records(self):
        """Yield (slot, record) for every live record, in slot order."""
        for i in range(self.num_slots):
            record = self.get(i)
            if record is not None:
                yield i, record

    def num_records(self) -> int:
        return sum(1 for _ in self.records())

    def to_bytes(self) -> bytes:
        return bytes(self._buf)
