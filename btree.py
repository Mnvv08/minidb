"""A disk-based B+ tree on top of the Pager.

File layout:   page 1 = meta (root page, free-list head), page 2.. = tree nodes and free pages.
Leaf node:     header | (key, value_len, value)...      next-leaf pointer in header
Internal node: header | child0 | (key, child)...        keys[i] = smallest key in children[i+1]
Free page:     header only; the "next" field links to the next free page (0 = end of list)
Keys are signed 64-bit integers; values are bytes.

Balance rules (every node except the root):
  leaf      >= leaf_capacity // 4 bytes used
  internal  >= max_internal_keys // 2 keys
"""
import bisect
import contextlib
import struct

from pager import PAGE_SIZE
from wal import WALPager

LEAF, INTERNAL, FREE = 1, 2, 3
NODE_HEADER = struct.Struct("<BHI")      # node type, number of keys, next page (leaf chain / free list)
HEADER_SIZE = NODE_HEADER.size           # 7 bytes
LEAF_CELL = struct.Struct("<qH")         # key, value length
CELL_OVERHEAD = LEAF_CELL.size           # 10 bytes
INTERNAL_ENTRY = struct.Struct("<qI")    # key, right child page
CHILD = struct.Struct("<I")
MAX_INTERNAL_KEYS = (PAGE_SIZE - HEADER_SIZE - CHILD.size) // INTERNAL_ENTRY.size   # 340

META_PAGE = 1
META = struct.Struct("<8sII")            # magic, root page number, free-list head (0 = empty)
META_MAGIC = b"BTREE001"

KEY_MIN = -(2 ** 63)
KEY_MAX = 2 ** 63 - 1


class BTreeError(Exception):
    """Base class for B+ tree errors."""


class DuplicateKeyError(BTreeError):
    def __init__(self, key):
        super().__init__(f"key {key} already exists")
        self.key = key


class CorruptTreeError(BTreeError):
    """The file does not contain a valid B+ tree."""


class Node:
    __slots__ = ("kind", "keys", "values", "children", "next_leaf")

    def __init__(self, kind):
        self.kind = kind
        self.keys = []
        self.values = []       # leaf only
        self.children = []     # internal only: len(keys) + 1 page numbers
        self.next_leaf = 0     # leaf only: page number of the next leaf, 0 = none


class BTree:
    def __init__(self, path, leaf_capacity=PAGE_SIZE, max_internal_keys=MAX_INTERNAL_KEYS,
                 pager_options=None):
        # Capacities are not stored in the file: reopen a file with the values it was built with.
        if not 64 <= leaf_capacity <= PAGE_SIZE:
            raise ValueError(f"leaf_capacity must be between 64 and {PAGE_SIZE}")
        if not 3 <= max_internal_keys <= MAX_INTERNAL_KEYS:
            raise ValueError(f"max_internal_keys must be between 3 and {MAX_INTERNAL_KEYS}")
        self.leaf_capacity = leaf_capacity
        self.max_internal_keys = max_internal_keys
        self.min_leaf_bytes = leaf_capacity // 4
        self.min_internal_keys = max_internal_keys // 2
        # Largest value for which a leaf always holds >= 4 cells, so splits always work
        self.max_value = (leaf_capacity - HEADER_SIZE) // 4 - CELL_OVERHEAD
        self.page_reads = 0            # how many pages were read (handy for showing costs)
        self.free_head = 0
        self._meta_dirty = False
        self._in_txn = False
        self.pager = WALPager(path, **(pager_options or {}))
        try:
            if self.pager.num_pages == 1:
                self._create()
            else:
                self._load_meta()
        except Exception:
            self.pager.close()
            raise

    # ---------- meta page ----------

    def _create(self):
        with self.transaction():
            if self.pager.allocate() != META_PAGE:
                raise BTreeError("unexpected page layout in a new file")
            root = self.pager.allocate()
            self._write_node(root, Node(LEAF))
            self.root = root
            self._write_meta()
        self.pager.flush()

    # ---------- transactions ----------

    @contextlib.contextmanager
    def transaction(self):
        """Group operations into one atomic, durable unit.

        Everything inside the block commits together with a single log write and fsync, or
        (on an exception, or a crash) not at all. Single insert/put/delete calls are
        transactions of their own; inside a block they simply join it.
        """
        if self._in_txn:
            yield
            return
        self.pager.begin()
        self._in_txn = True
        saved = (getattr(self, "root", None), self.free_head)
        try:
            yield
            self._save_meta_if_dirty()
            self.pager.commit()
        except BaseException:
            if self.pager._txn_pages is not None:
                self.pager.rollback()
            self.root, self.free_head = saved       # undo the in-memory state too
            self._meta_dirty = False
            raise
        finally:
            self._in_txn = False

    def checkpoint(self):
        """Copy committed changes into the database file and empty the log."""
        self.pager.checkpoint()

    def _write_meta(self):
        page = META.pack(META_MAGIC, self.root, self.free_head).ljust(PAGE_SIZE, b"\x00")
        self.pager.write_page(META_PAGE, page)
        self._meta_dirty = False

    def _save_meta_if_dirty(self):
        if self._meta_dirty:
            self._write_meta()

    def _load_meta(self):
        if self.pager.num_pages < 3:
            raise CorruptTreeError("file has no B+ tree (missing meta or root page)")
        magic, root, free_head = META.unpack_from(self.pager.read_page(META_PAGE), 0)
        if magic != META_MAGIC:
            raise CorruptTreeError("bad B+ tree meta page (wrong magic number)")
        if not 2 <= root < self.pager.num_pages:
            raise CorruptTreeError(f"root page {root} out of range")
        if free_head != 0 and not 2 <= free_head < self.pager.num_pages:
            raise CorruptTreeError(f"free-list head {free_head} out of range")
        self.root = root
        self.free_head = free_head

    # ---------- page allocation (with reuse of freed pages) ----------

    def _alloc(self):
        if self.free_head:
            page_no = self.free_head
            kind, _, nxt = NODE_HEADER.unpack_from(self.pager.read_page(page_no), 0)
            if kind != FREE:
                raise CorruptTreeError(f"page {page_no} is on the free list but is not a free page")
            self.free_head = nxt
            self._meta_dirty = True
            return page_no
        return self.pager.allocate()

    def _free(self, page_no):
        buf = bytearray(PAGE_SIZE)
        NODE_HEADER.pack_into(buf, 0, FREE, 0, self.free_head)
        self.pager.write_page(page_no, bytes(buf))
        self.free_head = page_no
        self._meta_dirty = True

    # ---------- node <-> page ----------

    def _read_node(self, page_no):
        self.page_reads += 1
        data = self.pager.read_page(page_no)
        kind, n, nxt = NODE_HEADER.unpack_from(data, 0)
        if kind == LEAF:
            node = Node(LEAF)
            node.next_leaf = nxt
            off = HEADER_SIZE
            for _ in range(n):
                if off + CELL_OVERHEAD > PAGE_SIZE:
                    raise CorruptTreeError(f"page {page_no}: leaf cell runs past the end of the page")
                key, vlen = LEAF_CELL.unpack_from(data, off)
                off += CELL_OVERHEAD
                if off + vlen > PAGE_SIZE:
                    raise CorruptTreeError(f"page {page_no}: leaf value runs past the end of the page")
                node.keys.append(key)
                node.values.append(data[off:off + vlen])
                off += vlen
            return node
        if kind == INTERNAL:
            if HEADER_SIZE + CHILD.size + n * INTERNAL_ENTRY.size > PAGE_SIZE:
                raise CorruptTreeError(f"page {page_no}: internal node runs past the end of the page")
            node = Node(INTERNAL)
            off = HEADER_SIZE
            node.children.append(CHILD.unpack_from(data, off)[0])
            off += CHILD.size
            for _ in range(n):
                key, child = INTERNAL_ENTRY.unpack_from(data, off)
                off += INTERNAL_ENTRY.size
                node.keys.append(key)
                node.children.append(child)
            return node
        raise CorruptTreeError(f"page {page_no}: unknown node type {kind}")

    def _write_node(self, page_no, node):
        buf = bytearray(PAGE_SIZE)
        if node.kind == LEAF:
            NODE_HEADER.pack_into(buf, 0, LEAF, len(node.keys), node.next_leaf)
            off = HEADER_SIZE
            for key, value in zip(node.keys, node.values):
                if off + CELL_OVERHEAD + len(value) > PAGE_SIZE:
                    raise BTreeError("leaf node does not fit in a page")
                LEAF_CELL.pack_into(buf, off, key, len(value))
                off += CELL_OVERHEAD
                buf[off:off + len(value)] = value
                off += len(value)
        else:
            if HEADER_SIZE + CHILD.size + len(node.keys) * INTERNAL_ENTRY.size > PAGE_SIZE:
                raise BTreeError("internal node does not fit in a page")
            NODE_HEADER.pack_into(buf, 0, INTERNAL, len(node.keys), 0)
            off = HEADER_SIZE
            CHILD.pack_into(buf, off, node.children[0])
            off += CHILD.size
            for key, child in zip(node.keys, node.children[1:]):
                INTERNAL_ENTRY.pack_into(buf, off, key, child)
                off += INTERNAL_ENTRY.size
        self.pager.write_page(page_no, bytes(buf))

    @staticmethod
    def _leaf_size(node):
        return HEADER_SIZE + sum(CELL_OVERHEAD + len(v) for v in node.values)

    def _underfull(self, node):
        if node.kind == LEAF:
            return self._leaf_size(node) < self.min_leaf_bytes
        return len(node.keys) < self.min_internal_keys

    @staticmethod
    def _cut_by_bytes(values):
        """Index that splits the cells into two halves of roughly equal byte size."""
        sizes = [CELL_OVERHEAD + len(v) for v in values]
        total = sum(sizes)
        acc = 0
        for cut, size in enumerate(sizes):
            acc += size
            if acc * 2 >= total:
                break
        cut += 1
        if not 1 <= cut < len(values):
            raise BTreeError("cannot split leaf")
        return cut

    # ---------- argument checks ----------

    @staticmethod
    def _check_key(key):
        if not isinstance(key, int) or isinstance(key, bool):
            raise TypeError("key must be an int")
        if not KEY_MIN <= key <= KEY_MAX:
            raise ValueError("key out of 64-bit range")

    def _check_value(self, value):
        if not isinstance(value, (bytes, bytearray)):
            raise TypeError("value must be bytes")
        if len(value) > self.max_value:
            raise ValueError(f"value of {len(value)} bytes exceeds the maximum of {self.max_value}")
        return bytes(value)

    # ---------- insert ----------

    def insert(self, key, value):
        """Add a new key. Raises DuplicateKeyError if it already exists."""
        self._check_key(key)
        value = self._check_value(value)
        with self.transaction():
            self._insert_checked(key, value)

    def put(self, key, value):
        """Add the key, or replace its value if it already exists."""
        self._check_key(key)
        value = self._check_value(value)          # validate first: never delete and then fail
        with self.transaction():                  # delete + insert commit together or not at all
            self.delete(key)                      # replacing = delete + insert keeps balance rules simple
            self._insert_checked(key, value)

    def _insert_checked(self, key, value):
        result = self._insert(self.root, key, value)
        if result is not None:                      # the root itself split: grow the tree
            split_key, right_page = result
            node = Node(INTERNAL)
            node.keys = [split_key]
            node.children = [self.root, right_page]
            new_root = self._alloc()
            self._write_node(new_root, node)
            self.root = new_root
            self._meta_dirty = True
        self._save_meta_if_dirty()

    def _insert(self, page_no, key, value):
        """Insert below page_no. Returns (separator_key, new_right_page) if this node split."""
        node = self._read_node(page_no)

        if node.kind == LEAF:
            i = bisect.bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                raise DuplicateKeyError(key)
            node.keys.insert(i, key)
            node.values.insert(i, value)
            if self._leaf_size(node) <= self.leaf_capacity:
                self._write_node(page_no, node)
                return None
            return self._split_leaf(page_no, node)

        i = bisect.bisect_right(node.keys, key)
        result = self._insert(node.children[i], key, value)
        if result is None:
            return None
        split_key, new_page = result
        node.keys.insert(i, split_key)
        node.children.insert(i + 1, new_page)
        if len(node.keys) <= self.max_internal_keys:
            self._write_node(page_no, node)
            return None
        return self._split_internal(page_no, node)

    def _split_leaf(self, page_no, node):
        cut = self._cut_by_bytes(node.values)       # split by bytes, so both halves surely fit
        right = Node(LEAF)
        right.keys, right.values = node.keys[cut:], node.values[cut:]
        right.next_leaf = node.next_leaf
        node.keys, node.values = node.keys[:cut], node.values[:cut]
        right_page = self._alloc()
        node.next_leaf = right_page
        self._write_node(right_page, right)
        self._write_node(page_no, node)
        return right.keys[0], right_page            # separator is copied up

    def _split_internal(self, page_no, node):
        mid = len(node.keys) // 2
        promote = node.keys[mid]
        right = Node(INTERNAL)
        right.keys = node.keys[mid + 1:]
        right.children = node.children[mid + 1:]
        node.keys = node.keys[:mid]
        node.children = node.children[:mid + 1]
        right_page = self._alloc()
        self._write_node(right_page, right)
        self._write_node(page_no, node)
        return promote, right_page                  # separator is moved up

    # ---------- delete ----------

    def delete(self, key):
        """Remove a key. Returns True if it existed, False if not."""
        self._check_key(key)
        with self.transaction():
            return self._delete_key(key)

    def _delete_key(self, key):
        found = self._delete(self.root, key)
        if found:
            # The root may have been left with a single child: that child becomes the root.
            while True:
                root = self._read_node(self.root)
                if root.kind == INTERNAL and not root.keys:
                    old_root = self.root
                    self.root = root.children[0]
                    self._free(old_root)
                    self._meta_dirty = True
                else:
                    break
        self._save_meta_if_dirty()
        return found

    def _delete(self, page_no, key):
        node = self._read_node(page_no)
        if node.kind == LEAF:
            i = bisect.bisect_left(node.keys, key)
            if i >= len(node.keys) or node.keys[i] != key:
                return False
            del node.keys[i]
            del node.values[i]
            self._write_node(page_no, node)
            return True

        i = bisect.bisect_right(node.keys, key)
        if not self._delete(node.children[i], key):
            return False
        if self._underfull(self._read_node(node.children[i])):
            self._rebalance(page_no, node, i)
        return True

    def _rebalance(self, parent_page, parent, i):
        """Child i of parent is underfull: merge it with a sibling, or redistribute between them."""
        left_idx = i - 1 if i > 0 else i            # always work on the pair (left_idx, left_idx + 1)
        left_page, right_page = parent.children[left_idx], parent.children[left_idx + 1]
        left, right = self._read_node(left_page), self._read_node(right_page)

        if left.kind == LEAF:
            keys, values = left.keys + right.keys, left.values + right.values
            pooled = HEADER_SIZE + sum(CELL_OVERHEAD + len(v) for v in values)
            if pooled <= self.leaf_capacity:        # everything fits in one leaf: merge
                left.keys, left.values = keys, values
                left.next_leaf = right.next_leaf
                self._write_node(left_page, left)
                self._free(right_page)
                del parent.keys[left_idx]
                del parent.children[left_idx + 1]
            else:                                   # too big to merge: split the pool evenly again
                cut = self._cut_by_bytes(values)
                left.keys, left.values = keys[:cut], values[:cut]
                right.keys, right.values = keys[cut:], values[cut:]
                self._write_node(left_page, left)
                self._write_node(right_page, right)
                parent.keys[left_idx] = right.keys[0]
        else:
            sep = parent.keys[left_idx]
            keys = left.keys + [sep] + right.keys   # the separator comes down from the parent
            children = left.children + right.children
            if len(keys) <= self.max_internal_keys:  # merge
                left.keys, left.children = keys, children
                self._write_node(left_page, left)
                self._free(right_page)
                del parent.keys[left_idx]
                del parent.children[left_idx + 1]
            else:                                   # redistribute: a new middle key goes up
                mid = len(keys) // 2
                left.keys, left.children = keys[:mid], children[:mid + 1]
                right.keys, right.children = keys[mid + 1:], children[mid + 1:]
                self._write_node(left_page, left)
                self._write_node(right_page, right)
                parent.keys[left_idx] = keys[mid]
        self._write_node(parent_page, parent)

    # ---------- lookup and scans ----------

    def get(self, key):
        """Return the value for key, or None."""
        self._check_key(key)
        page_no = self.root
        while True:
            node = self._read_node(page_no)
            if node.kind == LEAF:
                i = bisect.bisect_left(node.keys, key)
                if i < len(node.keys) and node.keys[i] == key:
                    return node.values[i]
                return None
            page_no = node.children[bisect.bisect_right(node.keys, key)]

    def __contains__(self, key):
        return self.get(key) is not None

    def path_to(self, key):
        """Pages visited when looking up key, root first (shows the cost of a lookup)."""
        self._check_key(key)
        path, page_no = [], self.root
        while True:
            path.append(page_no)
            node = self._read_node(page_no)
            if node.kind == LEAF:
                return path
            page_no = node.children[bisect.bisect_right(node.keys, key)]

    def scan(self, lo=None, hi=None):
        """Yield (key, value) in key order for lo <= key <= hi. None means unbounded.

        Descends once to the leaf that holds lo, then follows the leaf chain.
        Do not insert or delete while iterating.
        """
        if lo is not None:
            self._check_key(lo)
        if hi is not None:
            self._check_key(hi)
        if lo is not None and hi is not None and lo > hi:
            return
        node = self._read_node(self.root)
        while node.kind == INTERNAL:
            child = node.children[0] if lo is None else node.children[bisect.bisect_right(node.keys, lo)]
            node = self._read_node(child)
        i = 0 if lo is None else bisect.bisect_left(node.keys, lo)
        while True:
            for j in range(i, len(node.keys)):
                if hi is not None and node.keys[j] > hi:
                    return
                yield node.keys[j], node.values[j]
            if not node.next_leaf:
                return
            node = self._read_node(node.next_leaf)
            i = 0

    def items(self):
        """Yield every (key, value) in key order."""
        return self.scan()

    def count(self):
        return sum(1 for _ in self.items())

    def height(self):
        h, node = 1, self._read_node(self.root)
        while node.kind == INTERNAL:
            node = self._read_node(node.children[0])
            h += 1
        return h

    # ---------- statistics and self-check ----------

    def _free_pages(self):
        """Walk the free list. Returns the list of page numbers."""
        pages, seen, page_no = [], set(), self.free_head
        while page_no:
            if page_no in seen:
                raise CorruptTreeError(f"free list has a cycle at page {page_no}")
            if not 2 <= page_no < self.pager.num_pages:
                raise CorruptTreeError(f"free list points to invalid page {page_no}")
            seen.add(page_no)
            kind, _, nxt = NODE_HEADER.unpack_from(self.pager.read_page(page_no), 0)
            if kind != FREE:
                raise CorruptTreeError(f"page {page_no} is on the free list but is not a free page")
            pages.append(page_no)
            page_no = nxt
        return pages

    def stats(self):
        """Counts of keys, pages in the tree, pages on the free list and in the file."""
        tree_pages, keys = 0, 0
        stack = [self.root]
        while stack:
            node = self._read_node(stack.pop())
            tree_pages += 1
            if node.kind == LEAF:
                keys += len(node.keys)
            else:
                stack.extend(node.children)
        return {
            "keys": keys,
            "height": self.height(),
            "tree_pages": tree_pages,
            "free_pages": len(self._free_pages()),
            "file_pages": self.pager.num_pages,
        }

    def validate(self):
        """Check every structural rule of the tree. Returns the number of keys."""
        leaves, depths, tree_pages = [], set(), []

        def walk(page_no, lo, hi, depth):
            if page_no in tree_pages:
                raise CorruptTreeError(f"page {page_no} is reachable twice")
            tree_pages.append(page_no)
            node = self._read_node(page_no)
            keys = node.keys
            for a, b in zip(keys, keys[1:]):
                if a >= b:
                    raise CorruptTreeError(f"page {page_no}: keys are not strictly increasing")
            for k in keys:
                if (lo is not None and k < lo) or (hi is not None and k >= hi):
                    raise CorruptTreeError(f"page {page_no}: key {k} outside its allowed range")
            is_root = page_no == self.root
            if node.kind == LEAF:
                size = self._leaf_size(node)
                if size > self.leaf_capacity:
                    raise CorruptTreeError(f"page {page_no}: leaf is over capacity")
                if not is_root and size < self.min_leaf_bytes:
                    raise CorruptTreeError(f"page {page_no}: leaf is underfull ({size} bytes)")
                leaves.append((page_no, node))
                depths.add(depth)
                return
            if not keys:
                raise CorruptTreeError(f"page {page_no}: internal node without keys")
            if len(keys) > self.max_internal_keys:
                raise CorruptTreeError(f"page {page_no}: internal node is over capacity")
            if not is_root and len(keys) < self.min_internal_keys:
                raise CorruptTreeError(f"page {page_no}: internal node is underfull ({len(keys)} keys)")
            for i, child in enumerate(node.children):
                child_lo = lo if i == 0 else keys[i - 1]
                child_hi = hi if i == len(keys) else keys[i]
                walk(child, child_lo, child_hi, depth + 1)

        walk(self.root, None, None, 0)
        if len(depths) != 1:
            raise CorruptTreeError("leaves are not all at the same depth")
        expected_next = [page for page, _ in leaves[1:]] + [0]
        for (page_no, node), nxt in zip(leaves, expected_next):
            if node.next_leaf != nxt:
                raise CorruptTreeError(f"page {page_no}: broken leaf chain")

        free = self._free_pages()
        if set(free) & set(tree_pages):
            raise CorruptTreeError("a page is both in the tree and on the free list")
        accounted = len(tree_pages) + len(free) + 2          # + page 0 (file header) + page 1 (meta)
        if accounted != self.pager.num_pages:
            raise CorruptTreeError(
                f"{self.pager.num_pages - accounted} page(s) leaked: "
                "neither in the tree nor on the free list"
            )
        return sum(len(node.keys) for _, node in leaves)

    # ---------- lifecycle ----------

    def flush(self):
        self._save_meta_if_dirty()
        self.pager.flush()

    def close(self):
        self._save_meta_if_dirty()
        self.pager.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
