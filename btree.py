"""A disk-based B+ tree on top of the Pager.

File layout:   page 1 = meta (root page number), page 2.. = tree nodes.
Leaf node:     header | (key, value_len, value)...      next-leaf pointer in header
Internal node: header | child0 | (key, child)...        keys[i] = smallest key in children[i+1]
Keys are signed 64-bit integers; values are bytes.
"""
import bisect
import struct

from pager import PAGE_SIZE, Pager

LEAF, INTERNAL = 1, 2
NODE_HEADER = struct.Struct("<BHI")      # node type, number of keys, next-leaf page (0 = none)
HEADER_SIZE = NODE_HEADER.size           # 7 bytes
LEAF_CELL = struct.Struct("<qH")         # key, value length
CELL_OVERHEAD = LEAF_CELL.size           # 10 bytes
INTERNAL_ENTRY = struct.Struct("<qI")    # key, right child page
CHILD = struct.Struct("<I")
MAX_INTERNAL_KEYS = (PAGE_SIZE - HEADER_SIZE - CHILD.size) // INTERNAL_ENTRY.size   # 340

META_PAGE = 1
META = struct.Struct("<8sI")             # magic, root page number
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
    def __init__(self, path, leaf_capacity=PAGE_SIZE, max_internal_keys=MAX_INTERNAL_KEYS):
        # Capacities are not stored in the file: reopen a file with the values it was built with.
        if not 64 <= leaf_capacity <= PAGE_SIZE:
            raise ValueError(f"leaf_capacity must be between 64 and {PAGE_SIZE}")
        if not 3 <= max_internal_keys <= MAX_INTERNAL_KEYS:
            raise ValueError(f"max_internal_keys must be between 3 and {MAX_INTERNAL_KEYS}")
        self.leaf_capacity = leaf_capacity
        self.max_internal_keys = max_internal_keys
        # Largest value for which a leaf always holds >= 4 cells, so splits always work
        self.max_value = (leaf_capacity - HEADER_SIZE) // 4 - CELL_OVERHEAD
        self.pager = Pager(path)
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
        if self.pager.allocate() != META_PAGE:
            raise BTreeError("unexpected page layout in a new file")
        root = self.pager.allocate()
        self._write_node(root, Node(LEAF))
        self.root = root
        self._write_meta()
        self.pager.flush()

    def _write_meta(self):
        page = META.pack(META_MAGIC, self.root).ljust(PAGE_SIZE, b"\x00")
        self.pager.write_page(META_PAGE, page)

    def _load_meta(self):
        if self.pager.num_pages < 3:
            raise CorruptTreeError("file has no B+ tree (missing meta or root page)")
        magic, root = META.unpack_from(self.pager.read_page(META_PAGE), 0)
        if magic != META_MAGIC:
            raise CorruptTreeError("bad B+ tree meta page (wrong magic number)")
        if not 2 <= root < self.pager.num_pages:
            raise CorruptTreeError(f"root page {root} out of range")
        self.root = root

    # ---------- node <-> page ----------

    def _read_node(self, page_no):
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
        self._put(key, value, replace=False)

    def put(self, key, value):
        """Add the key, or replace its value if it already exists."""
        self._put(key, value, replace=True)

    def _put(self, key, value, replace):
        self._check_key(key)
        value = self._check_value(value)
        result = self._insert(self.root, key, value, replace)
        if result is not None:                      # the root itself split: grow the tree
            split_key, right_page = result
            node = Node(INTERNAL)
            node.keys = [split_key]
            node.children = [self.root, right_page]
            new_root = self.pager.allocate()
            self._write_node(new_root, node)
            self.root = new_root
            self._write_meta()

    def _insert(self, page_no, key, value, replace):
        """Insert below page_no. Returns (separator_key, new_right_page) if this node split."""
        node = self._read_node(page_no)

        if node.kind == LEAF:
            i = bisect.bisect_left(node.keys, key)
            if i < len(node.keys) and node.keys[i] == key:
                if not replace:
                    raise DuplicateKeyError(key)
                node.values[i] = value
            else:
                node.keys.insert(i, key)
                node.values.insert(i, value)
            if self._leaf_size(node) <= self.leaf_capacity:
                self._write_node(page_no, node)
                return None
            return self._split_leaf(page_no, node)

        i = bisect.bisect_right(node.keys, key)
        result = self._insert(node.children[i], key, value, replace)
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
        sizes = [CELL_OVERHEAD + len(v) for v in node.values]
        total = sum(sizes)
        acc = 0
        for cut, size in enumerate(sizes):          # split by bytes, so both halves surely fit
            acc += size
            if acc * 2 >= total:
                break
        cut += 1
        if not 1 <= cut < len(node.keys):
            raise BTreeError("cannot split leaf")
        right = Node(LEAF)
        right.keys, right.values = node.keys[cut:], node.values[cut:]
        right.next_leaf = node.next_leaf
        node.keys, node.values = node.keys[:cut], node.values[:cut]
        right_page = self.pager.allocate()
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
        right_page = self.pager.allocate()
        self._write_node(right_page, right)
        self._write_node(page_no, node)
        return promote, right_page                  # separator is moved up

    # ---------- lookup ----------

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

    def items(self):
        """Yield (key, value) in key order by following the leaf chain."""
        node = self._read_node(self.root)
        while node.kind == INTERNAL:
            node = self._read_node(node.children[0])
        while True:
            yield from zip(node.keys, node.values)
            if not node.next_leaf:
                return
            node = self._read_node(node.next_leaf)

    def count(self):
        return sum(1 for _ in self.items())

    def height(self):
        h, node = 1, self._read_node(self.root)
        while node.kind == INTERNAL:
            node = self._read_node(node.children[0])
            h += 1
        return h

    # ---------- self-check ----------

    def validate(self):
        """Check every structural rule of the tree. Returns the number of keys."""
        leaves, depths = [], set()

        def walk(page_no, lo, hi, depth):
            node = self._read_node(page_no)
            keys = node.keys
            for a, b in zip(keys, keys[1:]):
                if a >= b:
                    raise CorruptTreeError(f"page {page_no}: keys are not strictly increasing")
            for k in keys:
                if (lo is not None and k < lo) or (hi is not None and k >= hi):
                    raise CorruptTreeError(f"page {page_no}: key {k} outside its allowed range")
            if node.kind == LEAF:
                if self._leaf_size(node) > self.leaf_capacity:
                    raise CorruptTreeError(f"page {page_no}: leaf is over capacity")
                leaves.append((page_no, node))
                depths.add(depth)
                return
            if not keys:
                raise CorruptTreeError(f"page {page_no}: internal node without keys")
            if len(keys) > self.max_internal_keys:
                raise CorruptTreeError(f"page {page_no}: internal node is over capacity")
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
        return sum(len(node.keys) for _, node in leaves)

    # ---------- lifecycle ----------

    def flush(self):
        self.pager.flush()

    def close(self):
        self.pager.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
