# ndttt.py
# Dimension-agnostic n^d tic-tac-toe engine with w-in-a-row wins.
# Bitboards + precomputed line masks + Zobrist hashing + make/unmake.

from itertools import product
from random import Random


# used for translating between 1d flat indices and nd vectors
def strides(n, d):
    """Row-major strides so idx = sum(p[i] * s[i])."""
    s = [1]*d
    for i in range(d-2, -1, -1):
        s[i] = s[i+1] * n
    return s

def index_of(coord, s):
    return sum(ci*si for ci, si in zip(coord, s))

def coords_of(idx, n, d, s=None):
    if s is None: s = strides(n, d)
    out = [0]*d
    for i in range(d):
        out[i], idx = divmod(idx, s[i])
    return out

# our winning directions
def gen_canonical_dirs(d):
    """All {0,1}^d \\ {0} (axes + all diagonals; only forward steps)."""
    V = []
    for mask in range(1, 1 << d):
        v = [(mask >> i) & 1 for i in range(d)]
        V.append(v)
    return V  # len = 2^d - 1

def gen_lines(n, d, w):
    """
    Return:
      line_masks: list[int]  (each has w bits set)
      cell_to_lines: list[list[int]]  (line ids touching that cell)
      s: strides
    """
    s = strides(n, d)
    V = gen_canonical_dirs(d)

    N = n**d
    cell_to_lines = [[] for _ in range(N)]
    line_indices = []  # temp: list[list[cell_idx]]

    for v in V:
        # valid start ranges per dimension
        start_ranges = []
        for i, vi in enumerate(v):
            if vi == 1:
                start_ranges.append(range(0, n - w + 1))
            else:
                start_ranges.append(range(0, n))

        for start in product(*start_ranges):
            cells = []
            p = list(start)
            for _ in range(w):
                cells.append(index_of(p, s))
                for i, vi in enumerate(v):
                    p[i] += vi
            lid = len(line_indices)
            line_indices.append(cells)
            for idx in cells:
                cell_to_lines[idx].append(lid)

    # convert to bitmasks
    line_masks = []
    for cells in line_indices:
        m = 0
        for idx in cells:
            m |= (1 << idx)
        line_masks.append(m)

    return line_masks, cell_to_lines, s

class Zobrist:
    """64-bit Zobrist hashing for (board, side-to-move)."""
    def __init__(self, cells, seed=0xC0FFEE):
        rng = Random(seed)
        # z[player][cell]
        self.z = [[rng.getrandbits(64) for _ in range(cells)] for _ in range(2)]
        self.z_stm = rng.getrandbits(64)

    def piece(self, player, cell):
        return self.z[player][cell]

    @property
    def stm(self):
        return self.z_stm

class NDTTT:
    """
    General n^d Tic-Tac-Toe with w-in-a-row.
    Bitboards: P[0], P[1]; stm in {0,1}.
    """
    def __init__(self, n=4, d=4, w=4, seed=0xBADC0DE):
        assert 1 <= w <= n
        self.n, self.d, self.w = n, d, w
        self.N = n**d
        self.full_mask = (1 << self.N) - 1

        self.line_masks, self.cell_to_lines, self.strides = gen_lines(n, d, w)

        # state
        self.P = [0, 0]       # bitboards
        self.stm = 0          # side to move
        self.last_idx = None
        self.empty_count = self.N

        # history for undo
        self.hist = []

        # zobrist hash
        self.zob = Zobrist(self.N, seed)
        self.hash = 0

    # -------- queries --------
    def legal_mask(self):
        return (~(self.P[0] | self.P[1])) & self.full_mask

    def legal_moves(self):
        """Yield legal move indices in ascending order."""
        m = self.legal_mask()
        while m:
            lsb = m & -m
            idx = (lsb.bit_length() - 1)
            yield idx
            m &= m - 1

    def occupied(self, idx):
        bit = 1 << idx
        return (self.P[0] & bit) or (self.P[1] & bit)

    def is_win_after(self, idx, player=None):
        """Check only lines through idx. O(#lines through cell)."""
        if player is None:
            player = self.stm
        bb = self.P[player] | (1 << idx)
        for lid in self.cell_to_lines[idx]:
            lm = self.line_masks[lid]
            if (bb & lm) == lm:
                return True
        return False

    def result(self):
        """Return +1/-1/0 if terminal from the POV of player to move just before terminal step,
           else None. Use after a move if you need final outcome."""
        if self.last_idx is None:
            return None
        # The winner is the side who just moved (opposite of current stm)
        last_player = self.stm ^ 1
        for lid in self.cell_to_lines[self.last_idx]:
            lm = self.line_masks[lid]
            if (self.P[last_player] & lm) == lm:
                return 1 if last_player == 0 else -1
        if self.empty_count == 0:
            return 0
        return None

    # -------- make / unmake --------
    def push(self, idx):
        """Place stone for current player at idx. Returns (win, draw)."""
        assert not self.occupied(idx), "Illegal move"
        player = self.stm
        bit = 1 << idx

        # history: (idx, prev_last_idx, prev_stm, prev_hash)
        self.hist.append((idx, self.last_idx, self.stm, self.hash))

        # apply
        self.P[player] |= bit
        self.empty_count -= 1
        self.last_idx = idx

        # hash update
        self.hash ^= self.zob.piece(player, idx)
        self.hash ^= self.zob.stm  # toggle side

        # win/draw?
        win = any((self.P[player] & lm) == lm for lm in (self.line_masks[l] for l in self.cell_to_lines[idx]))
        draw = (not win) and (self.empty_count == 0)

        # switch side
        self.stm ^= 1
        return win, draw

    def pop(self):
        """Undo last move."""
        assert self.hist, "Nothing to undo"
        idx, prev_last_idx, prev_stm, prev_hash = self.hist.pop()

        # revert side first (push toggled it)
        self.stm ^= 1
        player = self.stm
        bit = 1 << idx

        # remove stone
        self.P[player] &= ~bit
        self.empty_count += 1
        self.last_idx = prev_last_idx
        self.hash = prev_hash

    # -------- helpers --------
    def idx(self, coord):
        return index_of(coord, self.strides)

    def coord(self, idx):
        return coords_of(idx, self.n, self.d, self.strides)

    def pretty(self, dims_per_line=2):
        """
        Debug print: flattens higher dims into blocks.
        dims_per_line = how many axes to show as a 2D grid per block.
        """
        assert 1 <= dims_per_line <= self.d
        import math
        # map each cell to char
        chars = ['.'] * self.N
        b0, b1 = self.P
        m = b0 | b1
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            chars[i] = 'X' if (b0 >> i) & 1 else 'O'
            m &= m - 1

        # recursively print in grouped slices
        def rec_print(prefix, dim):
            if dim == self.d - dims_per_line:
                # print a block (2D or more if dims_per_line>2)
                sizes = [self.n]*dims_per_line
                # iterate nested rows
                def idx_from(offsets):
                    coord = prefix + offsets
                    return index_of(coord, self.strides)
                # generate all rows
                grids = []
                def build_rows(cur, k):
                    if k == dims_per_line - 1:
                        row = []
                        for a in range(self.n):
                            idx = idx_from(cur + [a])
                            row.append(chars[idx])
                        grids.append(' '.join(row))
                    else:
                        for a in range(self.n):
                            build_rows(cur + [a], k+1)
                build_rows([], 0)
                print('\n'.join(grids))
                print()
            else:
                for a in range(self.n):
                    rec_print(prefix + [a], dim+1)

        rec_print([], 0)

# -------- small self-test --------
if __name__ == "__main__":
    # 4D 4x4x4x4, need 4-in-a-row
    g = NDTTT(n=4, d=4, w=4)

    # Play a main-diagonal (0,0,0,0) -> (1,1,1,1) -> (2,2,2,2) -> (3,3,3,3)
    # Indices:
    path = [g.idx([i,i,i,i]) for i in range(4)]
    win = False
    for t, idx in enumerate(path):
        w_, d_ = g.push(idx)   # X moves
        if w_: 
            win = True
            break
        # O plays some filler
        for m in g.legal_moves():
            g.push(m); break
    assert win, "Expected a diagonal win for X"
    print("Zobrist hash:", hex(g.hash))
    print("All good.")
