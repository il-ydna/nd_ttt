# ndttt_pygame.py
import pygame
from itertools import product
from random import Random

# ---------------------- Engine core (dimension-agnostic) ----------------------

def strides(n, d):
    s = [1]*d
    for i in range(d-2, -1, -1):
        s[i] = s[i+1]*n
    return s

def index_of(coord, s):
    return sum(ci*si for ci, si in zip(coord, s))

def coords_of(idx, n, d, s=None):
    if s is None: s = strides(n, d)
    out = [0]*d
    for i in range(d):
        out[i], idx = divmod(idx, s[i])
    return out

def gen_canonical_dirs(d):
    V = []
    for mask in range(1, 1 << d):
        v = [(mask >> i) & 1 for i in range(d)]
        V.append(v)
    return V  # len = 2^d - 1

def gen_lines(n, d, w):
    s = strides(n, d)
    V = gen_canonical_dirs(d)
    N = n**d
    cell_to_lines = [[] for _ in range(N)]
    line_indices = []

    for v in V:
        ranges = []
        for i, vi in enumerate(v):
            if vi == 1:
                ranges.append(range(0, n - w + 1))
            else:
                ranges.append(range(0, n))
        for start in product(*ranges):
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

    line_masks = []
    for cells in line_indices:
        m = 0
        for idx in cells:
            m |= (1 << idx)
        line_masks.append(m)

    return line_masks, cell_to_lines, s

class Zobrist:
    def __init__(self, cells, players=2, seed=0xC0FFEE):
        rng = Random(seed)
        self.z_piece = [[rng.getrandbits(64) for _ in range(cells)]
                        for _ in range(players)]
        self.z_stm = rng.getrandbits(64)

class NDTTT:
    def __init__(self, n=4, d=4, w=4, players=2, seed=0xBADC0DE):
        assert 1 <= w <= n
        self.n, self.d, self.w = n, d, w
        self.k = players
        self.N = n**d
        self.full_mask = (1 << self.N) - 1

        self.line_masks, self.cell_to_lines, self.strides = gen_lines(n, d, w)

        self.P = [0]*self.k
        self.stm = 0
        self.last_idx = None
        self.empty_count = self.N

        self.hist = []
        self.zob = Zobrist(self.N, players=self.k, seed=seed)
        self.hash = 0

    def legal_mask(self):
        occ = 0
        for bb in self.P: occ |= bb
        return (~occ) & self.full_mask

    def legal_moves(self):
        m = self.legal_mask()
        while m:
            lsb = m & -m
            i = lsb.bit_length() - 1
            yield i
            m &= m - 1

    def occupied(self, idx):
        b = 1 << idx
        for bb in self.P:
            if bb & b: return True
        return False

    def push(self, idx):
        assert not self.occupied(idx), "Illegal move"
        player = self.stm
        bit = 1 << idx
        self.hist.append((idx, self.last_idx, self.stm, self.hash))
        self.P[player] |= bit
        self.empty_count -= 1
        self.last_idx = idx

        self.hash ^= self.zob.z_piece[player][idx]
        self.hash ^= self.zob.z_stm

        # win/draw check only through lines that include idx
        win = any((self.P[player] & self.line_masks[L]) == self.line_masks[L]
                  for L in self.cell_to_lines[idx])
        is_draw = (not win) and (self.empty_count == 0)

        self.stm = (self.stm + 1) % self.k
        return win, is_draw

    def pop(self):
        assert self.hist, "Nothing to undo"
        idx, prev_last_idx, prev_stm, prev_hash = self.hist.pop()
        self.stm = prev_stm
        self.P[self.stm] &= ~(1 << idx)
        self.empty_count += 1
        self.last_idx = prev_last_idx
        self.hash = prev_hash

    def idx(self, coord):
        return index_of(coord, self.strides)

    def coord(self, idx):
        return coords_of(idx, self.n, self.d, self.strides)

# -------- helper to find the exact winning mask after a move ----------
def winning_mask_for_last_move(game: NDTTT):
    """Return (mask, winner_player_index) if someone just won; else (0, None)."""
    if game.last_idx is None: return 0, None
    last_player = (game.stm - 1) % game.k  # side who just moved
    bb = game.P[last_player]
    for L in game.cell_to_lines[game.last_idx]:
        lm = game.line_masks[L]
        if (bb & lm) == lm:
            return lm, last_player
    return 0, None

# ----------------------------- Pygame UI --------------------------------------

# Colors
BG = (18, 18, 20)
GRID = (120, 120, 130)
TEXT = (210, 210, 220)
HL_CROSS = (90, 160, 250)
P0_CLR = (40, 150, 255)
P1_CLR = (240, 80, 80)
WIN_GLOW = (250, 220, 70)

def main():
    n, d, w = 4, 4, 4
    cell = 42          # pixel size of each cell
    pad_board = 16     # space between mini-boards
    margin = 24        # outer margin
    stroke = 2

    pygame.init()
    pygame.display.set_caption("4D Tic-Tac-Toe (small multiples)")
    font = pygame.font.SysFont("consolas,menlo,monospace", 18)

    # Window size: display n×n boards, each board is (n*cell) square
    board_px = n * cell
    win_w = margin*2 + n * board_px + (n-1) * pad_board
    win_h = margin*2 + n * board_px + (n-1) * pad_board + 60  # extra for status
    screen = pygame.display.set_mode((win_w, win_h))

    game = NDTTT(n=n, d=d, w=w)
    rng = Random(1337)

    running = True
    winner = None
    win_mask = 0
    hover = None  # (x,y,z,w) under mouse or None

    def board_origin(zz, ww):
        ox = margin + ww * (board_px + pad_board)
        oy = margin + zz * (board_px + pad_board)
        return ox, oy

    def locate_mouse(mx, my):
        # find which mini-board (z, w) and which cell (x, y) the mouse is over
        for z in range(n):
            for wv in range(n):
                ox, oy = board_origin(z, wv)
                if ox <= mx < ox + board_px and oy <= my < oy + board_px:
                    lx, ly = mx - ox, my - oy
                    x = int(lx // cell)
                    y = int(ly // cell)
                    return (x, y, z, wv)
        return None

    def draw():
        screen.fill(BG)

        # crosshair highlight of same (x,y) across all slices if hovering
        if hover is not None and winner is None:
            xh, yh, _, _ = hover
            for z in range(n):
                for wv in range(n):
                    ox, oy = board_origin(z, wv)
                    pygame.draw.rect(screen, HL_CROSS, (ox + xh*cell, oy, stroke, board_px))
                    pygame.draw.rect(screen, HL_CROSS, (ox, oy + yh*cell, board_px, stroke))

        # draw all mini-boards
        for z in range(n):
            for wv in range(n):
                ox, oy = board_origin(z, wv)

                # grid
                for gy in range(n+1):
                    y0 = oy + gy*cell
                    pygame.draw.line(screen, GRID, (ox, y0), (ox + board_px, y0), stroke)
                for gx in range(n+1):
                    x0 = ox + gx*cell
                    pygame.draw.line(screen, GRID, (x0, oy), (x0, oy + board_px), stroke)

                # label
                lbl = font.render(f"z={z} w={wv}", True, TEXT)
                screen.blit(lbl, (ox, oy - 20))

                # stones & win glow
                # We draw a soft highlight for any cell that is in the winning mask
                # and then draw the stones on top.
                for y in range(n):
                    for x in range(n):
                        idx = game.idx([x, y, z, wv])
                        rect = pygame.Rect(ox + x*cell, oy + y*cell, cell, cell)

                        if win_mask and ((win_mask >> idx) & 1):
                            pygame.draw.rect(screen, WIN_GLOW, rect.inflate(-6, -6), border_radius=8)

                        # P0 circle, P1 square
                        if (game.P[0] >> idx) & 1:
                            pygame.draw.circle(screen, P0_CLR, rect.center, int(cell*0.35))
                        elif (game.P[1] >> idx) & 1:
                            inset = int(cell*0.22)
                            pygame.draw.rect(screen, P1_CLR, rect.inflate(-inset*2, -inset*2), border_radius=6)

        # status bar
        status_y = win_h - 50
        if winner is None:
            turn = "X" if game.stm == 0 else "O"
            msg = f"Turn: {turn} | Keys: U=Undo  R=Reset  SPACE=Random move  ESC=Quit"
        else:
            msg = f"Winner: {'X' if winner==0 else 'O'}  |  R=Reset  U=Undo"
        surf = font.render(msg, True, TEXT)
        screen.blit(surf, (margin, status_y))

        pygame.display.flip()

    clock = pygame.time.Clock()

    while running:
        clock.tick(60)
        mx, my = pygame.mouse.get_pos()
        hover = locate_mouse(mx, my)

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    running = False

                elif event.key == pygame.K_r:
                    # reset
                    game = NDTTT(n=n, d=d, w=w)
                    winner = None
                    win_mask = 0

                elif event.key == pygame.K_u:
                    # undo (pop once; if in a finished state, pop twice to fully revert last result)
                    if game.hist:
                        game.pop()
                        winner = None
                        win_mask = 0

                elif event.key == pygame.K_SPACE and winner is None:
                    # random move for current side (quick sim)
                    moves = list(game.legal_moves())
                    if moves:
                        mv = rng.choice(moves)
                        win, draw = game.push(mv)
                        if win:
                            win_mask, winner = winning_mask_for_last_move(game)
                        elif draw:
                            win_mask, winner = 0, None  # draw: no winner, no mask

            elif event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
                if winner is not None:
                    # ignore clicks when game over
                    pass
                else:
                    target = locate_mouse(mx, my)
                    if target is not None:
                        x, y, z, wv = target
                        idx = game.idx([x, y, z, wv])
                        if not game.occupied(idx):
                            win, is_draw = game.push(idx)
                            if win:
                                win_mask, winner = winning_mask_for_last_move(game)
                            elif is_draw:
                                win_mask, winner = 0, None  # draw

        draw()

    pygame.quit()

if __name__ == "__main__":
    main()
