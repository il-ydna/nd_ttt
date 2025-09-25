# ndttt_pygame_gl.py
import math
from itertools import product
from random import Random

import pygame
from pygame.locals import DOUBLEBUF, OPENGL

from OpenGL.GL import *
from OpenGL.GLU import *

# ---------------------- Engine (same core as before) ----------------------
def strides(n, d):
    s = [1]*d
    for i in range(d-2, -1, -1):
        s[i] = s[i+1]*n
    return s

def index_of(coord, s):
    return sum(ci*si for ci, si in zip(coord, s))

def gen_canonical_dirs(d):
    V = []
    for mask in range(1, 1 << d):
        v = [(mask >> i) & 1 for i in range(d)]
        V.append(v)
    return V

def gen_lines(n, d, w):
    s = strides(n, d)
    V = gen_canonical_dirs(d)
    N = n**d
    cell_to_lines = [[] for _ in range(N)]
    line_indices = []
    from itertools import product
    for v in V:
        ranges = []
        for i, vi in enumerate(v):
            ranges.append(range(0, n - w + 1) if vi == 1 else range(0, n))
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

# Win-mask for highlighting
def winning_mask_for_last_move(game: NDTTT):
    if game.last_idx is None: return 0, None
    last_player = (game.stm - 1) % game.k
    bb = game.P[last_player]
    for L in game.cell_to_lines[game.last_idx]:
        lm = game.line_masks[L]
        if (bb & lm) == lm:
            return lm, last_player
    return 0, None

def gl_init(width, height, fov=50.0):
    glViewport(0, 0, width, height)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glClearColor(0.08, 0.08, 0.09, 1.0)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(fov, width/float(height), 0.1, 1000.0)
    glMatrixMode(GL_MODELVIEW)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)


def draw_cube(size=1.0):
    s = size/2.0
    glBegin(GL_QUADS)
    # +X
    glNormal3f(1,0,0); glVertex3f(+s,-s,-s); glVertex3f(+s,+s,-s); glVertex3f(+s,+s,+s); glVertex3f(+s,-s,+s)
    # -X
    glNormal3f(-1,0,0); glVertex3f(-s,-s,-s); glVertex3f(-s,-s,+s); glVertex3f(-s,+s,+s); glVertex3f(-s,+s,-s)
    # +Y
    glNormal3f(0,1,0); glVertex3f(-s,+s,-s); glVertex3f(-s,+s,+s); glVertex3f(+s,+s,+s); glVertex3f(+s,+s,-s)
    # -Y
    glNormal3f(0,-1,0); glVertex3f(-s,-s,-s); glVertex3f(+s,-s,-s); glVertex3f(+s,-s,+s); glVertex3f(-s,-s,+s)
    # +Z
    glNormal3f(0,0,1); glVertex3f(-s,-s,+s); glVertex3f(+s,-s,+s); glVertex3f(+s,+s,+s); glVertex3f(-s,+s,+s)
    # -Z
    glNormal3f(0,0,-1); glVertex3f(-s,-s,-s); glVertex3f(-s,+s,-s); glVertex3f(+s,+s,-s); glVertex3f(+s,-s,-s)
    glEnd()

def draw_wire_box(size=1.0):
    s = size/2.0
    glBegin(GL_LINES)
    for a,b in [
        ((-s,-s,-s),(+s,-s,-s)), ((+s,-s,-s),(+s,+s,-s)), ((+s,+s,-s),(-s,+s,-s)), ((-s,+s,-s),(-s,-s,-s)),
        ((-s,-s,+s),(+s,-s,+s)), ((+s,-s,+s),(+s,+s,+s)), ((+s,+s,+s),(-s,+s,+s)), ((-s,+s,+s),(-s,-s,+s)),
        ((-s,-s,-s),(-s,-s,+s)), ((+s,-s,-s),(+s,-s,+s)), ((+s,+s,-s),(+s,+s,+s)), ((-s,+s,-s),(-s,+s,+s)),
    ]:
        glVertex3f(*a); glVertex3f(*b)
    glEnd()

def draw_grid(n, spacing, color=(0.35,0.38,0.45)):
    glColor3f(*color)
    glBegin(GL_LINES)
    half = (n-1)*spacing/2.0
    for i in range(n):
        t = -half + i*spacing
        # XY at Z=-half and Z=+half
        glVertex3f(-half, t, -half); glVertex3f(+half, t, -half)
        glVertex3f(t, -half, -half); glVertex3f(t, +half, -half)
        glVertex3f(-half, t, +half); glVertex3f(+half, t, +half)
        glVertex3f(t, -half, +half); glVertex3f(t, +half, +half)
        # edges along Z
        glVertex3f(-half, -half, t); glVertex3f(+half, -half, t)
        glVertex3f(-half, +half, t); glVertex3f(+half, +half, t)
        glVertex3f(-half, -half, t); glVertex3f(-half, +half, t)
        glVertex3f(+half, -half, t); glVertex3f(+half, +half, t)
    glEnd()

# --- new: draw a small right-hand-rule compass in its own mini-viewport
def draw_compass(x, y, size, yaw, pitch):
    glViewport(x, y, size, size)
    glScissor(x, y, size, size)
    glEnable(GL_SCISSOR_TEST)
    glClear(GL_DEPTH_BUFFER_BIT)
    glDisable(GL_CULL_FACE)

    glMatrixMode(GL_PROJECTION); glLoadIdentity()
    gluPerspective(30.0, 1.0, 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW); glLoadIdentity()
    # lock camera to look at origin but rotate compass opposite of world yaw/pitch
    gluLookAt(0,0,6, 0,0,0, 0,1,0)
    glRotatef(-pitch, 1,0,0)
    glRotatef(-yaw,   0,1,0)

    glLineWidth(3.0)
    glBegin(GL_LINES)
    # X axis (red)
    glColor3f(1,0.2,0.2); glVertex3f(0,0,0); glVertex3f(1.2,0,0)
    # Y axis (green)
    glColor3f(0.2,1,0.2); glVertex3f(0,0,0); glVertex3f(0,1.2,0)
    # Z axis (blue)
    glColor3f(0.3,0.6,1); glVertex3f(0,0,0); glVertex3f(0,0,1.2)
    glEnd()

    glDisable(GL_SCISSOR_TEST)
    glEnable(GL_CULL_FACE)

def draw_slice_planes(n, spacing, sel_x, sel_y, sel_z, emphasis=1.0):
    """
    Draw 3 translucent orthogonal planes through (sel_x, sel_y, sel_z),
    two-sided and without writing to the depth buffer.
    """
    half = (n-1)*spacing/2.0

    def grid_to_world(t):
        return -half + t * spacing

    # ----- state tweaks for translucent two-sided quads -----
    glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glDisable(GL_CULL_FACE)            # make planes visible from both sides
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDepthMask(GL_FALSE)              # don't write depth for translucent overlays
    glDisable(GL_LIGHTING)

    eps = 0.0008  # nudge to avoid z-fighting

    # X = const (YZ plane)
    xw = grid_to_world(sel_x)
    glColor4f(1.0, 0.25, 0.25, 0.14 * emphasis)   # reddish
    glBegin(GL_QUADS)
    glVertex3f(xw+eps, -half, -half)
    glVertex3f(xw+eps, +half, -half)
    glVertex3f(xw+eps, +half, +half)
    glVertex3f(xw+eps, -half, +half)
    glEnd()

    # Y = const (XZ plane)
    yw = grid_to_world(sel_y)
    glColor4f(0.25, 1.0, 0.35, 0.14 * emphasis)   # greenish
    glBegin(GL_QUADS)
    glVertex3f(-half, yw+eps, -half)
    glVertex3f(+half, yw+eps, -half)
    glVertex3f(+half, yw+eps, +half)
    glVertex3f(-half, yw+eps, +half)
    glEnd()

    # Z = const (XY plane)
    zw = grid_to_world(sel_z)
    glColor4f(0.35, 0.6, 1.0, 0.14 * emphasis)    # bluish
    glBegin(GL_QUADS)
    glVertex3f(-half, -half, zw+eps)
    glVertex3f(+half, -half, zw+eps)
    glVertex3f(+half, +half, zw+eps)
    glVertex3f(-half, +half, zw+eps)
    glEnd()

    # ----- restore previous state -----
    glDepthMask(GL_TRUE)
    glPopAttrib()

# --- replace your GLText with this drop-in ---
class GLText:
    """Cache text → OpenGL texture, draw in screen-space (orthographic)."""
    def __init__(self, font_name="consolas,menlo,monospace", pt=16, color=(220,220,230)):
        pygame.font.init()
        self.font = pygame.font.SysFont(font_name, pt)
        self.default_color = color
        self.cache = {}  # (text, color_tuple) -> (tex_id, w, h)

    def _upload_surface(self, surf):
        w, h = surf.get_size()
        surf = surf.convert_alpha()
        px = pygame.image.tostring(surf, "RGBA", True)
        tex_id = glGenTextures(1)
        glBindTexture(GL_TEXTURE_2D, tex_id)
        glPixelStorei(GL_UNPACK_ALIGNMENT, 1)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE)
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, w, h, 0, GL_RGBA, GL_UNSIGNED_BYTE, px)
        glBindTexture(GL_TEXTURE_2D, 0)
        return tex_id, w, h

    def get(self, text: str, color=None):
        color = color or self.default_color  # pygame expects 0–255 ints
        key = (text, color)
        if key in self.cache:
            return self.cache[key]
        surf = self.font.render(text, True, color)
        tex = self._upload_surface(surf)
        self.cache[key] = tex
        return tex

    def draw(self, text: str, x: int, y: int, window_w: int, window_h: int, color=None):
        tex_id, w, h = self.get(text, color=color)
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_TEXTURE_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnable(GL_TEXTURE_2D)
        glColor4f(1, 1, 1, 1)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, window_w, window_h, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        glBindTexture(GL_TEXTURE_2D, tex_id)
        glBegin(GL_QUADS)
        glTexCoord2f(0, 1); glVertex2f(x,   y)
        glTexCoord2f(1, 1); glVertex2f(x+w, y)
        glTexCoord2f(1, 0); glVertex2f(x+w, y+h)
        glTexCoord2f(0, 0); glVertex2f(x,   y+h)
        glEnd()
        glBindTexture(GL_TEXTURE_2D, 0)

        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glPopAttrib()
        return w, h  # handy if you want to align boxes next to text



# -- main with 2×2 viewports for w=0..3 and WASD+arrow x/y, A/D z, W/S w
def main():
    n, d, w = 4, 4, 4
    assert n == 4, "This UI lays out 4 w-slices. Set n=4 for now."
    pygame.init()
    W, H = 1280, 900
    pygame.display.set_mode((W, H), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("4D Tic-Tac-Toe — 4 viewports (w=0..3)")

    gl_init(W, H)
    rng = Random(1337)
    game = NDTTT(n=n, d=d, w=w)

    # text
    text = GLText(pt=16)  # or your preferred size/color


    # camera
    yaw, pitch, radius = 45.0, 25.0, 18.0
    dragging = False
    last_mouse = (0,0)
    # camera defaults
    DEFAULT_YAW, DEFAULT_PITCH, DEFAULT_RADIUS = 45.0, 25.0, 18.0
    yaw, pitch, radius = DEFAULT_YAW, DEFAULT_PITCH, DEFAULT_RADIUS

    # compass widget (top-left)
    COMP_SIZE = 100
    COMP_MARGIN = 10


    # selection & active w
    sel_x, sel_y, sel_z, sel_w = 0, 0, 0, 0
    winner = None
    win_mask = 0

    spacing = 2.0
    stone_size = 1.2
    half = (n-1)*spacing/2.0

    clock = pygame.time.Clock()

    def world_pos(x,y,z):
        return (-half + x*spacing, -half + y*spacing, -half + z*spacing)

    # compute 2x2 viewport rects for the four w-slices
    gutter = 12
    view_w = (W - gutter*3) // 2
    view_h = (H - gutter*3) // 2
    viewports = [
        (gutter,              H - (view_h+gutter), view_w, view_h),  # top-left  (w=0)
        (gutter*2 + view_w,   H - (view_h+gutter), view_w, view_h),  # top-right (w=1)
        (gutter,              gutter,              view_w, view_h),  # bottom-left (w=2)
        (gutter*2 + view_w,   gutter,              view_w, view_h),  # bottom-right (w=3)
    ]

    def setup_camera(vx, vy, vw, vh):
        glViewport(vx, vy, vw, vh)
        glMatrixMode(GL_PROJECTION); glLoadIdentity()
        gluPerspective(50.0, vw/float(vh), 0.1, 1000.0)
        glMatrixMode(GL_MODELVIEW); glLoadIdentity()
        eye_x = radius * math.cos(math.radians(pitch)) * math.cos(math.radians(yaw))
        eye_y = radius * math.sin(math.radians(pitch))
        eye_z = radius * math.cos(math.radians(pitch)) * math.sin(math.radians(yaw))
        gluLookAt(eye_x, eye_y, eye_z, 0, 0, 0, 0, 1, 0)

    def draw_one_wslice(ws):
        # grid
        glLineWidth(1.0)
        draw_grid(n, spacing, (0.35,0.38,0.45))
        # translucent planes through the selected voxel; stronger in active w
        emph = 1.0 if ws == sel_w else 0.6
        draw_slice_planes(n, spacing, sel_x, sel_y, sel_z, emphasis=emph)

        # stones for this w slice
        for z in range(n):
            for y in range(n):
                for x in range(n):
                    idx = game.idx([x,y,z,ws])
                    occupied0 = (game.P[0] >> idx) & 1
                    occupied1 = (game.P[1] >> idx) & 1
                    if occupied0 or occupied1 or (x==sel_x and y==sel_y and z==sel_z and ws==sel_w):
                        px,py,pz = world_pos(x,y,z)
                        glPushMatrix(); glTranslatef(px, py, pz)
                        # winning highlight
                        if win_mask and ((win_mask >> idx) & 1):
                            glColor3f(0.98, 0.86, 0.25)
                            draw_wire_box(stone_size*1.25)
                        # stone
                        if occupied0:
                            glColor3f(0.15, 0.55, 1.0)
                            draw_cube(stone_size)
                        elif occupied1:
                            glColor3f(0.9, 0.25, 0.25)
                            draw_cube(stone_size)
                        # selection cursor on active viewport (ws==sel_w)
                        if ws == sel_w and x==sel_x and y==sel_y and z==sel_z and winner is None:
                            glColor3f(0.92,0.95,1.0); glLineWidth(2.0)
                            draw_wire_box(stone_size*1.35)
                        glPopMatrix()

    def draw_frame():
        # clear once
        glViewport(0,0,W,H)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        # draw the 4 w-slices in their viewports
        for ws in range(4):
            vx, vy, vw, vh = viewports[ws]
            setup_camera(vx, vy, vw, vh)
            draw_one_wslice(ws)
            # draw border; highlight active w
            glMatrixMode(GL_PROJECTION); glLoadIdentity()
            glOrtho(0, vw, 0, vh, -1, 1)
            glMatrixMode(GL_MODELVIEW); glLoadIdentity()
            glDisable(GL_DEPTH_TEST)
            if ws == sel_w: glColor3f(0.98,0.86,0.25)  # gold highlight
            else:           glColor3f(0.5,0.55,0.6)
            glLineWidth(3.0)
            glBegin(GL_LINE_LOOP)
            glVertex2f(3,3); glVertex2f(vw-3,3); glVertex2f(vw-3,vh-3); glVertex2f(3,vh-3)
            glEnd()
            glEnable(GL_DEPTH_TEST)

        # draw the compass gizmo on top-left of the whole window
        # bottom-left coords for OpenGL viewport: (x, y_from_bottom)
        draw_compass(COMP_MARGIN, H - (COMP_MARGIN + COMP_SIZE), COMP_SIZE, yaw, pitch)

        help_lines = [
            "←/→: x-/+   |   ↑/↓: y-/+   |   A/D: z−/+   |   W/S: w−/+",
            "Enter: place   Space: random   U: undo   R: reset   ESC: quit",
        ]

        margin, y = 30, 20
        glViewport(0, 0, W, H)
        for line in help_lines:
            tex_id, w, h = text.get(line)   # get cached texture and its size
            x = W - w - margin              # right align
            text.draw(line, x, y, W, H)
            y += 20

        # --- bottom-left HUD (after your top-right help text) ---
        # choose player color (0–255 ints for font; GL quad uses 0–1 floats)
        P0_FONT = (38, 141, 255)   # blue-ish for X
        P1_FONT = (230, 64, 64)    # red-ish for O
        P0_GL   = (0.15, 0.55, 1.0)
        P1_GL   = (0.90, 0.25, 0.25)

        turn = 'X' if game.stm == 0 else 'O'
        font_color = P0_FONT if game.stm == 0 else P1_FONT
        chip_color = P0_GL   if game.stm == 0 else P1_GL

        cursor_line = f"Turn: {turn}   Cursor (x,y,z,w) = ({sel_x},{sel_y},{sel_z},{sel_w})"

        # bottom-left anchor
        margin_x = 30
        margin_y = 30
        # measure text to position chip nicely
        tex_w, tex_h = text.get(cursor_line, color=font_color)[1:3]
        x = margin_x
        y = H - margin_y - tex_h

        # draw colored chip box to the left of the text
        glPushAttrib(GL_ENABLE_BIT | GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glDisable(GL_DEPTH_TEST)
        glDisable(GL_CULL_FACE)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

        glMatrixMode(GL_PROJECTION); glPushMatrix(); glLoadIdentity()
        glOrtho(0, W, H, 0, -1, 1)
        glMatrixMode(GL_MODELVIEW); glPushMatrix(); glLoadIdentity()

        chip_w = 14; chip_h = tex_h - 4
        glColor4f(*chip_color, 1.0)
        glBegin(GL_QUADS)
        glVertex2f(x,           y+2)
        glVertex2f(x+chip_w,    y+2)
        glVertex2f(x+chip_w,    y+2+chip_h)
        glVertex2f(x,           y+2+chip_h)
        glEnd()

        glMatrixMode(GL_MODELVIEW); glPopMatrix()
        glMatrixMode(GL_PROJECTION); glPopMatrix()
        glPopAttrib()

        # draw the text itself, a little to the right of the chip
        text.draw(cursor_line, x + chip_w + 8, y, W, H, color=font_color)

        # --- bottom-right mini status (no backdrop) ---
        br_lines = [
            f"Active w slice: {sel_w}",
            "Click compass to reset view",
            "Made by Andy Li"
        ]

        margin_x, margin_y = 24, 24
        y = H - margin_y

        # Draw each line upward from the bottom, right-aligned
        for s in reversed(br_lines):
            tex_id, w_txt, h_txt = text.get(s)
            x = W - margin_x - w_txt
            y -= h_txt
            text.draw(s, x, y, W, H)
            y -= 6  # small gap between lines


        pygame.display.flip()

    running = True
    while running:
        clock.tick(60)
        for e in pygame.event.get():
            if e.type == pygame.QUIT:
                running = False
            elif e.type == pygame.MOUSEBUTTONDOWN:
                mx, my = pygame.mouse.get_pos()

                # Pygame coords are top-left origin; compass rect in that space:
                comp_left   = COMP_MARGIN
                comp_top    = COMP_MARGIN
                comp_right  = COMP_MARGIN + COMP_SIZE
                comp_bottom = COMP_MARGIN + COMP_SIZE

                if comp_left <= mx <= comp_right and comp_top <= my <= comp_bottom:
                    # reset camera
                    yaw, pitch, radius = DEFAULT_YAW, DEFAULT_PITCH, DEFAULT_RADIUS
                    dragging = False   # don't start an orbit drag when clicking the compass
                    continue           # skip the rest of MOUSEBUTTONDOWN handling

                if e.button == 1:
                    dragging = True; last_mouse = e.pos
                elif e.button == 4:
                    radius = max(6.0, radius - 1.0)
                elif e.button == 5:
                    radius = min(60.0, radius + 1.0)
            elif e.type == pygame.MOUSEBUTTONUP:
                if e.button == 1: dragging = False
            elif e.type == pygame.MOUSEMOTION and dragging:
                mx,my = e.pos
                dx = mx - last_mouse[0]; dy = my - last_mouse[1]
                yaw = (yaw + dx * 0.4) % 360
                pitch = max(-85.0, min(85.0, pitch - dy * 0.3))
                last_mouse = e.pos
            elif e.type == pygame.KEYDOWN:
                if e.key == pygame.K_ESCAPE:
                    running = False
                elif e.key == pygame.K_r:
                    game = NDTTT(n=n, d=d, w=w); winner = None; win_mask = 0
                elif e.key == pygame.K_u:
                    if game.hist:
                        game.pop(); winner = None; win_mask = 0
                elif e.key == pygame.K_SPACE and winner is None:
                    moves = list(game.legal_moves())
                    if moves:
                        mv = rng.choice(moves)
                        win, is_draw = game.push(mv)
                        if win: win_mask, winner = winning_mask_for_last_move(game)
                        elif is_draw: win_mask, winner = 0, None
                # x/y via arrows or WASD
                elif e.key == pygame.K_LEFT:
                    sel_x = (sel_x - 1) % n
                elif e.key == pygame.K_RIGHT:
                    sel_x = (sel_x + 1) % n
                elif e.key == pygame.K_UP:
                    sel_y = (sel_y + 1) % n
                elif e.key == pygame.K_DOWN:
                    sel_y = (sel_y - 1) % n
                elif e.key == pygame.K_d:
                    sel_z = (sel_z - 1) % n
                elif e.key == pygame.K_a:
                    sel_z = (sel_z + 1) % n
                # go ana
                elif e.key == pygame.K_s:
                    sel_w = (sel_w - 1) % n
                # go kata
                elif e.key == pygame.K_w:
                    sel_w = (sel_w + 1) % n
                elif e.key == pygame.K_RETURN and winner is None:
                    idx = game.idx([sel_x, sel_y, sel_z, sel_w])
                    if not game.occupied(idx):
                        win, is_draw = game.push(idx)
                        if win: win_mask, winner = winning_mask_for_last_move(game)
                        elif is_draw: win_mask, winner = 0, None

        draw_frame()

    pygame.quit()


if __name__ == "__main__":
    main()
