import sys, os, math, time, json, random
import pygame
import cv2
import numpy as np

# -------- Settings --------
WIDTH, HEIGHT = 480, 800
FPS = 60

GRAVITY = 0.35
JUMP_VELOCITY = -7.5
PIPE_GAP = 180
PIPE_SPEED = 3.5
PIPE_SPAWN_EVERY = 1.25  # seconds
BIRD_X = int(WIDTH * 0.28)

WEBCAM_INDEX = 0
WEBCAM_W, WEBCAM_H = 240, 180
HEAD_SMOOTH = 0.15          # 0..1 (higher = smoother)
JUMP_SENSITIVITY = 0.018    # movement threshold to trigger jump (relative units)
CALIBRATION_SAMPLES = 30
SAVE_FILE = "face_flappy_best.json"

# -------- Init --------
pygame.init()
pygame.display.set_caption("Face Flappy — Head to Fly")
screen = pygame.display.set_mode((WIDTH, HEIGHT))
clock = pygame.time.Clock()

# Fonts
def load_font(size, bold=False):
    f = pygame.font.SysFont("rubik,arial,verdana,dejavusans", size, bold=bold)
    if f is None:
        f = pygame.font.SysFont(None, size, bold=bold)
    return f

FONT_SM = load_font(18)
FONT_MD = load_font(28)
FONT_LG = load_font(46, bold=True)
FONT_XL = load_font(70, bold=True)

# Soft shadow text
def draw_text(surface, text, font, color, pos, shadow=True, center=False, alpha=255):
    txt_surf = font.render(text, True, color)
    if alpha != 255:
        txt_surf.set_alpha(alpha)
    x, y = pos
    if center:
        rect = txt_surf.get_rect(center=(x, y))
    else:
        rect = txt_surf.get_rect(topleft=(x, y))
    if shadow:
        shadow_surf = font.render(text, True, (0,0,0))
        shadow_rect = shadow_surf.get_rect(center=rect.center)
        shadow_surf.set_alpha(90)
        surface.blit(shadow_surf, (shadow_rect.x+2, shadow_rect.y+2))
    surface.blit(txt_surf, rect)

# Gradient background
def draw_vertical_gradient(surface, top_color, bottom_color):
    """Modern soft gradient background."""
    for i in range(HEIGHT):
        t = i / HEIGHT
        r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
        g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
        b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
        pygame.draw.line(surface, (r, g, b), (0, i), (WIDTH, i))

# Rounded rect helper
def draw_card(surface, rect, color=(255,255,255), radius=18, shadow=True):
    if shadow:
        shadow_rect = rect.move(0, 6)
        pygame.draw.rect(surface, (0,0,0,40), shadow_rect, border_radius=radius)
    pygame.draw.rect(surface, color, rect, border_radius=radius)

# Bird
class Bird:
    def __init__(self):
        self.y = HEIGHT * 0.5
        self.vy = 0
        self.radius = 18
        self.tilt = 0
        self.color = (255, 221, 0)  # modern yellow
        self.glow = (255, 255, 255)

    def update(self, head_jump=False):
        if head_jump:
            self.vy = JUMP_VELOCITY
        self.vy += GRAVITY
        self.y += self.vy
        self.y = max(self.radius, min(HEIGHT - self.radius, self.y))
        # tilt for flair
        self.tilt = max(-30, min(60, -self.vy * 4))

    def draw(self, surface):
        # simple modern bubble bird with eye
        x = BIRD_X
        y = int(self.y)
        # glow
        pygame.draw.circle(surface, (255,255,255), (x, y+1), self.radius+6)
        # body
        pygame.draw.circle(surface, self.color, (x, y), self.radius)
        # wing (rotating ellipse)
        wing_w, wing_h = 16, 10
        wing = pygame.Surface((wing_w*2, wing_h*2), pygame.SRCALPHA)
        pygame.draw.ellipse(wing, (255, 200, 0), (0, wing_h//2, wing_w*2, wing_h))
        wing = pygame.transform.rotate(wing, self.tilt)
        surface.blit(wing, (x - wing.get_width()//2 + 4, y - wing.get_height()//2 + 6))
        # eye
        pygame.draw.circle(surface, (255,255,255), (x+6, y-4), 6)
        pygame.draw.circle(surface, (10,10,10), (x+6, y-4), 3)

    def get_rect(self):
        return pygame.Rect(BIRD_X - self.radius, int(self.y) - self.radius, self.radius*2, self.radius*2)

# Pipes
class Pipe:
    def __init__(self, x):
        self.x = x
        center = random.randint(int(HEIGHT*0.25), int(HEIGHT*0.75))
        self.top_end = center - PIPE_GAP//2
        self.bot_start = center + PIPE_GAP//2
        self.w = 70
        # pastel green
        self.color = (98, 212, 141)

    def update(self, dt):
        self.x -= PIPE_SPEED

    def offscreen(self):
        return self.x + self.w < 0

    def draw(self, surface):
        r = 16
        # top pipe
        top_rect = pygame.Rect(self.x, 0, self.w, self.top_end)
        pygame.draw.rect(surface, self.color, top_rect, border_radius=r)
        # bottom pipe
        bot_rect = pygame.Rect(self.x, self.bot_start, self.w, HEIGHT-self.bot_start)
        pygame.draw.rect(surface, self.color, bot_rect, border_radius=r)
        # subtle highlights
        hi = (255,255,255,40)
        s_top = pygame.Surface((self.w, self.top_end), pygame.SRCALPHA)
        pygame.draw.rect(s_top, hi, (6, 0, 8, self.top_end), border_radius=r)
        surface.blit(s_top, (self.x, 0))
        s_bot = pygame.Surface((self.w, HEIGHT-self.bot_start), pygame.SRCALPHA)
        pygame.draw.rect(s_bot, hi, (6, 0, 8, HEIGHT-self.bot_start), border_radius=r)
        surface.blit(s_bot, (self.x, self.bot_start))

    def collides(self, rect: pygame.Rect) -> bool:
        top_rect = pygame.Rect(self.x, 0, self.w, self.top_end)
        bot_rect = pygame.Rect(self.x, self.bot_start, self.w, HEIGHT-self.bot_start)
        return rect.colliderect(top_rect) or rect.colliderect(bot_rect)

# Load best score
def load_best():
    if os.path.exists(SAVE_FILE):
        try:
            with open(SAVE_FILE, "r") as f:
                return json.load(f).get("best", 0)
        except Exception:
            return 0
    return 0

def save_best(best):
    try:
        with open(SAVE_FILE, "w") as f:
            json.dump({"best": best}, f)
    except Exception:
        pass

# Webcam + Mediapipe
try:
    import mediapipe as mp
    mp_face = mp.solutions.face_detection
    face_detector = mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    has_mediapipe = True
except Exception:
    face_detector = None
    has_mediapipe = False

cap = None
if has_mediapipe:
    cap = cv2.VideoCapture(WEBCAM_INDEX)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WEBCAM_W)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, WEBCAM_H)

def get_head_center_y():
    """
    Returns (found, y_rel)
    y_rel in [0..1], 0 = top of frame, 1 = bottom
    Applies smoothing globally with EMA.
    """
    global _smoothed_y
    if cap is None:
        return False, 0.5

    ok, frame = cap.read()
    if not ok:
        return False, 0.5
    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = face_detector.process(rgb) if face_detector else None

    y_rel = None
    if res and res.detections:
        det = res.detections[0]
        bbox = det.location_data.relative_bounding_box
        cy = bbox.ymin + bbox.height * 0.5  # center y of face box
        y_rel = float(np.clip(cy, 0.0, 1.0))
    else:
        y_rel = None

    if y_rel is None:
        return False, 0.5

    # EMA smoothing
    if _smoothed_y is None:
        _smoothed_y = y_rel
    else:
        _smoothed_y = (1.0 - HEAD_SMOOTH) * y_rel + HEAD_SMOOTH * _smoothed_y

    return True, _smoothed_y

def get_webcam_surface():
    """Return a pygame surface of the latest webcam frame (for PiP)."""
    if cap is None:
        return None
    ok, frame = cap.read()
    if not ok:
        return None
    frame = cv2.flip(frame, 1)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = cv2.resize(frame, (WEBCAM_W, WEBCAM_H))
    surf = pygame.image.frombuffer(frame.tobytes(), (WEBCAM_W, WEBCAM_H), "RGB")
    return surf

_smoothed_y = None  # global EMA store

# Game states
STATE_MENU = "menu"
STATE_PLAY = "play"
STATE_OVER = "over"

def main():
    global _smoothed_y
    running = True
    state = STATE_MENU
    bird = Bird()
    pipes = []
    score = 0
    best = load_best()
    last_spawn = 0
    passed_pipe_idx = -1
    last_head_y = None
    head_baseline = None
    calibrated = False
    spawn_accum = 0.0

    # colors
    grad_top = (120, 170, 255)     # soft blue
    grad_bottom = (248, 202, 255)  # blush violet
    ui_card = (255, 255, 255)

    # reset helpers
    def reset_game():
        nonlocal bird, pipes, score, last_spawn, passed_pipe_idx, spawn_accum
        bird = Bird()
        pipes = []
        score = 0
        last_spawn = 0
        passed_pipe_idx = -1
        spawn_accum = 0.0

    # main loop
    while running:
        dt = clock.tick(FPS) / 1000.0

        # EVENTS
        head_jump_request = False
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if state == STATE_MENU:
                    if event.key == pygame.K_SPACE and calibrated:
                        state = STATE_PLAY
                        reset_game()
                    elif event.key == pygame.K_c:
                        # calibration sequence
                        head_samples = []
                        for _ in range(CALIBRATION_SAMPLES):
                            found, y = get_head_center_y()
                            if found:
                                head_samples.append(y)
                            pygame.event.pump()
                            time.sleep(0.02)
                        if head_samples:
                            head_baseline = float(np.median(head_samples))
                            _smoothed_y = head_baseline
                            calibrated = True
                        else:
                            calibrated = False
                    elif event.key == pygame.K_ESCAPE:
                        running = False
                elif state == STATE_PLAY:
                    if event.key == pygame.K_SPACE:
                        head_jump_request = True  # allow keyboard jump
                    elif event.key == pygame.K_ESCAPE:
                        state = STATE_MENU
                elif state == STATE_OVER:
                    if event.key == pygame.K_SPACE:
                        state = STATE_MENU
                    elif event.key == pygame.K_ESCAPE:
                        running = False

        # UPDATE
        if state == STATE_PLAY:
            # Head movement -> jump detection
            found, y_rel = get_head_center_y() if has_mediapipe else (False, 0.5)
            if found and head_baseline is not None:
                if last_head_y is None:
                    last_head_y = y_rel
                # Movement: negative y means up in image coordinates
                # We want a jump when the head moves UP quickly (y decreases)
                dy = last_head_y - y_rel  # >0 when moving up
                if dy > JUMP_SENSITIVITY:
                    head_jump_request = True
                last_head_y = y_rel
            else:
                # if no face detected, allow keyboard only
                pass

            bird.update(head_jump_request)

            # pipes
            spawn_accum += dt
            if spawn_accum >= PIPE_SPAWN_EVERY:
                spawn_accum -= PIPE_SPAWN_EVERY
                pipes.append(Pipe(WIDTH + 30))

            for p in pipes:
                p.update(dt)
            pipes = [p for p in pipes if not p.offscreen()]

            # scoring
            for idx, p in enumerate(pipes):
                if p.x + p.w < BIRD_X and idx > passed_pipe_idx:
                    score += 1
                    passed_pipe_idx = idx

            # collisions + ground/ceiling
            bird_rect = bird.get_rect()
            hit = False
            if bird_rect.top <= 0 or bird_rect.bottom >= HEIGHT:
                hit = True
            else:
                for p in pipes:
                    if p.collides(bird_rect):
                        hit = True
                        break

            if hit:
                best = max(best, score)
                save_best(best)
                state = STATE_OVER

        # DRAW
        # background
        draw_vertical_gradient(screen, grad_top, grad_bottom)

        # subtle floating blobs for depth (modern aesthetic)
        t = pygame.time.get_ticks() / 1000.0
        for i in range(6):
            bx = (math.sin(t*0.6 + i) * 0.5 + 0.5) * WIDTH
            by = (math.cos(t*0.7 + i*1.3) * 0.5 + 0.5) * HEIGHT
            r = 60 + 20 * math.sin(t*1.1 + i*0.9)
            blob = pygame.Surface((int(r*2), int(r*2)), pygame.SRCALPHA)
            pygame.draw.circle(blob, (255,255,255,28), (int(r), int(r)), int(r))
            screen.blit(blob, (bx - r, by - r))

        if state == STATE_MENU:
            # Title card
            card_w, card_h = int(WIDTH*0.84), 320
            card_rect = pygame.Rect((WIDTH-card_w)//2, 120, card_w, card_h)
            draw_card(screen, card_rect, radius=24)

            draw_text(screen, "Face Flappy", FONT_XL, (30,30,30), (WIDTH//2, card_rect.y+70), center=True)
            draw_text(screen, "Move your head UP to flap", FONT_MD, (70,70,70), (WIDTH//2, card_rect.y+130), center=True)
            draw_text(screen, f"Best: {load_best()}", FONT_MD, (70,70,70), (WIDTH//2, card_rect.y+170), center=True)

            # controls box
            ctrl_rect = pygame.Rect((WIDTH-card_w)//2, card_rect.bottom+20, card_w, 160)
            draw_card(screen, ctrl_rect, radius=18)
            draw_text(screen, "1) Press C to calibrate (face centered, natural pose)", FONT_SM, (60,60,60), (ctrl_rect.x+18, ctrl_rect.y+22))
            draw_text(screen, "2) Press SPACE to start", FONT_SM, (60,60,60), (ctrl_rect.x+18, ctrl_rect.y+56))
            draw_text(screen, "Tip: You can also press SPACE to flap if tracking is lost.", FONT_SM, (60,60,60), (ctrl_rect.x+18, ctrl_rect.y+90))

            # idle bird
            idle_bird = Bird()
            idle_bird.y = card_rect.bottom - 30
            idle_bird.draw(screen)

            # webcam preview (PiP)
            surf = get_webcam_surface() if has_mediapipe else None
            if surf is not None:
                pip = pygame.transform.smoothscale(surf, (WEBCAM_W, WEBCAM_H))
                pip_rect = pygame.Rect(WIDTH- (WEBCAM_W+24), 24, WEBCAM_W, WEBCAM_H)
                draw_card(screen, pip_rect.inflate(16,16), color=(255,255,255), radius=16)
                screen.blit(pip, pip_rect)

            # footnote
            draw_text(screen, "Press ESC to quit", FONT_SM, (40,40,40), (20, HEIGHT-32))

        elif state == STATE_PLAY:
            # pipes
            for p in pipes:
                p.draw(screen)

            # ground line
            pygame.draw.rect(screen, (255,255,255, 60), (0, HEIGHT-20, WIDTH, 20), border_radius=10)

            # bird
            bird.draw(screen)

            # score
            draw_text(screen, str(score), FONT_LG, (35,35,35), (WIDTH//2, 30), center=True)

            # webcam PiP
            surf = get_webcam_surface() if has_mediapipe else None
            if surf is not None:
                pip = pygame.transform.smoothscale(surf, (int(WEBCAM_W*0.9), int(WEBCAM_H*0.9)))
                pip_rect = pygame.Rect(16, 16, pip.get_width(), pip.get_height())
                draw_card(screen, pip_rect.inflate(14,14), color=(255,255,255), radius=14)
                screen.blit(pip, pip_rect)

        elif state == STATE_OVER:
            # faint score in background
            draw_text(screen, str(score), FONT_XL, (30,30,30), (WIDTH//2, HEIGHT//2 - 140), center=True, alpha=50)

            # card
            card_w, card_h = int(WIDTH*0.84), 260
            card_rect = pygame.Rect((WIDTH-card_w)//2, (HEIGHT-card_h)//2 - 30, card_w, card_h)
            draw_card(screen, card_rect, radius=24)
            draw_text(screen, "Game Over", FONT_LG, (30,30,30), (WIDTH//2, card_rect.y+60), center=True)
            draw_text(screen, f"Score: {score}", FONT_MD, (70,70,70), (WIDTH//2, card_rect.y+110), center=True)
            draw_text(screen, f"Best: {load_best()}", FONT_MD, (70,70,70), (WIDTH//2, card_rect.y+146), center=True)
            draw_text(screen, "SPACE: Menu   ESC: Quit", FONT_SM, (80,80,80), (WIDTH//2, card_rect.y+192), center=True)

        pygame.display.flip()

    # Cleanup
    if cap is not None:
        cap.release()
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        if cap is not None:
            cap.release()
        pygame.quit()
        sys.exit(0)
