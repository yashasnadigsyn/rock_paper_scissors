## Imports
import pygame
from pygame.locals import *
from pygame_emojis import load_emoji
import cv2
import numpy as np
import sys, os
import random
from ultralytics import YOLO

## Model and Game Choices
model = YOLO('best.onnx')
choices = ['rock', 'paper', 'scissors']

## Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GREEN = (0, 255, 0)
RED = (255, 0, 0)

## Pygame Initialization
pygame.init()
WIDTH, HEIGHT = 1366, 768
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Rock_Paper_Scissors")
pygame.display.toggle_fullscreen()
pygame.mouse.set_visible(False)

## Game State
game_state = "START_SCREEN"

## Game Constants
FPS = 30
WINNING_SCORE = 5
VS_SCREEN_DURATION = 2000
RESULTS_SCREEN_DURATION_1P = 2000
RESULTS_SCREEN_DURATION_2P = 2500
DETECTION_WINDOW = 2000
WARNING_DURATION = 2000
CREDITS_SCROLL_SPEED = 7.0

## ASSETS
FACE_CASCADE = cv2.CascadeClassifier(os.path.join("assets", "haarcascade_face_detect.xml"))
START_IMAGE = pygame.transform.scale(pygame.image.load(os.path.join("assets", "starting.png")), (WIDTH, HEIGHT))
PLAY_WITH_IMAGE = pygame.transform.scale(pygame.image.load(os.path.join("assets", "play_with.jpeg")), (WIDTH, HEIGHT))
VS_IMAGE = pygame.transform.scale(pygame.image.load(os.path.join("assets", "versus.jpg")), (WIDTH, HEIGHT))
GOOD_EXAMPLE_IMG = pygame.transform.scale(pygame.image.load(os.path.join("assets", "rock_works.png")), (480, 360))
BAD_EXAMPLE_IMG = pygame.transform.scale(pygame.image.load(os.path.join("assets", "rock_not_works.png")), (480, 360))
HAYASAKA = pygame.transform.scale(pygame.image.load(os.path.join("assets", "yashasnadigsyn.png")), (200, 200))

## Define Fonts
BIG_FONT = pygame.font.Font(os.path.join("assets", "font", "anton.ttf"), 80)
MEDIUM_FONT = pygame.font.Font(os.path.join("assets", "font", "anton.ttf"), 40)
RULES_HEADING_FONT = pygame.font.Font(os.path.join("assets", "font", "rules.ttf"), 80)
RULES_TEXT_FONT = pygame.font.Font(os.path.join("assets", "font", "rules.ttf"), 40)
COUNTDOWN_FONT = pygame.font.Font(os.path.join("assets", "font", "anton.ttf"), 250)
CREDITS_FONT = pygame.font.Font(os.path.join("assets", "font", "opensans.ttf"), 30)

## MUSIC
pygame.mixer.init()
pygame.mixer.music.load(os.path.join("assets", "music", "background_music.mp3"))
pygame.mixer.music.set_volume(0.3)
pygame.mixer.music.play(-1)
typing_sound = pygame.mixer.Sound(os.path.join("assets", "music", "typing.mp3"))
typing_sound.set_volume(0.5)
vs_sound = pygame.mixer.Sound(os.path.join("assets", "music", "versus.mp3"))

## Webcam
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

## Rules Screen
RULES_1P_TEXT = [
    "You will play against the AI.",
    "Show Rock, Paper, or Scissors",
    "to the camera.",
    "First to get 5 points wins.",
    "Good luck!"
]
RULES_2P_TEXT = [
    "This is a 2 Player game.",
    "Player 1 is on the left,",
    "Player 2 is on the right.",
    "After the countdown,",
    "both players show their hand.",
    "First to get 5 points wins."
]
rules_to_display, fully_rendered_lines = [], []
current_rule_line, current_rule_char, last_rule_char_update = 0, 0, 0
RULE_CHAR_DELAY, RULE_LINE_DELAY = 50, 100

## 2P Pre-Game
player_to_capture, player1_face_img, player2_face_img, versus_screen_start_time = 1, None, None, 0

## 1P Game
player_score_1p, ai_score_1p, prev_player_score_1p, prev_ai_score_1p = 0, 0, -1, -1
player_choice_1p, ai_choice_1p, round_winner_1p, round_state_1p, results_start_time = "", "", "", "PLAYING", 0
player_score_surf_1p, ai_score_surf_1p = None, None

## 2P Game
player1_score_2p, player2_score_2p, prev_p1_score_2p, prev_p2_score_2p = 0, 0, -1, -1
player1_choice_2p, player2_choice_2p, round_winner_2p, round_state_2p = "", "", "", "COUNTDOWN"
countdown_number, countdown_start_time = 3, 0
warning_start_time, detection_start_time = 0, 0
p1_score_surf_2p, p2_score_surf_2p = None, None

## Credits Screen
CREDITS_TEXT = ["CREDITS", "", "A Game by yashasnadigsyn", "", "", "", "LOGO_PLACEHOLDER", "", "--- MUSIC & SOUND ---", "Background Music: Cyberwave Orchestra (Pixabay)", "Typing Sound: Dragon Studio (Pixabay)", "Versus Sound: Frank2023 (myinstants.com)", "", "--- ASSETS ---", "Start/Mode Screens: Generated via ChatGPT", "Versus Image: hito stountio (Pixabay)", "Face Detection Model: OpenCV Haar Cascade", "Hand Detection Model: Ultralytics YOLOv11", "", "Thanks for Playing!"]
rendered_credits, credits_y_pos = [], HEIGHT

## Helper function for determining winner
def determine_winner(p1, p2):
    """Compares two choices and determines the winner."""
    if p1 == p2: return "TIE"
    if (p1 == 'rock' and p2 == 'scissors') or (p1 == 'scissors' and p2 == 'paper') or (p1 == 'paper' and p2 == 'rock'):
        return "PLAYER 1 WINS"
    return "PLAYER 2 WINS"

## STARTING
def draw_start_screen(window):
    """Displays the initial start screen"""
    window.blit(START_IMAGE, (0, 0))
    if (pygame.time.get_ticks() // 500) % 2 == 0:
        text_surface = BIG_FONT.render("Press any key to start", True, WHITE)
        text_rect = text_surface.get_rect(center=(WIDTH / 2, 700))
        window.blit(text_surface, text_rect)

## INSTRUCTION
def draw_instruction_screen(window):
    """Displays instructions, warnings, and visual examples."""
    window.fill(BLACK)
    
    quit_surf = MEDIUM_FONT.render("Press 'Q' at any time to quit the game.", True, WHITE)
    window.blit(quit_surf, quit_surf.get_rect(center=(WIDTH / 2, 80)))

    warning_title_surf = RULES_HEADING_FONT.render("IMPORTANT!", True, RED)
    window.blit(warning_title_surf, warning_title_surf.get_rect(center=(WIDTH / 2, 180)))
    
    warning_text_surf = MEDIUM_FONT.render("For best results, play against a plain background.", True, WHITE)
    window.blit(warning_text_surf, warning_text_surf.get_rect(center=(WIDTH / 2, 260)))

    img_y_pos = 460
    good_img_rect = GOOD_EXAMPLE_IMG.get_rect(center=(WIDTH * 0.3, img_y_pos))
    bad_img_rect = BAD_EXAMPLE_IMG.get_rect(center=(WIDTH * 0.7, img_y_pos))
    window.blit(GOOD_EXAMPLE_IMG, good_img_rect)
    window.blit(BAD_EXAMPLE_IMG, bad_img_rect)

    check_surf = load_emoji("✅", (64, 64))
    cross_surf = load_emoji("❌", (64, 64))
    window.blit(check_surf, check_surf.get_rect(center=(WIDTH * 0.3, img_y_pos + 220)))
    window.blit(cross_surf, cross_surf.get_rect(center=(WIDTH * 0.7, img_y_pos + 220)))

    if (pygame.time.get_ticks() // 500) % 2 == 0:
        continue_surf = MEDIUM_FONT.render("Press SPACE to Continue", True, GREEN)
        window.blit(continue_surf, continue_surf.get_rect(center=(WIDTH / 2, HEIGHT - 50)))

## MODE SELECT
def draw_mode_select_screen(window):
    """Displays the 1P vs 2P mode selection screen."""
    window.blit(PLAY_WITH_IMAGE, (0, 0))
    if (pygame.time.get_ticks() // 500) % 2 == 0:
        text_surface = BIG_FONT.render("Press 1 for 1P or 2 for 2P", True, WHITE)
        text_rect = text_surface.get_rect(center=(WIDTH / 2, 700))
        window.blit(text_surface, text_rect)

## RULES
def draw_rules_screen(window):
    """Displays the rules with a line-by-line typewriter effect."""
    global current_rule_line, current_rule_char, last_rule_char_update, fully_rendered_lines
    window.fill(BLACK)

    heading_surf = RULES_HEADING_FONT.render("RULES", True, WHITE)
    window.blit(heading_surf, heading_surf.get_rect(center=(WIDTH / 2, 150)))

    for surf, rect in fully_rendered_lines:
        window.blit(surf, rect)

    current_time = pygame.time.get_ticks()
    
    if current_rule_line < len(rules_to_display) and current_time - last_rule_char_update > RULE_CHAR_DELAY:
        last_rule_char_update = current_time
        line_text = rules_to_display[current_rule_line]
        if current_rule_char < len(line_text):
            current_rule_char += 1; typing_sound.play()
        else:
            completed_surf = RULES_TEXT_FONT.render(line_text, True, WHITE)
            fully_rendered_lines.append((completed_surf, completed_surf.get_rect(center=(WIDTH / 2, 300 + (current_rule_line * 60)))))
            current_rule_line += 1
            current_rule_char = 0
            typing_sound.stop()
            pygame.time.wait(RULE_LINE_DELAY) 
            typing_sound.play()
    if current_rule_line < len(rules_to_display):
        partial_surf = RULES_TEXT_FONT.render(rules_to_display[current_rule_line][:current_rule_char], True, WHITE)
        window.blit(partial_surf, partial_surf.get_rect(center=(WIDTH / 2, 300 + (current_rule_line * 60))))
    else:
        typing_sound.stop()
        if (pygame.time.get_ticks() // 400) % 2 == 0:
            continue_surf = MEDIUM_FONT.render("Press SPACE to Continue", True, GREEN)
            window.blit(continue_surf, continue_surf.get_rect(center=(WIDTH / 2, HEIGHT - 100)))

## 2P FACE CAPTURE
def pre_game_for_2p(window, frame_cv, faces):
    """Displays camera feed and prompts for face capture."""
    frame_rgb = cv2.cvtColor(frame_cv, cv2.COLOR_BGR2RGB)
    frame_pygame = pygame.surfarray.make_surface(np.rot90(frame_rgb))
    window.blit(pygame.transform.scale(frame_pygame, (WIDTH, HEIGHT)), (0, 0))

    prompt_text = f"Player {player_to_capture}, show your face"
    player_face_text = BIG_FONT.render(prompt_text, True, WHITE)
    text_rect = player_face_text.get_rect(center=(WIDTH / 2, 100))
    pygame.draw.rect(window, BLACK, text_rect.inflate(20, 20))
    window.blit(player_face_text, text_rect)

    feedback_text, color = ("Looks good! Press SPACE to capture", GREEN) if len(faces) == 1 else ("Please show only ONE face!", RED)
    face_feedback_surf = MEDIUM_FONT.render(feedback_text, True, color)
    feedback_rect = face_feedback_surf.get_rect(center=(WIDTH / 2, HEIGHT - 100))
    pygame.draw.rect(window, BLACK, feedback_rect.inflate(20, 20))
    window.blit(face_feedback_surf, feedback_rect)

## VERSUS
def draw_versus_screen(window):
    """Displays the VS screen with captured player faces."""
    window.blit(VS_IMAGE, (0, 0))
    face_size = (300, 300)
    for i, p_img in enumerate([player1_face_img, player2_face_img]):
        if p_img is not None:
            p_face_rgb = cv2.cvtColor(p_img, cv2.COLOR_BGR2RGB)
            p_face_surf = pygame.transform.scale(pygame.surfarray.make_surface(np.rot90(p_face_rgb)), face_size)
            p_face_surf = pygame.transform.flip(p_face_surf, True, False)
            p_rect = p_face_surf.get_rect(center=(WIDTH * (0.25 + i * 0.5), HEIGHT / 2))
            pygame.draw.rect(window, WHITE, p_rect.inflate(10, 10), 5)
            window.blit(p_face_surf, p_rect)

## 1P GAME
def game_mode_1p(window, frame_cv):
    """Handles all logic and drawing for the 1-Player game mode."""
    global round_state_1p, results_start_time, player_score_1p, ai_score_1p
    global prev_player_score_1p, prev_ai_score_1p, player_score_surf_1p, ai_score_surf_1p

    if player_score_1p != prev_player_score_1p:
        player_score_surf_1p = MEDIUM_FONT.render(f"PLAYER: {player_score_1p}", True, WHITE)
        prev_player_score_1p = player_score_1p
    if ai_score_1p != prev_ai_score_1p:
        ai_score_surf_1p = MEDIUM_FONT.render(f"AI: {ai_score_1p}", True, WHITE)
        prev_ai_score_1p = ai_score_1p

    if round_state_1p == 'PLAYING':
        results = model(cv2.GaussianBlur(frame_cv, (3, 3), 0), verbose=False)
        annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
        frame_pygame = pygame.transform.scale(pygame.surfarray.make_surface(np.rot90(annotated_frame)), (WIDTH, HEIGHT))
        window.blit(frame_pygame, (0, 0))
        if (pygame.time.get_ticks() // 400) % 2 == 0:
            prompt_surf = MEDIUM_FONT.render("Show your hand and press SPACE to lock in!", True, GREEN)
            window.blit(prompt_surf, prompt_surf.get_rect(center=(WIDTH / 2, HEIGHT - 50)))
    elif round_state_1p == 'SHOW_RESULTS':
        window.fill(BLACK)
        winner_text_raw = determine_winner(player_choice_1p, ai_choice_1p).replace("PLAYER 1", "PLAYER").replace("PLAYER 2", "AI")
        color = GREEN if "PLAYER" in winner_text_raw else RED if "AI" in winner_text_raw else WHITE
        texts_to_draw = [
            (f"You chose: {player_choice_1p.upper()}", WHITE),
            (f"AI chose: {ai_choice_1p.upper()}", WHITE),
            (winner_text_raw, color)
        ]
        for i, (text, color) in enumerate(texts_to_draw):
            surf = BIG_FONT.render(text, True, color)
            window.blit(surf, surf.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 100 + i * 100)))
        if pygame.time.get_ticks() - results_start_time > RESULTS_SCREEN_DURATION_1P:
            round_state_1p = 'GAME_OVER' if player_score_1p >= WINNING_SCORE or ai_score_1p >= WINNING_SCORE else 'PLAYING'
    elif round_state_1p == 'GAME_OVER':
        draw_game_over_screen(window, "YOU WIN!" if player_score_1p >= WINNING_SCORE else "AI WINS!")

    if round_state_1p != 'GAME_OVER':
        window.blit(player_score_surf_1p, (50, 50))
        window.blit(ai_score_surf_1p, (WIDTH - ai_score_surf_1p.get_width() - 50, 50))

## 2P GAME
def game_mode_2p(window, frame_cv):
    """Handles all logic and drawing for the 2-Player game mode with automatic detection."""
    global round_state_2p, countdown_number, countdown_start_time, results_start_time, warning_start_time, detection_start_time
    global player1_score_2p, player2_score_2p, player1_choice_2p, player2_choice_2p, round_winner_2p
    global prev_p1_score_2p, prev_p2_score_2p, p1_score_surf_2p, p2_score_surf_2p

    if player1_score_2p != prev_p1_score_2p:
        p1_score_surf_2p = MEDIUM_FONT.render(f"PLAYER 1: {player1_score_2p}", True, WHITE)
        prev_p1_score_2p = player1_score_2p
    if player2_score_2p != prev_p2_score_2p:
        p2_score_surf_2p = MEDIUM_FONT.render(f"PLAYER 2: {player2_score_2p}", True, WHITE)
        prev_p2_score_2p = player2_score_2p
    
    results = model(cv2.GaussianBlur(frame_cv, (3, 3), 0), verbose=False)
    annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    frame_pygame = pygame.transform.scale(pygame.surfarray.make_surface(np.rot90(annotated_frame)), (WIDTH, HEIGHT))
    window.blit(frame_pygame, (0, 0))
    pygame.draw.line(window, WHITE, (WIDTH // 2, 0), (WIDTH // 2, HEIGHT), 5)

    if round_state_2p == 'COUNTDOWN':
        prompt_surf = MEDIUM_FONT.render("Get Ready! Show your hands!", True, GREEN)
        window.blit(prompt_surf, prompt_surf.get_rect(center=(WIDTH / 2, HEIGHT - 50)))
        if pygame.time.get_ticks() - countdown_start_time > 1000:
            countdown_number -= 1; countdown_start_time = pygame.time.get_ticks()
            if countdown_number == 0:
                round_state_2p = 'DETECTING'; detection_start_time = pygame.time.get_ticks()
        if countdown_number > 0:
            countdown_text = COUNTDOWN_FONT.render(str(countdown_number), True, RED)
            window.blit(countdown_text, countdown_text.get_rect(center=(WIDTH / 2, HEIGHT / 2)))
    elif round_state_2p == 'DETECTING':
        prompt_surf = MEDIUM_FONT.render("Detecting...", True, GREEN)
        window.blit(prompt_surf, prompt_surf.get_rect(center=(WIDTH / 2, HEIGHT - 50)))
        p1c, p2c = None, None
        for box in results[0].boxes:
            choice = results[0].names[int(box.cls[0])].lower()
            if box.xywh[0][0] < 640: p1c = choice
            else: p2c = choice
        if p1c and p2c:
            player1_choice_2p, player2_choice_2p = p1c, p2c
            round_winner_2p = determine_winner(p1c, p2c)
            if "PLAYER 1" in round_winner_2p: player1_score_2p += 1
            elif "PLAYER 2" in round_winner_2p: player2_score_2p += 1
            round_state_2p = 'SHOW_RESULTS'; results_start_time = pygame.time.get_ticks()
        elif pygame.time.get_ticks() - detection_start_time > DETECTION_WINDOW:
            round_state_2p = 'WARNING'; warning_start_time = pygame.time.get_ticks()
    elif round_state_2p == 'WARNING':
        warning_surf = MEDIUM_FONT.render("Both hands not detected! Try again.", True, RED)
        window.blit(warning_surf, warning_surf.get_rect(center=(WIDTH / 2, HEIGHT - 50)))
        if pygame.time.get_ticks() - warning_start_time > WARNING_DURATION:
            round_state_2p = 'COUNTDOWN'; countdown_number = 3; countdown_start_time = pygame.time.get_ticks()
    elif round_state_2p == 'SHOW_RESULTS':
        window.fill(BLACK)
        color = GREEN if "PLAYER 1" in round_winner_2p else RED if "PLAYER 2" in round_winner_2p else WHITE
        texts_to_draw = [(f"P1 chose: {player1_choice_2p.upper()}", WHITE), (f"P2 chose: {player2_choice_2p.upper()}", WHITE), (round_winner_2p, color)]
        for i, (text, color) in enumerate(texts_to_draw):
            surf = BIG_FONT.render(text, True, color)
            window.blit(surf, surf.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 100 + i * 100)))
        if pygame.time.get_ticks() - results_start_time > RESULTS_SCREEN_DURATION_2P:
            if player1_score_2p >= WINNING_SCORE or player2_score_2p >= WINNING_SCORE:
                round_state_2p = 'GAME_OVER'
            else:
                round_state_2p = 'COUNTDOWN'; countdown_number = 3; countdown_start_time = pygame.time.get_ticks()
    elif round_state_2p == 'GAME_OVER':
        draw_game_over_screen(window, "PLAYER 1 WINS!" if player1_score_2p >= WINNING_SCORE else "PLAYER 2 WINS!")

    if round_state_2p not in ['SHOW_RESULTS', 'GAME_OVER']:
        window.blit(p1_score_surf_2p, (50, 50))
        window.blit(p2_score_surf_2p, (WIDTH - p2_score_surf_2p.get_width() - 50, 50))

## GAME OVER
def draw_game_over_screen(window, winner_text):
    """Displays the final game over screen."""
    window.fill(BLACK)
    color = GREEN if "PLAYER 1" in winner_text or "YOU" in winner_text else RED
    winner_surf = BIG_FONT.render(winner_text, True, color)
    prompt_surf = MEDIUM_FONT.render("Press 'M' to return to Menu", True, WHITE)
    quit_surf = MEDIUM_FONT.render("Press 'Q' to Quit!", True, WHITE)
    window.blit(winner_surf, winner_surf.get_rect(center=(WIDTH / 2, HEIGHT / 2 - 50)))
    window.blit(prompt_surf, prompt_surf.get_rect(center=(WIDTH / 2, HEIGHT / 2 + 50)))
    window.blit(quit_surf, quit_surf.get_rect(center=(WIDTH / 2, HEIGHT /2 + 100)))

## CREDITS
def draw_credits_screen(window):
    """Displays the scrolling credits and returns to menu when finished."""
    global credits_y_pos, game_state
    window.fill(BLACK)
    credits_y_pos -= CREDITS_SCROLL_SPEED
    current_y = credits_y_pos
    for surf, rect in rendered_credits:
        rect.center = (WIDTH / 2, current_y)
        window.blit(surf, rect)
        current_y += rect.height + 5
    if current_y < 0:
        sys.exit()

## MAIN GAME LOOP
try:
    running = True
    clock = pygame.time.Clock()
    while running:
        clock.tick(FPS)
        ret, frame_cv = cap.read()
        if not ret: print("Failed to open camera!"); break
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q:
                    if game_state in ["GAME_1P", "GAME_2P", "GAME_OVER"]:
                        game_state = "CREDITS"
                        credits_y_pos = HEIGHT + 100
                        rendered_credits.clear()
                        for line in CREDITS_TEXT:
                            if line == "LOGO_PLACEHOLDER":
                                rendered_credits.append((HAYASAKA, HAYASAKA.get_rect()))
                            else:
                                font = BIG_FONT if line == "CREDITS" else CREDITS_FONT
                                rendered_credits.append((font.render(line, True, WHITE), font.render(line, True, WHITE).get_rect()))
                    else:
                        running = False
                
                # STATE TRANSITION
                if game_state == "START_SCREEN":
                    game_state = "INSTRUCTION_SCREEN"
                elif game_state == "INSTRUCTION_SCREEN":
                    if event.key == pygame.K_SPACE:
                        game_state = "MODE_SELECT"
                elif game_state == "MODE_SELECT":
                    if event.key == pygame.K_1:
                        game_state, rules_to_display = "RULES_1P", RULES_1P_TEXT; pygame.mixer.music.pause()
                        current_rule_line, current_rule_char, fully_rendered_lines = 0, 0, []
                    elif event.key == pygame.K_2:
                        game_state, rules_to_display = "RULES_2P", RULES_2P_TEXT; pygame.mixer.music.pause()
                        current_rule_line, current_rule_char, fully_rendered_lines = 0, 0, []
                elif game_state in ["RULES_1P", "RULES_2P"]:
                    if event.key == pygame.K_SPACE and current_rule_line >= len(rules_to_display):
                        pygame.mixer.music.unpause()
                        if game_state == "RULES_1P":
                            game_state = "GAME_1P"
                            player_score_1p, ai_score_1p, prev_player_score_1p, prev_ai_score_1p = 0, 0, -1, -1
                            round_state_1p = "PLAYING"
                        else:
                            game_state = "PRE_GAME_2P"
                elif game_state == "PRE_GAME_2P":
                    faces = FACE_CASCADE.detectMultiScale(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY), 1.3, 5)
                    if event.key == pygame.K_SPACE and len(faces) == 1:
                        x, y, w, h = faces[0]
                        if player_to_capture == 1:
                            player1_face_img, player_to_capture = frame_cv[y:y + h, x:x + w].copy(), 2
                        else:
                            player2_face_img = frame_cv[y:y + h, x:x + w].copy()
                            game_state = "VERSUS_SCREEN"; versus_screen_start_time = pygame.time.get_ticks(); vs_sound.play()
                elif game_state == "GAME_1P":
                    if round_state_1p == 'PLAYING' and event.key == pygame.K_SPACE:
                        results = model(frame_cv, verbose=True)[0]
                        if len(results.boxes) == 1:
                            player_choice_1p = results.names[int(results.boxes[0].cls[0])].lower()
                            ai_choice_1p = random.choice(choices)
                            round_winner_1p = determine_winner(player_choice_1p, ai_choice_1p)
                            if "PLAYER 1" in round_winner_1p: player_score_1p += 1
                            elif "PLAYER 2" in round_winner_1p: ai_score_1p += 1
                            round_state_1p = 'SHOW_RESULTS'; results_start_time = pygame.time.get_ticks()
                    elif round_state_1p == 'GAME_OVER' and event.key == pygame.K_m:
                        game_state = "MODE_SELECT"
                elif game_state == "GAME_2P":
                    if round_state_2p == 'GAME_OVER' and event.key == pygame.K_m:
                        game_state = "MODE_SELECT"

        window.fill(BLACK)
        if game_state == "START_SCREEN": draw_start_screen(window)
        elif game_state == "INSTRUCTION_SCREEN": draw_instruction_screen(window)
        elif game_state == "MODE_SELECT": draw_mode_select_screen(window)
        elif game_state in ["RULES_1P", "RULES_2P"]: draw_rules_screen(window)
        elif game_state == "GAME_1P": game_mode_1p(window, frame_cv)
        elif game_state == "PRE_GAME_2P":
            faces = FACE_CASCADE.detectMultiScale(cv2.cvtColor(frame_cv, cv2.COLOR_BGR2GRAY), 1.3, 5)
            for (x, y, w, h) in faces: cv2.rectangle(frame_cv, (x, y), (x + w, y + h), (0, 255, 0), 3)
            pre_game_for_2p(window, frame_cv, faces)
        elif game_state == "VERSUS_SCREEN":
            draw_versus_screen(window)
            if pygame.time.get_ticks() - versus_screen_start_time > VS_SCREEN_DURATION:
                game_state = "GAME_2P"
                player1_score_2p, player2_score_2p, prev_p1_score_2p, prev_p2_score_2p = 0, 0, -1, -1
                round_state_2p, countdown_number = "COUNTDOWN", 3
                countdown_start_time = pygame.time.get_ticks()
        elif game_state == "GAME_2P": game_mode_2p(window, frame_cv)
        elif game_state == "CREDITS": draw_credits_screen(window)
        pygame.display.update()

except Exception as e:
    print(f"An error occurred: {e}")

finally:
    print("Exiting...")
    pygame.quit()
    cap.release()
    cv2.destroyAllWindows()
    sys.exit()