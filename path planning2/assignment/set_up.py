import pygame
from pathlib import Path

from assignment import constants as C
from assignment import tools

pygame.init()
pygame.mixer.init()
SCREEN = pygame.display.set_mode((C.SCREEN_W, C.SCREEN_H))

pygame.display.set_caption("eee")

SOURCE_DIR = Path(__file__).resolve().parent / "source"
GRAPHICS = tools.load_graphics(str(SOURCE_DIR / "image"))

music_dir = SOURCE_DIR / "music"
SOUND = tools.load_sound(str(music_dir)) if music_dir.exists() else {}
