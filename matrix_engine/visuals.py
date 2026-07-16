import sys
import time

from .config import GREEN, MAGENTA, RESET

# Globaler Fast-Mode: überspringt alle Animationen (CLI-Flag --fast)
_FAST_MODE = False


def set_fast_mode(enabled):
    global _FAST_MODE
    _FAST_MODE = enabled


class MatrixVisualizer:
    @staticmethod
    def type_writer(text, speed=0.02, color=GREEN):
        if _FAST_MODE:
            print(f"{color}{text}{RESET}")
            return
        sys.stdout.write(color)
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(speed)
        sys.stdout.write(RESET + '\n')

    @staticmethod
    def loading_animation(text, duration=1.0):
        if _FAST_MODE:
            print(f"{GREEN}[✓] {text}{RESET}")
            return
        end_time = time.time() + duration
        chars = "|/-\\"
        idx = 0
        while time.time() < end_time:
            sys.stdout.write(f'\r{MAGENTA}{chars[idx % len(chars)]} {text}...{RESET}')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.05)
        sys.stdout.write(f'\r{GREEN}[✓] {text}{RESET}\n')
