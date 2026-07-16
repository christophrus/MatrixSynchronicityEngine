# --- KONFIGURATION ---

# ANSI-Farben
GREEN = '\033[92m'
WHITE = '\033[97m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m'
RESET = '\033[0m'

# Dateinamen
WEIGHTS_FILE = "adaptive_weights.json"
HISTORY_CACHE_FILE = "history_cache.csv"

# Externe Datenquellen
DATA_URL = "https://raw.githubusercontent.com/christophrus/MatrixEurojackpot/refs/heads/main/history.csv"
FINANCE_API_URL = "https://api.frankfurter.app/latest?from=USD&to=EUR"
CALENDAR_API_URL = "https://zenquotes.io/api/today"

# Primzahlen im Eurojackpot-Bereich (1-50)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}

LEARNING_RATE = 0.15
SEED_MOD = 10**14

# Deutsche Datumsnamen (locale-unabhängig)
WOCHENTAGE = ["Montag", "Dienstag", "Mittwoch", "Donnerstag", "Freitag", "Samstag", "Sonntag"]
MONATE = ["Januar", "Februar", "März", "April", "Mai", "Juni",
          "Juli", "August", "September", "Oktober", "November", "Dezember"]


def format_german_date(dt):
    """Formatiert ein Datum als 'Dienstag, 21. Juli 2026'."""
    return f"{WOCHENTAGE[dt.weekday()]}, {dt.day:02d}. {MONATE[dt.month - 1]} {dt.year}"
