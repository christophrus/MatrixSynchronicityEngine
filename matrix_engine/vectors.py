import datetime
import hashlib
import json
import random
import sys
import time
import urllib.request

from .config import BLUE, GREEN, RESET, YELLOW, CALENDAR_API_URL, FINANCE_API_URL


def get_moon_phase(date):
    """Berechnet die Mondphase."""
    diff = date - datetime.datetime(2001, 1, 1)
    days = diff.days + (diff.seconds / 86400)
    lunations = 0.20439731 + (days * 0.03386319269)
    return lunations % 1.0


def get_moon_description(phase):
    """Gibt eine beschreibende Bezeichnung für die Mondphase zurück."""
    if phase < 0.05 or phase > 0.95: return "Neumond (Low Gravity)"
    if 0.45 < phase < 0.55: return "Vollmond (High Gravity)"
    if phase < 0.5: return "Zunehmender Mond"
    return "Abnehmender Mond"


def get_next_jackpot_date(now=None):
    """Nächster Ziehungstermin (Dienstag/Freitag, 21:00)."""
    if now is None:
        now = datetime.datetime.now()
    candidate = now
    while True:
        if candidate.weekday() in [1, 4]:
            draw_time = candidate.replace(hour=21, minute=0, second=0, microsecond=0)
            if draw_time > now:
                return draw_time
        candidate += datetime.timedelta(days=1)
        candidate = candidate.replace(hour=7, minute=0)


def get_user_synchronicity_key():
    print(f"\n{YELLOW}[OPTIONAL] SYNCHRONIZITÄTS-SCHLÜSSEL EINGEBEN")
    print("Gib ein Wort/Zahl ein oder drücke ENTER für reine Physik.")
    try:
        user_input = input(f"{GREEN}>>> {RESET}")
    except KeyboardInterrupt:
        sys.exit()

    if not user_input.strip():
        return 0, None

    hash_object = hashlib.sha256(user_input.encode())
    seed_int = int(hash_object.hexdigest(), 16)
    return seed_int % 10**15, user_input.strip()


def get_external_receipt_seed():
    """Fordert die Losnummer ab und generiert einen Seed."""
    print(f"\n{BLUE}[PFLICHT] LOSNUMMER/QUITTUNGSNUMMER EINGEBEN")
    print("Diese Nummer dient als permanenter, physikalischer Zufallsvektor.")
    try:
        user_input = input(f"{GREEN}>>> {RESET}")
    except KeyboardInterrupt:
        sys.exit()

    hash_input = user_input.strip() if user_input.strip() else "0"

    hash_object = hashlib.sha256(hash_input.encode())
    seed_int = int(hash_object.hexdigest(), 16)

    status = f"Losnummer-Vektor erkannt (Input: {hash_input})"
    return seed_int % 10**15, status, hash_input


def get_external_financial_seed():
    try:
        with urllib.request.urlopen(FINANCE_API_URL, timeout=5) as response:
            data = json.loads(response.read().decode())
        rate = data.get('rates', {}).get('EUR')
        if rate is None: raise ValueError("API returned no EUR rate.")
        rate_int = int(rate * 10**7)
        status = f"Finanz-Vektor erkannt (Rate={rate:.5f})"
        return rate_int, status
    except Exception as e:
        status = f"Finanz-Vektor fehlgeschlagen. Nutze Fallback. (Error: {e.__class__.__name__})"
        now = datetime.datetime.now()
        seconds_since_midnight = (now - now.replace(hour=0, minute=0, second=0, microsecond=0)).total_seconds()
        return int(seconds_since_midnight * 1000) % 10000000, status


def get_external_weather_seed(target_date):
    date_str = target_date.strftime("%Y-%m-%d")
    API_URL = (
        "https://api.open-meteo.com/v1/forecast?"
        "latitude=60.1695&longitude=24.9355&"
        "daily=temperature_2m_max&temperature_unit=celsius&"
        "timezone=Europe%2FLondon&"
        f"start_date={date_str}&end_date={date_str}"
    )
    try:
        with urllib.request.urlopen(API_URL, timeout=5) as response:
            data = json.loads(response.read().decode())
        daily_data = data.get('daily')
        if not daily_data or not daily_data.get('temperature_2m_max'): raise ValueError("Forecast data not available.")
        temp = daily_data['temperature_2m_max'][0]
        temp_int = int(temp * 100) % 10000
        status = f"Atmosphären-Vektor erkannt (Forecast Max Temp={temp}°C für {date_str})"
        return temp_int, status
    except Exception as e:
        status = f"Atmosphären-Vektor fehlgeschlagen. Nutze Fallback. (Error: {e.__class__.__name__})"
        # Eigene RNG-Instanz, damit der globale Zufallszustand unberührt bleibt
        fallback_rng = random.Random(int(time.time()))
        return fallback_rng.randint(100, 3000), status


def get_external_calendar_seed(target_date):
    try:
        with urllib.request.urlopen(CALENDAR_API_URL, timeout=5) as response:
            data = json.loads(response.read().decode())
        quote_text = data[0].get('q')
        if quote_text is None: raise ValueError("API returned no quote text.")
        hash_object = hashlib.sha256(quote_text.encode())
        seed_int = int(hash_object.hexdigest(), 16)
        final_seed_int = seed_int % 10**12
        status = f"Kalender-Vektor erkannt (Zitat: {quote_text[:50]}...)"
        return final_seed_int, status
    except Exception as e:
        status = f"Kalender-Vektor fehlgeschlagen. Nutze Fallback. (Error: {e.__class__.__name__})"
        fallback_string = str(target_date.date()) + "FALLBACK"
        hash_object = hashlib.sha256(fallback_string.encode())
        return int(hash_object.hexdigest(), 16) % 10**12, status


def get_tesla_resonance_seed(target_date):
    TESLA_CONSTANT = 369
    day_of_year = target_date.timetuple().tm_yday
    resonance_seed = (day_of_year * TESLA_CONSTANT) % 10000000
    status = f"Tesla-Resonanzvektor erkannt (DOY={day_of_year} * 369)"
    return resonance_seed, status
