import csv
import datetime
import io
import os
import urllib.request

from .config import DATA_URL, HISTORY_CACHE_FILE, GREEN, RED, YELLOW, RESET


def _parse_history_csv(data_content):
    """Parst den CSV-Inhalt in (mains, euros, dates)."""
    main_history = []
    euro_history = []
    date_history = []

    file_stream = io.StringIO(data_content)
    reader = csv.reader(file_stream)

    try:
        next(reader)
    except StopIteration:
        return [], [], []

    for row in reader:
        try:
            if len(row) < 8: continue
            date_obj = datetime.datetime.strptime(row[0], "%d/%m/%Y")
            main_nums = sorted([int(row[i]) for i in range(1, 6)])
            euro_nums = sorted([int(row[i]) for i in range(6, 8)])
            date_history.append(date_obj)
            main_history.append(main_nums)
            euro_history.append(euro_nums)
        except ValueError:
            continue

    return main_history, euro_history, date_history


def load_historical_data():
    """Lädt die Ziehungs-Historie: erst Web, bei Fehler lokaler Cache.

    Rückgabe: (mains, euros, dates, latest_actual_draw)
    """
    data_content = None

    try:
        with urllib.request.urlopen(DATA_URL, timeout=15) as response:
            data_content = response.read().decode('utf-8')
        # Erfolgreichen Download als Offline-Fallback cachen
        try:
            with open(HISTORY_CACHE_FILE, 'w', encoding='utf-8') as f:
                f.write(data_content)
        except OSError as e:
            print(f"{YELLOW}[!] Cache konnte nicht geschrieben werden: {e}{RESET}")
    except Exception as e:
        print(f"{RED}[NETZWERK FEHLER] Download fehlgeschlagen: {e}{RESET}")
        if os.path.exists(HISTORY_CACHE_FILE):
            cache_age = datetime.datetime.now() - datetime.datetime.fromtimestamp(
                os.path.getmtime(HISTORY_CACHE_FILE)
            )
            print(f"{YELLOW}[!] Nutze lokalen Cache ({HISTORY_CACHE_FILE}, "
                  f"{cache_age.days} Tage alt).{RESET}")
            with open(HISTORY_CACHE_FILE, 'r', encoding='utf-8') as f:
                data_content = f.read()
        else:
            print(f"{RED}[✗] Kein lokaler Cache vorhanden. "
                  f"Ohne Ziehungs-Historie ist keine sinnvolle Analyse möglich.{RESET}")

    if data_content is None:
        return [], [], [], (None, None, None)

    main_history, euro_history, date_history = _parse_history_csv(data_content)

    latest_actual_draw = (None, None, None)
    if date_history:
        latest_actual_draw = (date_history[-1], main_history[-1], euro_history[-1])
        print(f"{GREEN}[✓] Historie geladen: {len(date_history)} Ziehungen "
              f"(letzte: {date_history[-1].strftime('%d.%m.%Y')}).{RESET}")

    return main_history, euro_history, date_history, latest_actual_draw


def filter_draws_before(cutoff_date, dates, mains, euros):
    """Liefert nur Ziehungen VOR dem Stichtag (Datenstand zum Vorhersagezeitpunkt)."""
    f_dates, f_mains, f_euros = [], [], []
    for i, d in enumerate(dates):
        if d.date() < cutoff_date.date():
            f_dates.append(d)
            f_mains.append(mains[i])
            f_euros.append(euros[i])
    return f_dates, f_mains, f_euros
