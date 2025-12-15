import random
import time
import sys
import os
import collections
import csv
import datetime
import hashlib
import urllib.request
import io
import math
import json 

# --- KONFIGURATION ---
GREEN = '\033[92m'
WHITE = '\033[97m'
RED = '\033[91m'
BLUE = '\033[94m'
YELLOW = '\033[93m'
MAGENTA = '\033[95m' 
RESET = '\033[0m'

# Dateinamen
WEIGHTS_FILE = "adaptive_weights.json"
DATA_URL = "https://raw.githubusercontent.com/christophrus/MatrixEurojackpot/refs/heads/main/history.csv"
FINANCE_API_URL = "https://api.frankfurter.app/latest?from=USD&to=EUR" 
CALENDAR_API_URL = "https://zenquotes.io/api/today" 

# Primzahlen im Eurojackpot-Bereich (1-50)
PRIMES = {2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47}
LEARNING_RATE = 0.15  # Erhöht für stärkere Divergenz
MOMENTUM = {}  # Globaler Momentum-Speicher

# --- DATEN-PERSISTENZ & LERN-LOGIK ---

def load_or_initialize_weights():
    """Lädt Koeffizienten oder initialisiert sie, falls die Datei fehlt."""
    try:
        with open(WEIGHTS_FILE, 'r') as f:
            data = json.load(f)
        return data['LERN_KOEFFIZIENTEN'], data['LERN_HISTORIE']
    except (FileNotFoundError, json.JSONDecodeError):
        initial_weights = {
            "Finanz_W": 1.0,
            "Wetter_W": 1.0,
            "Kalender_W": 1.0,
            "Tesla_W": 1.0,
            "Mond_W": 1.0,
            "User_Sync_W": 1.0
        }
        return initial_weights, []

def save_weights(weights, history):
    """Speichert die aktualisierten Koeffizienten und die Historie."""
    data = {
        "LERN_KOEFFIZIENTEN": weights,
        "LERN_HISTORIE": history
    }
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n{GREEN}[✓] LERN-KOEFFIZIENTEN gespeichert in {WEIGHTS_FILE}{RESET}")

def calculate_error_score(pred_main, pred_euro, actual_main, actual_euro):
    """Berechnet den Fehler-Score basierend auf Treffern (0.00 = Perfekt, 1.00 = Null Treffer)."""
    
    main_hits = len(set(pred_main).intersection(set(actual_main)))
    euro_hits = len(set(pred_euro).intersection(set(actual_euro)))
    
    total_hits = main_hits + euro_hits
    
    # Fehler = Abweichung von der perfekten Vorhersage (7 Treffer)
    return (7 - total_hits) / 7

def calculate_vector_contribution(prediction, vector_key, actual_main, actual_euro):
    """Misst den isolierten Beitrag eines Vektors durch direkten Vergleich."""
    
    seeds = prediction.get('vector_seeds', {})
    weights_used = prediction.get('weights_used', {})
    time_seed = prediction.get('time_seed', 0)
    user_seed = prediction.get('user_seed', 0)
    receipt_seed = prediction.get('receipt_seed', 0)
    
    # Erstelle Seed MIT diesem Vektor (Original)
    full_seed = time_seed ^ receipt_seed
    
    for key in weights_used.keys():
        vname = key.replace('_W', '')
        
        if key == "User_Sync_W":
            vseed = user_seed
        else:
            vseed = seeds.get(vname, 0)
        
        weighted = int(vseed * weights_used[key]) % 10**14
        full_seed ^= weighted
    
    # Erstelle Seed OHNE diesen Vektor
    reduced_seed = time_seed ^ receipt_seed
    
    for key in weights_used.keys():
        if key == vector_key:
            continue  # Überspringe diesen Vektor
        
        vname = key.replace('_W', '')
        
        if key == "User_Sync_W":
            vseed = user_seed
        else:
            vseed = seeds.get(vname, 0)
        
        weighted = int(vseed * weights_used[key]) % 10**14
        reduced_seed ^= weighted
    
    # Simuliere beide Vorhersagen
    random.seed(full_seed)
    full_main = sorted(random.sample(range(1, 51), 5))
    full_euro = sorted(random.sample(range(1, 13), 2))
    full_error = calculate_error_score(full_main, full_euro, actual_main, actual_euro)
    
    random.seed(reduced_seed)
    reduced_main = sorted(random.sample(range(1, 51), 5))
    reduced_euro = sorted(random.sample(range(1, 13), 2))
    reduced_error = calculate_error_score(reduced_main, reduced_euro, actual_main, actual_euro)
    
    # Positiv = Vektor hat geholfen, Negativ = Vektor hat geschadet
    return reduced_error - full_error

def run_adaptive_learning_check(current_weights, history, latest_actual_draw):
    """Führt differentielle Anpassung mit Momentum durch (V1.5)."""
    global MOMENTUM
    
    latest_actual_date, actual_main, actual_euro = latest_actual_draw
    new_weights = current_weights.copy()
    weights_adjusted = False
    
    # Initialisiere Momentum beim ersten Durchlauf
    if not MOMENTUM:
        MOMENTUM = {key: 0.0 for key in current_weights.keys()}
    
    if not latest_actual_date:
        return current_weights, False

    latest_actual_date_str = latest_actual_date.strftime("%Y-%m-%d")

    for prediction in history:
        if not prediction.get('is_evaluated', True):
            
            if latest_actual_date_str == prediction['date']:
                
                # Original-Fehler berechnen
                original_error = calculate_error_score(
                    prediction['main'], prediction['euro'], actual_main, actual_euro
                )
                
                original_hits = 7 - int(original_error * 7)
                print(f"\n{GREEN}>>> LEARNING CHECK: {prediction['date']}{RESET}")
                print(f"    Original Error: {original_error:.4f} ({original_hits} Treffer)")
                print(f"    Predicted: {prediction['main']} + {prediction['euro']}")
                print(f"    Actual:    {actual_main} + {actual_euro}")
                
                # Analysiere jeden Vektor einzeln
                for key in new_weights.keys():
                    
                    # User-Sync nur bewerten, wenn er verwendet wurde
                    if key == "User_Sync_W" and not prediction.get('user_key_used'):
                        print(f"  {YELLOW}Vektor {key}: ÜBERSPRUNGEN (nicht verwendet){RESET}")
                        continue
                    
                    # Berechne isolierten Beitrag
                    contribution = calculate_vector_contribution(
                        prediction, key, actual_main, actual_euro
                    )
                    
                    # Starke Anpassung für deutliche Signale
                    base_adjustment = contribution * LEARNING_RATE * 3.0
                    
                    # Momentum anwenden (70% alter Momentum + 30% neue Anpassung)
                    MOMENTUM[key] = 0.7 * MOMENTUM[key] + 0.3 * base_adjustment
                    
                    # Finale Anpassung mit Momentum
                    old_w = new_weights[key]
                    new_w = old_w + MOMENTUM[key]
                    
                    # Begrenze Gewichte auf sinnvollen Bereich
                    new_w = max(0.05, min(3.0, new_w))
                    
                    new_weights[key] = new_w
                    weights_adjusted = True
                    
                    # Status-Symbol basierend auf Änderung
                    if new_w > old_w + 0.01:
                        symbol = "↑"
                        color = GREEN
                    elif new_w < old_w - 0.01:
                        symbol = "↓"
                        color = RED
                    else:
                        symbol = "→"
                        color = YELLOW
                    
                    print(f"  {color}{symbol} {key}: {old_w:.4f} -> {new_w:.4f}{RESET}")
                    print(f"     Beitrag: {contribution:+.4f} | Momentum: {MOMENTUM[key]:+.4f}")
                
                # Historie aktualisieren
                prediction['is_evaluated'] = True
                prediction['actual_main'] = actual_main
                prediction['actual_euro'] = actual_euro
                
                print(f"\n{GREEN}{'='*60}")
                print(f"LERNEN ABGESCHLOSSEN - Gewichte divergiert!")
                print(f"{'='*60}{RESET}\n")

    return new_weights, weights_adjusted

# --- HILFSFUNKTIONEN ---

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

def get_next_jackpot_date():
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
        random.seed(int(time.time()))
        return random.randint(100, 3000), status

def get_external_calendar_seed(target_date):
    url = CALENDAR_API_URL
    try:
        with urllib.request.urlopen(url, timeout=5) as response:
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

def load_historical_data_from_web():
    main_history = []
    euro_history = []
    date_history = [] 
    latest_actual_draw = (None, None, None)

    try:
        with urllib.request.urlopen(DATA_URL) as response:
            data_content = response.read().decode('utf-8')
            
        file_stream = io.StringIO(data_content)
        reader = csv.reader(file_stream)
        
        try:
            next(reader)
        except StopIteration:
            return [], [], [], latest_actual_draw

        for row in reader:
            try:
                if len(row) < 8: continue
                date_obj = datetime.datetime.strptime(row[0], "%d/%m/%Y")
                date_history.append(date_obj)
                main_nums = sorted([int(row[i]) for i in range(1, 6)])
                euro_nums = sorted([int(row[i]) for i in range(6, 8)])
                main_history.append(main_nums)
                euro_history.append(euro_nums)
                
            except ValueError:
                continue
        
        if date_history:
            latest_actual_draw = (date_history[-1], main_history[-1], euro_history[-1])

    except Exception as e:
        print(f"{RED}[NETZWERK FEHLER] {e}{RESET}")
            
    return main_history, euro_history, date_history, latest_actual_draw

# --- LEARNING ENGINE ---

class LearningEngine:
    def __init__(self, weights, target_date):
        self.target_date = target_date
        self.weights = weights
        self.weights_main = collections.defaultdict(lambda: 1.0)
        self.weights_euro = collections.defaultdict(lambda: 1.0)

    def calculate_factors(self, dates, mains, euros):
        # 1. ZEIT-RESONANZ und ENTROPIE
        for i, hist_date in enumerate(dates):
            score = 0
            if hist_date.day == self.target_date.day: score += 4.0
            if hist_date.month == self.target_date.month: score += 2.0
            if hist_date.weekday() == self.target_date.weekday(): score += 1.0
            if score > 0:
                for n in mains[i]: self.weights_main[n] += score
                for n in euros[i]: self.weights_euro[n] += score
        
        last_seen_main = {}
        total_draws = len(dates)
        for i, draw in enumerate(mains):
            for num in draw:
                last_seen_main[num] = i
        for n in range(1, 51):
            gaps = total_draws - last_seen_main.get(n, 0)
            if gaps > 10: 
                self.weights_main[n] += (gaps * 0.1)

        # 2. LUNARE GRAVITATION
        moon_phase = get_moon_phase(self.target_date)
        gravity_factor = 2.0 * self.weights.get("Mond_W", 1.0) 
        
        for num in self.weights_main:
            relative_pos = (num - 25) / 25 
            moon_bias = -math.cos(moon_phase * 2 * math.pi) 
            if moon_bias > 0 and relative_pos > 0:
                self.weights_main[num] += moon_bias * gravity_factor
            elif moon_bias < 0 and relative_pos < 0:
                self.weights_main[num] += abs(moon_bias) * gravity_factor

    def get_weighted_seed(self, time_seed, user_seed_int, user_key_str, receipt_seed_int):
        
        vis = MatrixVisualizer()
        vis.loading_animation("Stelle Tesla-Uplink her", duration=0.1)
        tesla_seed, tesla_status = get_tesla_resonance_seed(self.target_date)
        vis.loading_animation("Stelle Kalender-Uplink her", duration=0.1)
        calendar_seed, calendar_status = get_external_calendar_seed(self.target_date)
        vis.loading_animation("Stelle Finanz-Uplink her", duration=0.1) 
        financial_seed, financial_status = get_external_financial_seed()
        vis.loading_animation("Stelle Atmosphären-Uplink her", duration=0.1)
        weather_seed, weather_status = get_external_weather_seed(self.target_date)
        
        W = self.weights
        
        raw_vector_seeds = {
            "Finanz": financial_seed,
            "Wetter": weather_seed,
            "Kalender": calendar_seed,
            "Tesla": tesla_seed,
        }
        
        seed_financial = int(financial_seed * W['Finanz_W']) % 10**14
        seed_weather   = int(weather_seed * W['Wetter_W']) % 10**14
        seed_calendar  = int(calendar_seed * W['Kalender_W']) % 10**14
        seed_tesla     = int(tesla_seed * W['Tesla_W']) % 10**14
        seed_user_sync = int(user_seed_int * W['User_Sync_W']) % 10**14
        seed_receipt = receipt_seed_int % 10**14

        final_seed = (
            time_seed ^ seed_user_sync ^ seed_financial ^ seed_weather ^ 
            seed_calendar ^ seed_tesla ^ seed_receipt 
        )
        
        return final_seed, {
            "calendar": (seed_calendar, calendar_status), 
            "financial": (seed_financial, financial_status), 
            "weather": (seed_weather, weather_status), 
            "tesla": (seed_tesla, tesla_status),
            "user_sync": (seed_user_sync, f"User-Sync-Vektor erkannt (Key: {user_key_str if user_key_str else 'N/A'})"),
            "receipt": (seed_receipt, f"Losnummer-Vektor erkannt (Seed)")
        }, raw_vector_seeds
    
    def predict_physically_accurate(self, pool_range, count, is_euro=False):
        pool = list(pool_range)
        current_weights = self.weights_euro if is_euro else self.weights_main
        
        for attempt in range(1000):
            weights = [current_weights[n] if current_weights[n] > 0 else 0.1 for n in pool]
            temp_weights = weights.copy()
            candidates = []
            
            for _ in range(count):
                if sum(temp_weights) == 0: temp_weights = [1]*len(temp_weights)
                pick = random.choices(pool, weights=temp_weights, k=1)[0]
                candidates.append(pick)
                idx = pool.index(pick)
                temp_weights[idx] = 0
            
            candidates.sort()
            if PhysicsEngine.check_reality_integrity(candidates, is_euro):
                return candidates, attempt + 1
        return candidates, 999 

class PhysicsEngine:
    @staticmethod
    def check_reality_integrity(numbers, is_euro=False):
        if is_euro: return True
        s = sum(numbers)
        if s < 95 or s > 180: return False
        primes_count = sum(1 for n in numbers if n in PRIMES)
        if primes_count == 0 or primes_count == 5: return False
        return True

class MatrixVisualizer:
    @staticmethod
    def type_writer(text, speed=0.02, color=GREEN):
        sys.stdout.write(color)
        for char in text:
            sys.stdout.write(char)
            sys.stdout.flush()
            time.sleep(speed)
        sys.stdout.write(RESET + '\n')

    @staticmethod
    def loading_animation(text, duration=1.0):
        end_time = time.time() + duration
        chars = "|/-\\" 
        idx = 0
        while time.time() < end_time:
            sys.stdout.write(f'\r{MAGENTA}{chars[idx % len(chars)]} {text}...{RESET}')
            sys.stdout.flush()
            idx += 1
            time.sleep(0.05) 
        sys.stdout.write(f'\r{GREEN}[✓] {text}{RESET}\n')

def main():
    vis = MatrixVisualizer()
    os.system('cls' if os.name == 'nt' else 'clear')
    print(f"{GREEN}")
    print(r"""
     __  __       _        _       
    |  \/  | __ _| |_ _ __(_)_  __ 
    | |\/| |/ _` | __| '__| \ \/ / 
    | |  | | (_| | |_| |  | |>  <  
    |_|  | |\__,_|\__|_|  |_/_/\_\ 
      ADAPTIVE LEARNER V1.5 (Vektor-Divergenz)
    """)
    print(f"{RESET}")
    
    current_weights, history = load_or_initialize_weights()
    print(f"{GREEN}[✓] LERN-DATEN geladen: {len(history)} Einträge, {len(current_weights)} Vektoren.{RESET}")

    vis.loading_animation("Downloade aktuelle Ziehungs-Historie")
    mains, euros, dates, latest_actual_draw = load_historical_data_from_web()
    
    new_weights, weights_updated = run_adaptive_learning_check(current_weights, history, latest_actual_draw)
    
    if weights_updated:
        current_weights = new_weights
        save_weights(current_weights, history)
        vis.type_writer(f"{MAGENTA}*** LERNEN ABGESCHLOSSEN. NEUE, DIVERGIERENDE GEWICHTE WERDEN VERWENDET ***{RESET}", speed=0.01)

    if dates:
        print("\n" + "=" * 60)
        print(f"{YELLOW}>>> SYSTEM LOG: LETZTE 5 ZIEHUNGEN (Historie) <<<{RESET}")
        print("-" * 60)
        
        count = 0
        for i in range(len(dates) - 1, -1, -1):
            if count >= 5: break
            
            main_str = ', '.join(f'{n:02d}' for n in mains[i])
            euro_str = ', '.join(f'{n:02d}' for n in euros[i])
            date_str = dates[i].strftime('%d.%m.%Y')
            
            print(f"[{date_str}] Haupt: {main_str} | Euro: {euro_str}")
            count += 1
        
        print("=" * 60)
    
    next_draw = get_next_jackpot_date()
    time_seed = int(next_draw.timestamp())
    
    receipt_seed_int, receipt_status, receipt_input = get_external_receipt_seed()
    user_seed_int, user_str = get_user_synchronicity_key()

    engine = LearningEngine(current_weights, next_draw)
    vis.type_writer(f"\n{BLUE}--- LESE GEWICHTETE VEKTOREN (V1.5) ---{RESET}", speed=0.01)
    
    final_seed, vector_details, raw_vector_seeds = engine.get_weighted_seed(time_seed, user_seed_int, user_str, receipt_seed_int) 
    
    vis.type_writer(receipt_status, color=YELLOW, speed=0.01) 
    vis.type_writer(vector_details['tesla'][1], color=YELLOW if "erkannt" in vector_details['tesla'][1] else RED, speed=0.01)
    vis.type_writer(vector_details['calendar'][1], color=YELLOW if "erkannt" in vector_details['calendar'][1] else RED, speed=0.01)
    vis.type_writer(vector_details['financial'][1], color=YELLOW if "erkannt" in vector_details['financial'][1] else RED, speed=0.01)
    vis.type_writer(vector_details['weather'][1], color=YELLOW if "erkannt" in vector_details['weather'][1] else RED, speed=0.01)
    
    vis.type_writer("Initialisiere Quanten-Simulation...", speed=0.02)
    
    random.seed(final_seed)
    
    moon_phase = get_moon_phase(next_draw)
    moon_desc = get_moon_description(moon_phase)

    print("\n" + "-" * 60)
    vis.type_writer(f"Zeit-Ziel:   {WHITE}{next_draw.strftime('%d.%m.%Y %H:%M')}{RESET}", color=WHITE, speed=0.02)
    vis.type_writer(f"Mond-Status: {MAGENTA}{moon_desc} ({moon_phase:.2f}){RESET}", color=MAGENTA, speed=0.02)
    
    vis.type_writer(f"Losnummer (Zuordnung): {YELLOW}{receipt_input}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"User-Key:    {YELLOW}{user_str if user_str else 'Keine (Standard-Physik)'}{RESET}", color=YELLOW, speed=0.02)
    
    vis.type_writer(f"Tesla-Vektor (W={current_weights['Tesla_W']:.3f}): {YELLOW}{vector_details['tesla'][0]}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Kalender-Vektor (W={current_weights['Kalender_W']:.3f}): {YELLOW}{vector_details['calendar'][0]}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Finanz-Vektor (W={current_weights['Finanz_W']:.3f}): {YELLOW}{vector_details['financial'][0]}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Atmosphär-Vektor (W={current_weights['Wetter_W']:.3f}): {YELLOW}{vector_details['weather'][0]}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Sync-Vektor (W={current_weights.get('User_Sync_W', 0.0):.3f}): {YELLOW}{vector_details['user_sync'][0]}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Matrix-Seed: {BLUE}{final_seed}{RESET}", color=BLUE, speed=0.01)
    print("-" * 60 + "\n")
    time.sleep(0.5)

    vis.loading_animation("Berechne Temporale Resonanz")
    vis.loading_animation("Scanne Entropie & 'kalte' Zahlen")
    vis.loading_animation(f"Simuliere Lunare Gravitation (W={current_weights['Mond_W']:.3f})")
    
    engine.calculate_factors(dates, mains, euros)

    vis.loading_animation("Filtere durch Realitäts-Glockenkurve")
    
    pred_main, attempts = engine.predict_physically_accurate(range(1, 51), 5, is_euro=False)
    pred_euro, _ = engine.predict_physically_accurate(range(1, 13), 2, is_euro=True)
    
    prediction_data = {
        "date": next_draw.strftime("%Y-%m-%d"),
        "main": pred_main,
        "euro": pred_euro,
        "seed": final_seed,
        "weights_used": current_weights,
        "time_seed": time_seed,
        "user_seed": user_seed_int,
        "receipt_seed": receipt_seed_int,
        "vector_seeds": raw_vector_seeds,
        "user_key_used": user_str, 
        "receipt_key_used": receipt_input, 
        "is_evaluated": False 
    }
    history.append(prediction_data)
    save_weights(current_weights, history)

    print("\n" + "=" * 60)
    print(f"{WHITE}>>> VORHERSAGE FÜR {next_draw.strftime('%A, %d. %B %Y')} <<<{RESET}")
    print(f"Kollisions-Iterationen (Reality Check): {attempts}")
    print("=" * 60)
    
    s_sum = sum(pred_main)
    primes = sum(1 for n in pred_main if n in PRIMES)
    
    print(f"\n{GREEN}HAUPTZAHLEN:{RESET}")
    print(f"  [{', '.join(f'{n:02d}' for n in pred_main)}]")
    print(f"  {BLUE}(Summe: {s_sum} | Primzahlen: {primes}){RESET}")
    
    print(f"\n{GREEN}EUROZAHLEN:{RESET}")
    print(f"  [{', '.join(f'{n:02d}' for n in pred_euro)}]")

    print("\n" + "-" * 60)
    vis.type_writer("Prediction saved. System lernt mit jedem Durchlauf.", speed=0.05)
    
if __name__ == "__main__":
    main()