import collections
import math
import random

from .config import PRIMES, SEED_MOD
from .popularity import is_popular_combination
from .vectors import (
    get_external_calendar_seed,
    get_external_financial_seed,
    get_external_weather_seed,
    get_moon_phase,
    get_tesla_resonance_seed,
)


def weighted_seed(raw_seed, weight):
    """Skaliert einen Roh-Seed mit seinem Lern-Gewicht (auf 14 Stellen begrenzt)."""
    return int(raw_seed * weight) % SEED_MOD


def combine_seeds(time_seed, receipt_seed, user_seed, raw_vector_seeds, weights, exclude=None):
    """Kombiniert alle Vektoren per XOR zum finalen Matrix-Seed.

    Einzige Quelle der Seed-Kombination — wird identisch für die Vorhersage
    und für die Leave-One-Out-Analyse im Lernen verwendet (`exclude` lässt
    einen Vektor weg).
    """
    final_seed = time_seed ^ (receipt_seed % SEED_MOD)

    for key, weight in weights.items():
        if key == exclude:
            continue
        if key == "User_Sync_W":
            raw = user_seed
        else:
            raw = raw_vector_seeds.get(key.replace('_W', ''), 0)
        final_seed ^= weighted_seed(raw, weight)

    return final_seed


def gather_live_vectors(target_date, vis):
    """Ruft alle externen Vektor-Quellen ab und liefert Roh-Seeds + Status."""
    vis.loading_animation("Stelle Tesla-Uplink her", duration=0.1)
    tesla_seed, tesla_status = get_tesla_resonance_seed(target_date)
    vis.loading_animation("Stelle Kalender-Uplink her", duration=0.1)
    calendar_seed, calendar_status = get_external_calendar_seed(target_date)
    vis.loading_animation("Stelle Finanz-Uplink her", duration=0.1)
    financial_seed, financial_status = get_external_financial_seed()
    vis.loading_animation("Stelle Atmosphären-Uplink her", duration=0.1)
    weather_seed, weather_status = get_external_weather_seed(target_date)

    raw_vector_seeds = {
        "Finanz": financial_seed,
        "Wetter": weather_seed,
        "Kalender": calendar_seed,
        "Tesla": tesla_seed,
    }
    statuses = {
        "tesla": tesla_status,
        "calendar": calendar_status,
        "financial": financial_status,
        "weather": weather_status,
    }
    return raw_vector_seeds, statuses


class PhysicsEngine:
    @staticmethod
    def check_reality_integrity(numbers, is_euro=False):
        if is_euro: return True
        s = sum(numbers)
        if s < 95 or s > 180: return False
        primes_count = sum(1 for n in numbers if n in PRIMES)
        if primes_count == 0 or primes_count == 5: return False
        return True


class LearningEngine:
    def __init__(self, weights, target_date, rng=None):
        self.target_date = target_date
        self.weights = weights
        # Lokale RNG-Instanz statt globalem Zufallszustand -> reproduzierbar
        self.rng = rng if rng is not None else random.Random()
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
            last_idx = last_seen_main.get(n)
            # Gap = Anzahl Ziehungen seit dem letzten Auftreten (0 = letzte Ziehung)
            gap = total_draws if last_idx is None else total_draws - 1 - last_idx
            if gap > 10:
                self.weights_main[n] += (gap * 0.1)

        # 2. LUNARE GRAVITATION (über den gesamten Zahlenraum, nicht nur bekannte Keys)
        moon_phase = get_moon_phase(self.target_date)
        gravity_factor = 2.0 * self.weights.get("Mond_W", 1.0)
        moon_bias = -math.cos(moon_phase * 2 * math.pi)

        for num in range(1, 51):
            relative_pos = (num - 25) / 25
            if moon_bias > 0 and relative_pos > 0:
                self.weights_main[num] += moon_bias * gravity_factor
            elif moon_bias < 0 and relative_pos < 0:
                self.weights_main[num] += abs(moon_bias) * gravity_factor

    def predict_physically_accurate(self, pool_range, count, is_euro=False, recent_mains=None):
        pool = list(pool_range)
        current_weights = self.weights_euro if is_euro else self.weights_main

        candidates = []
        for attempt in range(1000):
            weights = [current_weights[n] if current_weights[n] > 0 else 0.1 for n in pool]
            temp_weights = weights.copy()
            candidates = []

            for _ in range(count):
                if sum(temp_weights) == 0: temp_weights = [1]*len(temp_weights)
                pick = self.rng.choices(pool, weights=temp_weights, k=1)[0]
                candidates.append(pick)
                idx = pool.index(pick)
                temp_weights[idx] = 0

            candidates.sort()
            if not PhysicsEngine.check_reality_integrity(candidates, is_euro):
                continue
            if not is_euro:
                popular, _ = is_popular_combination(candidates, recent_mains)
                if popular:
                    continue
            return candidates, attempt + 1
        return candidates, 999


def generate_prediction(seed, weights, dates, mains, euros, target_date):
    """Deterministische Vorhersage-Pipeline: gleicher Input -> gleicher Tipp.

    Wird sowohl für die Live-Vorhersage als auch für die Leave-One-Out-Analyse
    im Lernen und für das Backtesting verwendet.
    """
    rng = random.Random(seed)
    engine = LearningEngine(weights, target_date, rng)
    engine.calculate_factors(dates, mains, euros)
    pred_main, attempts = engine.predict_physically_accurate(
        range(1, 51), 5, is_euro=False, recent_mains=mains
    )
    pred_euro, _ = engine.predict_physically_accurate(range(1, 13), 2, is_euro=True)
    return pred_main, pred_euro, attempts
