"""Ehrliches Backtesting: misst die Trefferquote der Engine gegen die Zufallserwartung.

Erwartungswerte (hypergeometrisch, fix für jede Tipp-Strategie):
  E[Haupt-Treffer] = 5 * 5/50  = 0.5
  E[Euro-Treffer]  = 2 * 2/12  ≈ 0.333
"""

import collections
import hashlib
import math
import random

from .config import BLUE, GREEN, WHITE, YELLOW, RESET, SEED_MOD
from .engine import generate_prediction

MIN_TRAINING_DRAWS = 100


def _hypergeom_pmf(k, winners, pool, picks):
    """P(genau k Treffer) beim Tippen von `picks` aus `pool` mit `winners` Gewinnzahlen."""
    if k > picks or picks - k > pool - winners:
        return 0.0
    return (math.comb(winners, k) * math.comb(pool - winners, picks - k)
            / math.comb(pool, picks))


def _deterministic_seed(date_obj):
    """Reproduzierbarer Seed pro Ziehungsdatum (ohne interaktive Eingaben)."""
    digest = hashlib.sha256(date_obj.strftime("%Y-%m-%d").encode()).hexdigest()
    return int(digest, 16) % SEED_MOD


def run_backtest(n_draws, dates, mains, euros, weights):
    total = len(dates)
    start = max(MIN_TRAINING_DRAWS, total - n_draws)
    if start >= total:
        print(f"{YELLOW}[!] Zu wenig Historie für einen Backtest "
              f"(mindestens {MIN_TRAINING_DRAWS + 1} Ziehungen nötig).{RESET}")
        return

    engine_main = collections.Counter()
    engine_euro = collections.Counter()
    random_main = collections.Counter()
    random_euro = collections.Counter()
    rng = random.Random(42)

    tested = 0
    for i in range(start, total):
        target_date = dates[i].replace(hour=21, minute=0, second=0, microsecond=0)
        train_dates = dates[:i]
        train_mains = mains[:i]
        train_euros = euros[:i]

        seed = _deterministic_seed(dates[i])
        pred_main, pred_euro, _ = generate_prediction(
            seed, weights, train_dates, train_mains, train_euros, target_date
        )
        engine_main[len(set(pred_main) & set(mains[i]))] += 1
        engine_euro[len(set(pred_euro) & set(euros[i]))] += 1

        # Vergleichstipp: rein zufällig, ohne jede "Intelligenz"
        rand_main = rng.sample(range(1, 51), 5)
        rand_euro = rng.sample(range(1, 13), 2)
        random_main[len(set(rand_main) & set(mains[i]))] += 1
        random_euro[len(set(rand_euro) & set(euros[i]))] += 1

        tested += 1

    print("\n" + "=" * 68)
    print(f"{WHITE}>>> BACKTEST: {tested} Ziehungen "
          f"({dates[start].strftime('%d.%m.%Y')} – {dates[-1].strftime('%d.%m.%Y')}) <<<{RESET}")
    print("=" * 68)

    def _print_table(title, engine_counts, random_counts, winners, pool, picks):
        print(f"\n{GREEN}{title}{RESET}")
        print(f"  {'Treffer':>8} | {'Engine':>8} | {'Zufall':>8} | {'Erwartung':>10}")
        print(f"  {'-'*8}-+-{'-'*8}-+-{'-'*8}-+-{'-'*10}")
        for k in range(picks + 1):
            expected = _hypergeom_pmf(k, winners, pool, picks) * tested
            print(f"  {k:>8} | {engine_counts.get(k, 0):>8} | "
                  f"{random_counts.get(k, 0):>8} | {expected:>10.1f}")

        avg_engine = sum(k * c for k, c in engine_counts.items()) / tested
        avg_random = sum(k * c for k, c in random_counts.items()) / tested
        avg_expected = picks * winners / pool
        print(f"  {BLUE}Ø Treffer: Engine {avg_engine:.3f} | Zufall {avg_random:.3f} | "
              f"Erwartung {avg_expected:.3f}{RESET}")
        return avg_engine, avg_expected

    avg_main, exp_main = _print_table("HAUPTZAHLEN (5 aus 50):",
                                      engine_main, random_main, 5, 50, 5)
    avg_euro, exp_euro = _print_table("EUROZAHLEN (2 aus 12):",
                                      engine_euro, random_euro, 2, 12, 2)

    dev_main = (avg_main - exp_main) / exp_main * 100
    dev_euro = (avg_euro - exp_euro) / exp_euro * 100

    print("\n" + "-" * 68)
    print(f"{WHITE}FAZIT:{RESET}")
    print(f"  Abweichung von der Zufallserwartung: "
          f"Haupt {dev_main:+.1f}% | Euro {dev_euro:+.1f}%")
    print(f"{YELLOW}  Hinweis: Lotterie-Ziehungen sind unabhängig und gleichverteilt.")
    print(f"  Abweichungen in dieser Größenordnung sind reines Stichprobenrauschen —")
    print(f"  keine Tipp-Strategie kann die Erwartung von 0.5 + 0.33 Treffern schlagen.")
    print(f"  Der reale Vorteil dieses Programms liegt im Anti-Popularitäts-Filter:")
    print(f"  unpopuläre Kombinationen teilen den Gewinn im Erfolgsfall seltener.{RESET}")
    print("-" * 68)
