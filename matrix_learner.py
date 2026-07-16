import argparse
import os
import sys
import time

# Windows-Konsolen laufen oft mit cp1252 und können ✓/↑/↓ nicht darstellen
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from matrix_engine.config import (
    BLUE, GREEN, MAGENTA, RED, WHITE, YELLOW, RESET,
    PRIMES, SEED_MOD, format_german_date,
)
from matrix_engine.visuals import MatrixVisualizer, set_fast_mode
from matrix_engine.persistence import load_or_initialize_state, save_state
from matrix_engine.data import load_historical_data
from matrix_engine.learning import run_adaptive_learning_check
from matrix_engine.engine import combine_seeds, gather_live_vectors, generate_prediction, weighted_seed
from matrix_engine.backtest import run_backtest
from matrix_engine.vectors import (
    get_moon_phase, get_moon_description, get_next_jackpot_date,
    get_user_synchronicity_key, get_external_receipt_seed,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Matrix Adaptive Learner V2.0 — Eurojackpot-Engine mit ehrlichem Backtesting."
    )
    parser.add_argument("--fast", action="store_true",
                        help="Animationen und Typewriter-Effekte überspringen.")
    parser.add_argument("--no-input", action="store_true",
                        help="Nicht-interaktiv: Losnummer/User-Key werden nicht abgefragt.")
    parser.add_argument("--backtest", nargs="?", const=100, type=int, metavar="N",
                        help="Backtest über die letzten N Ziehungen (Default 100) statt Vorhersage.")
    return parser.parse_args()


def print_banner():
    print(f"{GREEN}")
    print(r"""
     __  __       _        _
    |  \/  | __ _| |_ _ __(_)_  __
    | |\/| |/ _` | __| '__| \ \/ /
    | |  | | (_| | |_| |  | |>  <
    |_|  | |\__,_|\__|_|  |_/_/\_\
      ADAPTIVE LEARNER V2.0 (Deterministische Pipeline)
    """)
    print(f"{RESET}")


def print_recent_draws(dates, mains, euros):
    print("\n" + "=" * 60)
    print(f"{YELLOW}>>> SYSTEM LOG: LETZTE 5 ZIEHUNGEN (Historie) <<<{RESET}")
    print("-" * 60)
    for i in range(len(dates) - 1, max(len(dates) - 6, -1), -1):
        main_str = ', '.join(f'{n:02d}' for n in mains[i])
        euro_str = ', '.join(f'{n:02d}' for n in euros[i])
        print(f"[{dates[i].strftime('%d.%m.%Y')}] Haupt: {main_str} | Euro: {euro_str}")
    print("=" * 60)


def main():
    args = parse_args()
    set_fast_mode(args.fast)

    vis = MatrixVisualizer()
    os.system('cls' if os.name == 'nt' else 'clear')
    print_banner()

    current_weights, momentum, history = load_or_initialize_state()
    print(f"{GREEN}[✓] LERN-DATEN geladen: {len(history)} Einträge, "
          f"{len(current_weights)} Vektoren.{RESET}")

    vis.loading_animation("Downloade aktuelle Ziehungs-Historie")
    mains, euros, dates, latest_actual_draw = load_historical_data()

    # --- BACKTEST-MODUS ---
    if args.backtest is not None:
        if not dates:
            print(f"{RED}[✗] Backtest ohne Ziehungs-Historie nicht möglich.{RESET}")
            sys.exit(1)
        run_backtest(args.backtest, dates, mains, euros, current_weights)
        return

    # --- ADAPTIVES LERNEN ---
    new_weights, momentum, weights_updated = run_adaptive_learning_check(
        current_weights, momentum, history, dates, mains, euros
    )
    if weights_updated:
        current_weights = new_weights
        save_state(current_weights, momentum, history)
        vis.type_writer(f"{MAGENTA}*** LERNEN ABGESCHLOSSEN. NEUE, DIVERGIERENDE GEWICHTE "
                        f"WERDEN VERWENDET ***{RESET}", speed=0.01)

    if dates:
        print_recent_draws(dates, mains, euros)

    # --- VORHERSAGE ---
    next_draw = get_next_jackpot_date()
    time_seed = int(next_draw.timestamp())

    if args.no_input:
        receipt_seed_int, receipt_status, receipt_input = 0, "Losnummer-Vektor übersprungen (--no-input)", "N/A"
        user_seed_int, user_str = 0, None
    else:
        receipt_seed_int, receipt_status, receipt_input = get_external_receipt_seed()
        user_seed_int, user_str = get_user_synchronicity_key()

    vis.type_writer(f"\n{BLUE}--- LESE GEWICHTETE VEKTOREN (V2.0) ---{RESET}", speed=0.01)
    raw_vector_seeds, statuses = gather_live_vectors(next_draw, vis)

    final_seed = combine_seeds(
        time_seed, receipt_seed_int, user_seed_int, raw_vector_seeds, current_weights
    )

    vis.type_writer(receipt_status, color=YELLOW, speed=0.01)
    for key in ("tesla", "calendar", "financial", "weather"):
        status = statuses[key]
        vis.type_writer(status, color=YELLOW if "erkannt" in status else RED, speed=0.01)

    vis.type_writer("Initialisiere Quanten-Simulation...", speed=0.02)

    moon_phase = get_moon_phase(next_draw)
    moon_desc = get_moon_description(moon_phase)

    W = current_weights
    print("\n" + "-" * 60)
    vis.type_writer(f"Zeit-Ziel:   {WHITE}{next_draw.strftime('%d.%m.%Y %H:%M')}{RESET}", color=WHITE, speed=0.02)
    vis.type_writer(f"Mond-Status: {MAGENTA}{moon_desc} ({moon_phase:.2f}){RESET}", color=MAGENTA, speed=0.02)
    vis.type_writer(f"Losnummer (Zuordnung): {YELLOW}{receipt_input}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"User-Key:    {YELLOW}{user_str if user_str else 'Keine (Standard-Physik)'}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Tesla-Vektor (W={W['Tesla_W']:.3f}): {YELLOW}{weighted_seed(raw_vector_seeds['Tesla'], W['Tesla_W'])}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Kalender-Vektor (W={W['Kalender_W']:.3f}): {YELLOW}{weighted_seed(raw_vector_seeds['Kalender'], W['Kalender_W'])}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Finanz-Vektor (W={W['Finanz_W']:.3f}): {YELLOW}{weighted_seed(raw_vector_seeds['Finanz'], W['Finanz_W'])}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Atmosphär-Vektor (W={W['Wetter_W']:.3f}): {YELLOW}{weighted_seed(raw_vector_seeds['Wetter'], W['Wetter_W'])}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Sync-Vektor (W={W.get('User_Sync_W', 0.0):.3f}): {YELLOW}{weighted_seed(user_seed_int, W.get('User_Sync_W', 0.0))}{RESET}", color=YELLOW, speed=0.02)
    vis.type_writer(f"Matrix-Seed: {BLUE}{final_seed}{RESET}", color=BLUE, speed=0.01)
    print("-" * 60 + "\n")
    if not args.fast:
        time.sleep(0.5)

    vis.loading_animation("Berechne Temporale Resonanz")
    vis.loading_animation("Scanne Entropie & 'kalte' Zahlen")
    vis.loading_animation(f"Simuliere Lunare Gravitation (W={W['Mond_W']:.3f})")
    vis.loading_animation("Filtere durch Realitäts-Glockenkurve & Popularitäts-Check")

    pred_main, pred_euro, attempts = generate_prediction(
        final_seed, current_weights, dates, mains, euros, next_draw
    )

    prediction_data = {
        "date": next_draw.strftime("%Y-%m-%d"),
        "main": pred_main,
        "euro": pred_euro,
        "seed": final_seed,
        "weights_used": dict(current_weights),
        "time_seed": time_seed,
        "user_seed": user_seed_int,
        "receipt_seed": receipt_seed_int,
        "vector_seeds": raw_vector_seeds,
        "user_key_used": user_str,
        "receipt_key_used": receipt_input,
        "is_evaluated": False,
    }
    history.append(prediction_data)
    save_state(current_weights, momentum, history)

    print("\n" + "=" * 60)
    print(f"{WHITE}>>> VORHERSAGE FÜR {format_german_date(next_draw)} <<<{RESET}")
    print(f"Kollisions-Iterationen (Reality- & Popularitäts-Check): {attempts}")
    print("=" * 60)

    s_sum = sum(pred_main)
    primes = sum(1 for n in pred_main if n in PRIMES)

    print(f"\n{GREEN}HAUPTZAHLEN:{RESET}")
    print(f"  [{', '.join(f'{n:02d}' for n in pred_main)}]")
    print(f"  {BLUE}(Summe: {s_sum} | Primzahlen: {primes} | Anti-Popularitäts-Check: bestanden){RESET}")

    print(f"\n{GREEN}EUROZAHLEN:{RESET}")
    print(f"  [{', '.join(f'{n:02d}' for n in pred_euro)}]")

    print("\n" + "-" * 60)
    vis.type_writer("Prediction saved. System lernt mit jedem Durchlauf.", speed=0.05)


if __name__ == "__main__":
    main()
