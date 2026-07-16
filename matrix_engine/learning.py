import datetime

from .config import GREEN, RED, YELLOW, RESET, LEARNING_RATE
from .data import filter_draws_before
from .engine import combine_seeds, generate_prediction


def calculate_error_score(pred_main, pred_euro, actual_main, actual_euro):
    """Berechnet den Fehler-Score basierend auf Treffern (0.00 = Perfekt, 1.00 = Null Treffer)."""
    main_hits = len(set(pred_main).intersection(set(actual_main)))
    euro_hits = len(set(pred_euro).intersection(set(actual_euro)))
    total_hits = main_hits + euro_hits
    return (7 - total_hits) / 7


def _is_legacy_entry(prediction):
    """Alte Einträge ohne vollständige Seed-Daten können nicht ausgewertet werden."""
    return (not prediction.get('vector_seeds')
            or not prediction.get('time_seed')
            or not prediction.get('weights_used'))


def _prediction_target_datetime(prediction):
    """Rekonstruiert den Ziehungs-Zeitpunkt (Ziehungen sind immer 21:00)."""
    return datetime.datetime.strptime(prediction['date'], "%Y-%m-%d").replace(hour=21)


def _replay_prediction(prediction, target_date, dates, mains, euros, exclude=None):
    """Spielt die ECHTE Vorhersage-Pipeline nach — optional ohne einen Vektor.

    `dates/mains/euros` müssen bereits auf den Datenstand zum Vorhersagezeitpunkt
    gefiltert sein (nur Ziehungen vor dem Vorhersagedatum).

    Sonderfall Mond: der Mond-Vektor wirkt nicht über den Seed, sondern über die
    Gravitations-Gewichtung in calculate_factors — sein Leave-One-Out schaltet
    daher die Gravitation aus statt einen (leeren) Seed zu entfernen.
    """
    weights_used = prediction['weights_used']
    seed = combine_seeds(
        prediction['time_seed'],
        prediction['receipt_seed'],
        prediction['user_seed'],
        prediction['vector_seeds'],
        weights_used,
        exclude=exclude,
    )
    gen_weights = weights_used
    if exclude == "Mond_W":
        gen_weights = dict(weights_used)
        gen_weights["Mond_W"] = 0.0
    pred_main, pred_euro, _ = generate_prediction(
        seed, gen_weights, dates, mains, euros, target_date
    )
    return pred_main, pred_euro


def run_adaptive_learning_check(current_weights, momentum, history, all_dates, all_mains, all_euros):
    """Führt differentielle Anpassung mit Momentum durch (V2.0).

    Misst Vektor-Beiträge per Leave-One-Out gegen die ECHTE Pipeline und
    validiert vorher per Sanity-Check, dass die gespeicherte Vorhersage
    exakt rekonstruierbar ist.

    Rückgabe: (new_weights, momentum, weights_adjusted)
    """
    new_weights = current_weights.copy()
    momentum = dict(momentum)
    weights_adjusted = False

    if not all_dates:
        return new_weights, momentum, False

    historical_draws = {}
    for i, date_obj in enumerate(all_dates):
        historical_draws[date_obj.strftime("%Y-%m-%d")] = (all_mains[i], all_euros[i])

    for prediction in history:
        if prediction.get('is_evaluated', True):
            continue
        if prediction['date'] not in historical_draws:
            continue  # Ziehung noch nicht in der Historie -> später auswerten

        if _is_legacy_entry(prediction):
            prediction['is_evaluated'] = True
            prediction['legacy'] = True
            print(f"{YELLOW}[!] Eintrag {prediction['date']} übersprungen: "
                  f"unvollständige Seed-Daten (legacy).{RESET}")
            continue

        actual_main, actual_euro = historical_draws[prediction['date']]
        target_date = _prediction_target_datetime(prediction)
        f_dates, f_mains, f_euros = filter_draws_before(target_date, all_dates, all_mains, all_euros)

        # SANITY-CHECK: die rekonstruierte Vorhersage muss der gespeicherten entsprechen,
        # sonst wäre jede Leave-One-Out-Messung bezugslos.
        recon_main, recon_euro = _replay_prediction(prediction, target_date, f_dates, f_mains, f_euros)
        if recon_main != prediction['main'] or recon_euro != prediction['euro']:
            prediction['is_evaluated'] = True
            prediction['legacy'] = True
            print(f"{YELLOW}[!] Eintrag {prediction['date']} übersprungen: Vorhersage nicht "
                  f"rekonstruierbar (Pipeline geändert?). Kein Lernen aus diesem Eintrag.{RESET}")
            continue

        original_error = calculate_error_score(
            prediction['main'], prediction['euro'], actual_main, actual_euro
        )
        original_hits = 7 - round(original_error * 7)

        print(f"\n{GREEN}>>> LEARNING CHECK: {prediction['date']}{RESET}")
        print(f"    Original Error: {original_error:.4f} ({original_hits} Treffer)")
        print(f"    Predicted: {prediction['main']} + {prediction['euro']}")
        print(f"    Actual:    {actual_main} + {actual_euro}")

        for key in new_weights.keys():
            if key == "User_Sync_W" and not prediction.get('user_key_used'):
                print(f"  {YELLOW}Vektor {key}: ÜBERSPRUNGEN (nicht verwendet){RESET}")
                continue

            # Leave-One-Out gegen die echte Pipeline: Vorhersage ohne diesen Vektor
            reduced_main, reduced_euro = _replay_prediction(
                prediction, target_date, f_dates, f_mains, f_euros, exclude=key
            )
            reduced_error = calculate_error_score(reduced_main, reduced_euro, actual_main, actual_euro)

            # Positiv = Vektor hat geholfen, Negativ = Vektor hat geschadet
            contribution = reduced_error - original_error

            base_adjustment = contribution * LEARNING_RATE * 3.0
            momentum[key] = 0.7 * momentum.get(key, 0.0) + 0.3 * base_adjustment

            old_w = new_weights[key]
            new_w = max(0.05, min(3.0, old_w + momentum[key]))
            new_weights[key] = new_w
            weights_adjusted = True

            if new_w > old_w + 0.01:
                symbol, color = "↑", GREEN
            elif new_w < old_w - 0.01:
                symbol, color = "↓", RED
            else:
                symbol, color = "→", YELLOW

            print(f"  {color}{symbol} {key}: {old_w:.4f} -> {new_w:.4f}{RESET}")
            print(f"     Beitrag: {contribution:+.4f} | Momentum: {momentum[key]:+.4f}")

        prediction['is_evaluated'] = True
        prediction['actual_main'] = actual_main
        prediction['actual_euro'] = actual_euro

        print(f"\n{GREEN}{'='*60}")
        print("LERNEN ABGESCHLOSSEN - Gewichte divergiert!")
        print(f"{'='*60}{RESET}\n")

    return new_weights, momentum, weights_adjusted
