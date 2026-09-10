"""Faktor-Scanner: misst, ob externe Faktoren nachweisbaren Einfluss auf die Ziehungen haben.

Prämisse: Wenn das System deterministisch ist und ein Faktor (Mond, Tesla-Resonanz,
Kalender, ...) das Ergebnis beeinflusst, MUSS dieser Einfluss als statistisches
Signal in der Ziehungshistorie messbar sein. Dieser Scanner testet das ehrlich:

  1. Jede Ziehung wird pro Faktor einem Bin zugeordnet (z.B. Mondphase -> Neumond).
  2. Pro Faktor werden Teststatistiken berechnet:
       - Summen-Resonanz: unterscheidet sich die mittlere Zahlensumme zwischen Bins?
       - Zonen-Verteilung: verschiebt der Faktor Zahlen zwischen den Zonen 1-10 ... 41-50?
       - Euro-Resonanz: unterscheidet sich die Eurozahlen-Summe zwischen Bins?
  3. Signifikanz per Permutationstest (Bin-Labels werden zufällig neu verteilt) —
     keine Verteilungsannahmen, reine Empirie.
  4. Bonferroni-Korrektur über alle Tests, damit bei vielen Tests nicht zufällig
     "Signale" entstehen (bei 18 Tests wären sonst ~1 Fehlalarm bei alpha=0.05 normal).

Findet der Scanner einen Faktor mit korrigiert-signifikantem p-Wert, der auch in
einer zweiten, späteren Datenhälfte bestehen bleibt, wäre das ein echter Kandidat,
um die Vektor-Gewichte datengetrieben zu setzen.
"""

import datetime
import random

from .config import BLUE, GREEN, RED, WHITE, YELLOW, RESET
from .vectors import get_moon_phase, get_moon_description

DEFAULT_PERMUTATIONS = 2000
ALPHA = 0.05

# Erste Ziehung mit Eurozahlen aus 1-12 (vorher 1-10) — Euro-Tests davor wären
# ein Zeitachsen-Artefakt, kein Faktor-Signal.
EURO_12_ERA_START = datetime.date(2022, 3, 29)


# --- FAKTOR-DEFINITIONEN (Datum -> Bin-Label) ---

def _bin_moon(date):
    return get_moon_description(get_moon_phase(date))


def _bin_weekday(date):
    return ["Mo", "Di", "Mi", "Do", "Fr", "Sa", "So"][date.weekday()]


def _bin_month(date):
    return f"M{date.month:02d}"


def _bin_day_zone(date):
    if date.day <= 10:
        return "Tag 1-10"
    if date.day <= 20:
        return "Tag 11-20"
    return "Tag 21-31"


def _bin_tesla(date):
    """Tesla-3-6-9-Resonanz: digitale Wurzel des Tags im Jahr.

    Hinweis: doy * 369 (wie im Seed-Vektor) hat IMMER die digitale Wurzel 9,
    weil 369 durch 9 teilbar ist — dieser 'Vektor' ist eine Konstante und kann
    prinzipiell keine Information tragen. Getestet wird deshalb die eigentliche
    3-6-9-Hypothese: Ziehungstage, deren digitale Wurzel 3, 6 oder 9 ist.
    """
    root = 1 + (date.timetuple().tm_yday - 1) % 9
    if root in (3, 6, 9):
        return f"Resonanz {root}"
    return "Keine Resonanz"


def _bin_season(date):
    m = date.month
    if m in (12, 1, 2):
        return "Winter"
    if m in (3, 4, 5):
        return "Frühling"
    if m in (6, 7, 8):
        return "Sommer"
    return "Herbst"


FACTORS = [
    ("Mondphase", _bin_moon),
    ("Wochentag", _bin_weekday),
    ("Monat", _bin_month),
    ("Tag im Monat", _bin_day_zone),
    ("Tesla-Resonanz (369)", _bin_tesla),
    ("Jahreszeit", _bin_season),
]


# --- TESTSTATISTIKEN ---

def _between_group_ss(values, labels):
    """Zwischen-Gruppen-Streuung: Summe n_b * (mean_b - grand_mean)^2.

    Groß, wenn sich die Bin-Mittelwerte stark unterscheiden -> Faktor-Einfluss.
    """
    grand_mean = sum(values) / len(values)
    sums, counts = {}, {}
    for v, lab in zip(values, labels):
        sums[lab] = sums.get(lab, 0.0) + v
        counts[lab] = counts.get(lab, 0) + 1
    return sum(
        counts[lab] * ((sums[lab] / counts[lab]) - grand_mean) ** 2
        for lab in counts
    )


def _zone_chi2(zone_counts, labels):
    """Chi-Quadrat der Kontingenztafel Bin x Zahlen-Zone (1-10, ..., 41-50)."""
    n_zones = len(zone_counts[0])
    table = {}
    for zc, lab in zip(zone_counts, labels):
        row = table.setdefault(lab, [0] * n_zones)
        for z in range(n_zones):
            row[z] += zc[z]

    row_totals = {lab: sum(row) for lab, row in table.items()}
    col_totals = [sum(table[lab][z] for lab in table) for z in range(n_zones)]
    grand = sum(row_totals.values())

    chi2 = 0.0
    for lab, row in table.items():
        for z in range(n_zones):
            expected = row_totals[lab] * col_totals[z] / grand
            if expected > 0:
                chi2 += (row[z] - expected) ** 2 / expected
    return chi2


def _permutation_test(values, labels, statistic, n_perms, rng):
    """p-Wert: Anteil der Label-Permutationen mit Statistik >= beobachtet."""
    observed = statistic(values, labels)
    shuffled = list(labels)
    at_least_as_extreme = 0
    for _ in range(n_perms):
        rng.shuffle(shuffled)
        if statistic(values, shuffled) >= observed:
            at_least_as_extreme += 1
    # +1-Korrektur: die beobachtete Anordnung zählt als eine Permutation mit
    return observed, (at_least_as_extreme + 1) / (n_perms + 1)


# --- SCANNER ---

def run_factor_scan(dates, mains, euros, n_perms=DEFAULT_PERMUTATIONS):
    if len(dates) < 100:
        print(f"{YELLOW}[!] Zu wenig Historie für einen Faktor-Scan "
              f"(mindestens 100 Ziehungen nötig).{RESET}")
        return

    draw_times = [d.replace(hour=21) for d in dates]
    main_sums = [sum(m) for m in mains]
    zone_counts = [
        [sum(1 for n in m if lo < n <= lo + 10) for lo in range(0, 50, 10)]
        for m in mains
    ]
    # Euro-Tests nur in der 2-aus-12-Ära, sonst misst man den Regelwechsel
    euro_idx = [i for i, d in enumerate(dates) if d.date() >= EURO_12_ERA_START]
    euro_sums = [sum(euros[i]) for i in euro_idx]

    rng = random.Random(369)  # fixer Seed -> reproduzierbarer Scan
    results = []

    print("\n" + "=" * 72)
    print(f"{WHITE}>>> FAKTOR-SCAN: {len(dates)} Ziehungen | "
          f"{n_perms} Permutationen pro Test <<<{RESET}")
    print("=" * 72)
    print("Hypothese: Ist das System deterministisch von diesen Faktoren beeinflusst,")
    print("muss sich das als signifikante Abweichung von der Zufallsverteilung zeigen.\n")

    for factor_name, bin_fn in FACTORS:
        labels = [bin_fn(t) for t in draw_times]
        n_bins = len(set(labels))
        if n_bins < 2:
            print(f"{YELLOW}[!] {factor_name}: nur ein Bin in der Historie — "
                  f"übersprungen.{RESET}")
            continue

        stat, p = _permutation_test(main_sums, labels, _between_group_ss, n_perms, rng)
        results.append((factor_name, "Summen-Resonanz (Haupt)", n_bins, stat, p))

        stat, p = _permutation_test(zone_counts, labels, _zone_chi2, n_perms, rng)
        results.append((factor_name, "Zonen-Verteilung (Haupt)", n_bins, stat, p))

        euro_labels = [labels[i] for i in euro_idx]
        if len(set(euro_labels)) >= 2 and len(euro_sums) >= 50:
            stat, p = _permutation_test(euro_sums, euro_labels,
                                        _between_group_ss, n_perms, rng)
            results.append((factor_name, "Summen-Resonanz (Euro)", n_bins, stat, p))

    corrected_alpha = ALPHA / len(results)

    print(f"  {'Faktor':<22} | {'Test':<26} | {'Bins':>4} | {'p-Wert':>8} | Befund")
    print(f"  {'-'*22}-+-{'-'*26}-+-{'-'*4}-+-{'-'*8}-+-{'-'*16}")
    signals = []
    for factor_name, test_name, n_bins, stat, p in results:
        if p < corrected_alpha:
            verdict, color = "SIGNAL!", GREEN
            signals.append((factor_name, test_name, p))
        elif p < ALPHA:
            verdict, color = "schwach (unkorr.)", YELLOW
        else:
            verdict, color = "Rauschen", RED
        print(f"  {factor_name:<22} | {test_name:<26} | {n_bins:>4} | "
              f"{p:>8.4f} | {color}{verdict}{RESET}")

    print("\n" + "-" * 72)
    print(f"{WHITE}FAZIT:{RESET}")
    print(f"  Tests: {len(results)} | Signifikanzschwelle (Bonferroni-korrigiert): "
          f"p < {corrected_alpha:.4f}")
    if signals:
        print(f"{GREEN}  {len(signals)} Faktor(en) mit korrigiert-signifikantem Signal:{RESET}")
        for factor_name, test_name, p in signals:
            print(f"{GREEN}    - {factor_name} / {test_name} (p={p:.4f}){RESET}")
        print(f"{YELLOW}  Nächster Schritt: Signal auf einer späteren, unabhängigen")
        print(f"  Datenhälfte replizieren, bevor Gewichte darauf gestützt werden.{RESET}")
    else:
        print(f"{YELLOW}  Kein Faktor zeigt ein korrigiert-signifikantes Signal.")
        print(f"  'schwach (unkorr.)' bedeutet: bei {len(results)} Tests sind einzelne")
        print(f"  p-Werte unter {ALPHA} durch reinen Zufall zu erwarten.")
        print(f"  Empirischer Stand damit: Diese Faktoren beeinflussen die Ziehung")
        print(f"  nicht messbar — die Gewichte sollten sie nicht überbewerten.{RESET}")
    print("-" * 72)

    return results
