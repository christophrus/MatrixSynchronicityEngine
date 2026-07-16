"""Anti-Popularitäts-Filter.

Die Gewinnwahrscheinlichkeit ist für jede Kombination identisch — aber die
Gewinnhöhe nicht: populäre Kombinationen werden von vielen Spielern getippt
und der Gewinn wird geteilt. Dieser Filter verwirft Tipps, die statistisch
überdurchschnittlich oft gespielt werden.
"""

RECENT_DRAWS_WINDOW = 50


def _is_arithmetic_sequence(numbers):
    """Erkennt perfekte arithmetische Folgen wie 5-10-15-20-25 oder 1-2-3-4-5."""
    diffs = {numbers[i + 1] - numbers[i] for i in range(len(numbers) - 1)}
    return len(diffs) == 1


def is_popular_combination(numbers, recent_mains=None):
    """Prüft, ob eine (sortierte) Hauptzahlen-Kombination als 'populär' gilt.

    Rückgabe: (True, Grund) wenn populär, sonst (False, None).
    """
    # Geburtstags-Muster: alle Zahlen im Tag-Bereich 1-31
    if all(n <= 31 for n in numbers):
        return True, "Alle Zahlen ≤ 31 (Geburtstags-Muster)"

    # Tag+Monat-Geburtstage: zu viele Zahlen im Bereich 1-12
    if sum(1 for n in numbers if n <= 12) > 3:
        return True, "Mehr als 3 Zahlen ≤ 12 (Tag/Monat-Muster)"

    # Arithmetische Folgen / Straßen (5-10-15-20-25, 1-2-3-4-5, ...)
    if _is_arithmetic_sequence(numbers):
        return True, "Arithmetische Folge (Tippschein-Muster)"

    # Exakte Wiederholung einer kürzlichen Ziehung (wird oft nachgespielt)
    if recent_mains:
        recent = recent_mains[-RECENT_DRAWS_WINDOW:]
        if list(numbers) in [list(draw) for draw in recent]:
            return True, f"Identisch mit einer der letzten {RECENT_DRAWS_WINDOW} Ziehungen"

    return False, None
