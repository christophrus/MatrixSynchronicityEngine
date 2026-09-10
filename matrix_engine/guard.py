"""Matrix-Wächter: lässt Faktor- und Exploit-Scan bei jedem Start automatisch mitlaufen.

Damit der Programmstart schnell bleibt:
  - Kurzfassung mit reduziertem Rechenbudget (die Flags --factor-scan /
    --exploit-scan liefern weiterhin die Vollanalyse mit großem Budget).
  - Ergebnis wird in scan_state.json gecacht und nur neu berechnet, wenn seit
    dem letzten Scan eine neue Ziehung in der Historie aufgetaucht ist.

Schlägt ein Test unterhalb der Bonferroni-Schwelle an, warnt der Wächter laut
und empfiehlt die Vollanalyse — erst Replikation auf unabhängigen Daten würde
aus einem Verdacht einen echten Befund machen.
"""

import json
import os

from .config import BLUE, GREEN, YELLOW, RESET, SCAN_STATE_FILE
from .factors import ALPHA, run_factor_scan
from .exploit import run_exploit_scan

GUARD_FACTOR_PERMS = 500
GUARD_EXPLOIT_SIMS = 200


def _load_state():
    if not os.path.exists(SCAN_STATE_FILE):
        return None
    try:
        with open(SCAN_STATE_FILE, "r", encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def _save_state(state):
    try:
        with open(SCAN_STATE_FILE, "w", encoding="utf-8") as f:
            json.dump(state, f, indent=2, ensure_ascii=False)
    except OSError as e:
        print(f"{YELLOW}[!] Wächter-Status konnte nicht gespeichert werden: {e}{RESET}")


def _summarize(pairs, n_tests):
    """Fasst (Name, p)-Paare zusammen: Signale unter Bonferroni-Schwelle + min p."""
    corrected_alpha = ALPHA / n_tests
    signals = [(name, p) for name, p in pairs if p < corrected_alpha]
    min_name, min_p = min(pairs, key=lambda item: item[1])
    return {
        "tests": n_tests,
        "signals": [{"name": name, "p": p} for name, p in signals],
        "min_p": min_p,
        "min_name": min_name,
    }


def _print_summary(state):
    factor = state["factor"]
    exploit = state["exploit"]
    stand = state["latest_draw"]

    def _line(label, summary, ok_text):
        if summary is None:
            print(f"{YELLOW}[WÄCHTER] {label}: übersprungen (zu wenig Historie).{RESET}")
            return
        if summary["signals"]:
            names = ", ".join(s["name"] for s in summary["signals"])
            print(f"{GREEN}[WÄCHTER] {label}: SIGNAL bei {names}! "
                  f"Bitte Vollanalyse fahren (--factor-scan / --exploit-scan) "
                  f"und auf neuen Daten replizieren.{RESET}")
        else:
            print(f"{BLUE}[WÄCHTER] {label}: {ok_text} "
                  f"(min p={summary['min_p']:.3f}, {summary['tests']} Tests).{RESET}")

    print(f"{BLUE}[WÄCHTER] Datenstand: Ziehung vom {stand} "
          f"({'neu gescannt' if state.get('fresh') else 'Cache, keine neue Ziehung'}).{RESET}")
    _line("Faktor-Scan ", factor, "kein Faktor-Einfluss messbar")
    _line("Exploit-Scan", exploit, "alle RNG-Angriffe abgewehrt")


def run_guard_checks(dates, mains, euros):
    """Kompakter Auto-Scan bei jedem Start; rechnet nur bei neuen Ziehungen neu."""
    if not dates:
        return

    latest_draw = dates[-1].strftime("%Y-%m-%d")
    state = _load_state()

    if state and state.get("latest_draw") == latest_draw:
        state["fresh"] = False
        _print_summary(state)
        return

    print(f"{BLUE}[WÄCHTER] Neue Ziehung erkannt — fahre Kurz-Scans "
          f"({GUARD_FACTOR_PERMS} Perm. / {GUARD_EXPLOIT_SIMS} Sim.)...{RESET}")

    factor_results = run_factor_scan(dates, mains, euros,
                                     n_perms=GUARD_FACTOR_PERMS, quiet=True)
    exploit_results = run_exploit_scan(dates, mains, euros,
                                       n_sims=GUARD_EXPLOIT_SIMS, quiet=True)

    state = {
        "latest_draw": latest_draw,
        "factor": _summarize(
            [(f"{fac}/{test}", p) for fac, test, _, _, p in factor_results],
            len(factor_results),
        ) if factor_results else None,
        "exploit": _summarize(exploit_results, len(exploit_results))
        if exploit_results else None,
        "fresh": True,
    }
    _save_state({k: v for k, v in state.items() if k != "fresh"})
    _print_summary(state)
