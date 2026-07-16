import json

from .config import GREEN, RESET, WEIGHTS_FILE

DEFAULT_WEIGHTS = {
    "Finanz_W": 1.0,
    "Wetter_W": 1.0,
    "Kalender_W": 1.0,
    "Tesla_W": 1.0,
    "Mond_W": 1.0,
    "User_Sync_W": 1.0,
}


def load_or_initialize_state():
    """Lädt Gewichte, Momentum und Historie (mit Migration alter Dateien).

    Rückgabe: (weights, momentum, history)
    """
    try:
        with open(WEIGHTS_FILE, 'r') as f:
            data = json.load(f)
        weights = data['LERN_KOEFFIZIENTEN']
        history = data['LERN_HISTORIE']
        # Migration: alte Dateien haben kein persistiertes Momentum
        momentum = data.get('MOMENTUM', {key: 0.0 for key in weights})
        for key in weights:
            momentum.setdefault(key, 0.0)
        return weights, momentum, history
    except (FileNotFoundError, json.JSONDecodeError, KeyError):
        weights = dict(DEFAULT_WEIGHTS)
        momentum = {key: 0.0 for key in weights}
        return weights, momentum, []


def save_state(weights, momentum, history):
    """Speichert Gewichte, Momentum und Historie."""
    data = {
        "LERN_KOEFFIZIENTEN": weights,
        "MOMENTUM": momentum,
        "LERN_HISTORIE": history,
    }
    with open(WEIGHTS_FILE, 'w') as f:
        json.dump(data, f, indent=4)
    print(f"\n{GREEN}[✓] LERN-KOEFFIZIENTEN gespeichert in {WEIGHTS_FILE}{RESET}")
