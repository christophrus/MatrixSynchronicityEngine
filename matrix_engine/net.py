"""Gemeinsame HTTP-Helfer für alle externen Datenquellen.

Einige APIs (z.B. Frankfurter hinter Cloudflare) blockieren den
Standard-User-Agent von Python ("Python-urllib/3.x") mit HTTP 403 —
daher senden alle Abrufe einen Browser-üblichen User-Agent.
"""

import json
import urllib.request

USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"


def http_get(url, timeout=5):
    """GET-Request mit Browser-User-Agent; liefert den Body als Text."""
    req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=timeout) as response:
        return response.read().decode("utf-8")


def http_get_json(url, timeout=5):
    """GET-Request mit Browser-User-Agent; liefert den Body als JSON."""
    return json.loads(http_get(url, timeout=timeout))
