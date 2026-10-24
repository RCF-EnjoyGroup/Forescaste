"""Tests del expediente de añadas graduadas (track record).

El registro acumula graduaciones por corrida; el expediente las
deduplica por AÑADA (publicación), cada una con su graduación más
reciente — la base de la vista «¿el forecast está mejorando?».
"""

import pytest

from src.monitoring.accuracy_tracking import build_track_record


def _record(run_id, mode, ts, cutoff, vintage=None):
    return {
        "run_id": run_id, "mode": mode, "timestamp": ts,
        "cutoff": cutoff, "run_dir": f"outputs/runs/{run_id}",
        "metrics": {}, "vintage_tracking": vintage,
    }


def _vt(start, end, wmape, n_obs, published_run_id=None):
    v = {
        "n_obs": n_obs,
        "window": {"start": start, "end": end},
        "global": {"wmape": wmape, "mape": wmape * 100, "bias": 0.02,
                   "n_obs": n_obs},
        "by_property": [
            {"property": "FIESTA", "n_obs": n_obs, "wmape": wmape,
             "bias": 0.02, "tracking_signal": 6.1},
        ],
        "interval_coverage": {"global": {"coverage": 0.85, "n_obs": n_obs,
                                          "nominal": 0.8}},
    }
    if published_run_id:
        v["published_run_id"] = published_run_id
    return v


def test_deduplica_por_annada_y_toma_la_graduacion_mas_reciente():
    """Dos corridas gradúan la MISMA añada — gana la graduación más reciente."""
    records = [
        _record("futuro-A", "futuro", "2026-10-20T10:00:00+00:00", "2026-10-06"),
        _record("real-1", "real", "2026-10-21T10:00:00+00:00", "2026-07-08",
                vintage=_vt("2026-10-07", "2026-10-12", 0.12, 36,
                            published_run_id="futuro-A")),
        _record("real-2", "real", "2026-10-22T10:00:00+00:00", "2026-07-08",
                vintage=_vt("2026-10-07", "2026-10-19", 0.09, 78,
                            published_run_id="futuro-A")),
    ]
    track = build_track_record(records)
    assert len(track) == 1, "mismas añadas se deduplican"
    assert track[0]["wmape"] == pytest.approx(0.09), "gana la graduación más reciente"
    assert track[0]["n_obs"] == 78


def test_atribucion_por_corte_cuando_falta_published_run_id():
    """Graduaciones previas al campo: la añada se identifica por cutoff.

    La publicación que cubre window.start tiene corte window.start − 1d.
    """
    records = [
        _record("futuro-vieja", "futuro", "2026-10-19T22:00:00+00:00", "2026-10-06"),
        _record("real-x", "real", "2026-10-21T15:00:00+00:00", "2026-07-08",
                vintage=_vt("2026-10-07", "2026-10-19", 0.093, 78)),
        # sin published_run_id en el vintage → atribuye por cutoff 2026-10-06
    ]
    track = build_track_record(records)
    assert len(track) == 1
    assert track[0]["published_run_id"] == "futuro-vieja", "atribución por cutoff"


def test_orden_por_publicacion_y_annadas_distintas():
    records = [
        _record("futuro-1", "futuro", "2026-10-19T22:00:00+00:00", "2026-10-06"),
        _record("futuro-2", "futuro", "2026-10-23T17:45:00+00:00", "2026-10-19"),
        _record("r1", "real", "2026-10-21T15:00:00+00:00", "2026-07-08",
                vintage=_vt("2026-10-07", "2026-10-19", 0.093, 78,
                            published_run_id="futuro-1")),
        _record("r2", "real", "2026-10-23T18:00:00+00:00", "2026-07-24",
                vintage=_vt("2026-10-20", "2026-10-22", 0.044, 18,
                            published_run_id="futuro-2")),
    ]
    track = build_track_record(records)
    assert len(track) == 2
    assert track[0]["published_run_id"] == "futuro-1"
    assert track[1]["published_run_id"] == "futuro-2"
    assert track[0]["cutoff"] == "2026-10-06"
    assert track[1]["cutoff"] == "2026-10-19"


def test_sin_graduaciones_devuelve_vacio():
    records = [_record("futuro-1", "futuro", "t1", "2026-10-06")]
    assert build_track_record(records) == []
    assert build_track_record([]) == []


def test_expide_worst_ts_y_cobertura():
    records = [
        _record("futuro-1", "futuro", "2026-10-19T22:00:00+00:00", "2026-10-06"),
        _record("r1", "real", "2026-10-21T15:00:00+00:00", "2026-07-08",
                vintage=_vt("2026-10-07", "2026-10-19", 0.093, 78,
                            published_run_id="futuro-1")),
    ]
    track = build_track_record(records)
    e = track[0]
    assert e["tracking_signal_worst"]["property"] == "FIESTA"
    assert e["tracking_signal_worst"]["value"] == pytest.approx(6.1)
    assert e["band_coverage"] == pytest.approx(0.85)
