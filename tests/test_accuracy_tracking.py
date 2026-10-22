"""Tests del tracking forecast-vs-real por vintage (spec ``monitoring``).

El forecast PUBLICADO por la corrida anterior se puntúa tal cual lo vio el
negocio cuando los reales se materializan — distinto del backtest interno
(mide el artefacto publicado, no una simulación). ``forecast_revision``
mide el delta entre corridas consecutivas sobre estancias aún futuras.
"""

import numpy as np
import pandas as pd
import pytest

from src.monitoring.accuracy_tracking import (
    forecast_revision,
    load_published_forecast,
    score_vintage,
)


def _published(n_future=30, n_past_materialized=20, seed=3):
    """Dataset estilo future_forecast_dataset.csv de una corrida anterior."""
    rng = np.random.default_rng(seed)
    cutoff = pd.Timestamp("2026-10-06")
    rows = []
    for i in range(n_future):
        stay = cutoff + pd.Timedelta(days=i + 1)
        rows.append({
            "property": "corin",
            "stay_date": stay,
            "forecast_date": cutoff,
            "lead_days": i + 1,
            "final_forecast_rn": 30.0 + rng.normal(0, 2),
            "p10_forecast": 20.0,
            "p50_forecast": 30.0,
            "p90_forecast": 40.0,
        })
    return pd.DataFrame(rows)


def _actuals(through: pd.Timestamp, seed=3):
    """Reales que cubren parte del horizonte publicado."""
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2026-10-01", through, freq="D")
    return pd.DataFrame({
        "property": "corin",
        "stay_date": dates,
        "room_nights": 30.0 + rng.normal(0, 2, len(dates)),
    })


# ------------------------------------------------------------ score_vintage


def test_score_vintage_solo_materializado():
    published = _published(n_future=30)
    actuals = _actuals(pd.Timestamp("2026-10-20"))  # 14 estancias con real
    report = score_vintage(published, actuals, through=pd.Timestamp("2026-10-20"))
    assert report is not None
    assert report["n_obs"] == 14, "solo estancias con real ya materializado"
    assert report["window"]["start"] == "2026-10-07"
    assert report["window"]["end"] == "2026-10-20"
    # métricas y desglose por banda presentes
    assert {"global", "by_lead_band", "interval_coverage"} <= set(report)
    assert report["global"]["wmape"] == pytest.approx(
        report["global"]["mae"] / 30.0, rel=0.35
    )


def test_score_vintage_cobertura_de_banda_por_vintage():
    """La banda publicada se mide contra los reales materializados."""
    published = _published()
    actuals = _actuals(pd.Timestamp("2026-10-20"))
    report = score_vintage(published, actuals, through=pd.Timestamp("2026-10-20"))
    cov = report["interval_coverage"]
    assert cov["global"]["n_obs"] == 14
    # banda 20-40 con reales ~N(30,2): cobertura alta
    assert cov["global"]["coverage"] >= 0.7


def test_score_vintage_sin_reales_devuelve_none():
    published = _published(n_future=30)
    actuals = _actuals(pd.Timestamp("2026-10-05"))  # antes del horizonte
    assert score_vintage(published, actuals, through=pd.Timestamp("2026-10-05")) is None


def test_score_vintage_sin_columna_punto_devuelve_none():
    published = _published().drop(columns=["final_forecast_rn", "p50_forecast"])
    actuals = _actuals(pd.Timestamp("2026-10-20"))
    assert score_vintage(published, actuals, through=pd.Timestamp("2026-10-20")) is None


# -------------------------------------------------------- forecast_revision


def test_revision_delta_entre_corridas():
    prev = _published(seed=3)
    cur = prev.copy()
    cur["final_forecast_rn"] = cur["final_forecast_rn"] * 1.10  # +10%
    rev = forecast_revision(cur, prev)
    assert rev is not None
    assert rev["n_stays"] == len(prev)
    assert rev["delta_pct"] == pytest.approx(0.10, rel=1e-6)


def test_revision_sin_solape_devuelve_none():
    prev = _published()
    cur = prev.copy()
    cur["stay_date"] = cur["stay_date"] + pd.Timedelta(days=400)
    assert forecast_revision(cur, prev) is None


def test_revision_previo_vacio_devuelve_none():
    prev = _published()
    assert forecast_revision(prev, pd.DataFrame()) is None
    assert forecast_revision(pd.DataFrame(), prev) is None


# ------------------------------------------------------ load_published_csv


def test_load_published_forecast_roundtrip(tmp_path):
    published = _published()
    run_dir = tmp_path / "futuro-1"
    run_dir.mkdir()
    published.to_csv(run_dir / "future_forecast_dataset.csv", index=False)
    loaded = load_published_forecast(run_dir)
    assert loaded is not None
    assert len(loaded) == len(published)
    assert pd.api.types.is_datetime64_any_dtype(loaded["stay_date"])


def test_load_published_forecast_ausente(tmp_path):
    assert load_published_forecast(tmp_path / "no-existe") is None
    run_dir = tmp_path / "vacio"
    run_dir.mkdir()
    (run_dir / "future_forecast_dataset.csv").write_text("columna_rara\n1\n", encoding="utf-8")
    assert load_published_forecast(run_dir) is None


# ------------------------------------------------- detalle vintage (CSV)


def test_vintage_detalle_por_estancia():
    """El detalle trae por estancia: publicado, real, error y banda."""
    from src.monitoring.accuracy_tracking import vintage_detail

    published = _published(n_future=30)
    actuals = _actuals(pd.Timestamp("2026-10-20"))  # 14 estancias con real
    detail = vintage_detail(
        published, actuals,
        through=pd.Timestamp("2026-10-20"),
        published_run_id="futuro-20261020-1304",
    )
    assert not detail.empty
    assert list(detail.columns) == [
        "property", "stay_date", "forecast_date", "lead_days", "model_name",
        "current_booked_rn", "p10_forecast", "published_forecast_rn",
        "p90_forecast", "actual_rn", "error_rn", "error_pct", "in_band",
        "published_run_id",
    ]
    # error coherente por fila
    assert np.allclose(
        detail["error_rn"],
        detail["published_forecast_rn"] - detail["actual_rn"],
    )
    # banda publicada 20-40 contra reales ~N(30,2): dentro
    assert detail["in_band"].mean() >= 0.7
    # trazabilidad de la corrida que publicó
    assert (detail["published_run_id"] == "futuro-20261020-1304").all()
    # orden estable
    assert detail.equals(detail.sort_values(["property", "stay_date"]).reset_index(drop=True))


def test_vintage_detalle_sin_materializados_devuelve_vacio():
    from src.monitoring.accuracy_tracking import vintage_detail

    published = _published(n_future=30)
    actuals = _actuals(pd.Timestamp("2026-10-05"))  # antes del horizonte
    detail = vintage_detail(
        published, actuals, through=pd.Timestamp("2026-10-05")
    )
    assert detail.empty, "sin estancias materializadas el detalle va vacío, nunca inventado"


# ------------------------------------------------- tracking signal (añada)


def _published_flat(n=20, forecast=30.0):
    cutoff = pd.Timestamp("2026-10-06")
    dates = pd.date_range(cutoff + pd.Timedelta(days=1), periods=n)
    return pd.DataFrame({
        "property": ["corin"] * n,
        "stay_date": dates,
        "forecast_date": cutoff,
        "lead_days": list(range(1, n + 1)),
        "final_forecast_rn": [forecast] * n,
        "p50_forecast": [forecast] * n,
        "p10_forecast": [forecast - 5.0] * n,
        "p90_forecast": [forecast + 5.0] * n,
    })


def test_tracking_signal_simetrica_cerca_de_cero():
    """Errores simétricos alrededor del punto → TS ≈ 0 (sin sesgo sistemático)."""
    from src.monitoring.accuracy_tracking import score_vintage

    published = _published_flat(n=20, forecast=30.0)
    dates = published["stay_date"]
    # alternando 29 y 31: errores +1/−1 simétricos
    actuals = pd.DataFrame({
        "property": ["corin"] * len(dates),
        "stay_date": dates,
        "room_nights": [29.0 if i % 2 == 0 else 31.0 for i in range(len(dates))],
    })
    report = score_vintage(published, actuals, through=dates.max())
    ts = report["global"]["tracking_signal"]
    assert abs(ts) < 0.5, f"errores simétricos → TS≈0, obtuve {ts}"


def test_tracking_signal_unidireccional_alta():
    """Errar consistente en UNA dirección → TS alta (el patrón que el MAPE esconde)."""
    from src.monitoring.accuracy_tracking import score_vintage

    published = _published_flat(n=20, forecast=30.0)
    dates = published["stay_date"]
    actuals = pd.DataFrame({
        "property": ["corin"] * len(dates),
        "stay_date": dates,
        "room_nights": [27.0] * len(dates),  # siempre +3 de error
    })
    report = score_vintage(published, actuals, through=dates.max())
    ts = report["global"]["tracking_signal"]
    assert ts > 4.0, f"sesgo sostenido +3 RN en 20 estancias → TS>4, obtuve {ts}"
    assert report["global"]["wmape"] == pytest.approx(60.0 / 540.0)  # 3/27


def test_tracking_signal_ausente_sin_vintage():
    from src.monitoring.accuracy_tracking import score_vintage

    published = _published_flat(n=5)
    actuals = pd.DataFrame({
        "property": ["corin"] * 3,
        "stay_date": published["stay_date"].iloc[:3],
        "room_nights": [30.0, 30.0, 30.0],
    })
    assert score_vintage(published, actuals, through=pd.Timestamp("2026-10-05")) is None
