"""Corrida FUTURA (pronóstico hasta una fecha sin reales aún).

La ventana futura no tiene reales observados; por eso las métricas, los
desgloses y los residuos se inyectan desde un backtest real sobre la ventana
ANTERIOR al corte (`eval_frame`). Estos tests bloquean que:

1. `eval_frame` alimenta métricas + 6 desgloses y los criterios pasan sin
   `actual` (sin inventar reales futuros);
2. el manifiesto etiqueta `metrics_source: injected_backtest` y documenta la
   ventana de pronóstico;
3. los KPIs de accuracy salen del backtest, no de NaN;
4. `per_property_backtest` devuelve la ventana pedida, completamente cubierta
   por reales observados.
"""

import numpy as np
import pandas as pd

from src.forecasting.run_real import per_property_backtest
from src.forecasting.runner import run_end_to_end
from src.monitoring.run_manifest import read_manifest


def _sources():
    """Esquema real (alias + pickup sin snapshot + acumulado), 150 días."""
    dates = pd.date_range("2026-01-01", periods=150, freq="D")
    rng = np.random.default_rng(3)
    dem = pd.DataFrame(
        {
            "property": ["Hotel Royal Corin"] * len(dates),
            "stay_date": dates,
            "rooms_sold": np.clip(60 + rng.normal(0, 8, len(dates)), 0, None).round(),
        }
    )
    cap = pd.DataFrame(
        {
            "property": ["CORIN"] * len(dates),
            "business_date": dates,
            "available_rooms": [90] * len(dates),
        }
    )
    rows = []
    final = dem.set_index("stay_date")["rooms_sold"]
    for stay in dates[::3]:
        for lead in [45, 30, 14, 7, 3, 1]:
            snap = stay - pd.Timedelta(days=lead)
            if snap > pd.Timestamp("2026-04-30"):
                continue
            total = float(final.get(stay, 55.0))
            booked = total * (1 - np.exp(-lead / 25))
            rows.append(
                {
                    "property": "Hotel Royal Corin",
                    "stay_date": stay,
                    "lead_days": lead,
                    "rooms_booked": max(booked, 0.0),
                }
            )
    return dem, cap, pd.DataFrame(rows)


def _eval_frame(dem: pd.DataFrame) -> pd.DataFrame:
    """Backtest inyectado: ventana 2026-04-01..04-30 (dentro de los reales)."""
    win = dem.loc[
        (dem["stay_date"] >= "2026-04-01") & (dem["stay_date"] <= "2026-04-30")
    ].copy()
    win["lead_days"] = 15
    win["forecast"] = win["rooms_sold"] * 1.05
    return win.rename(columns={"rooms_sold": "actual"})[
        ["property", "stay_date", "lead_days", "forecast", "actual"]
    ]


def test_modo_futuro_metricas_desde_backtest_inyectado(tmp_path):
    """Sin `actual` (ventana futura) los criterios pasan vía eval_frame."""
    dem, cap, pick = _sources()
    eval_frame = _eval_frame(dem)
    cutoff = pd.Timestamp("2026-05-31")  # último real; horizonte hacia adelante

    result = run_end_to_end(
        demand=dem,
        capacity=cap,
        pickup=pick,
        run_dir=tmp_path / "futuro",
        cutoff=cutoff,
        actual=None,  # la ventana futura no tiene reales
        eval_frame=eval_frame,
        sample_data=False,
    )

    assert result.checks.all_passed, (
        f"checks fallidos: {result.checks.failures()} details={result.checks.details}"
    )
    # métricas y los 6 desgloses provienen del backtest inyectado
    assert {"mape", "wmape", "mae", "rmse", "bias"} <= set(result.metrics)
    for name in ("lead_band", "property", "season", "month", "day_of_week", "weekend_weekday"):
        assert name in result.breakdowns and not result.breakdowns[name].empty
    # el pronóstico es futuro: después del corte
    ft = result.forecast_table
    assert ft["stay_date"].min() > cutoff
    # pero las métricas NO son las del futuro: son las del backtest previo
    assert result.metrics["n_obs"] == len(eval_frame)


def test_manifest_etiqueta_metrics_source_y_ventana(tmp_path):
    """El manifiesto dice `injected_backtest` y documenta la ventana futura."""
    dem, cap, pick = _sources()
    cutoff = pd.Timestamp("2026-05-31")
    result = run_end_to_end(
        demand=dem,
        capacity=cap,
        pickup=pick,
        run_dir=tmp_path / "futuro_manifest",
        cutoff=cutoff,
        actual=None,
        eval_frame=_eval_frame(dem),
        sample_data=False,
    )
    stored = read_manifest(result.manifest_path)
    assert stored["metrics_source"] == "injected_backtest"
    assert stored["interval_source"] == "real_backtest"
    window = stored["forecast_window"]
    assert window["start"] == "2026-06-01"
    assert pd.Timestamp(window["end"]) == cutoff + pd.Timedelta(days=90)
    # la validación declarada es la ventana futura (convención contigua sin
    # solape: training termina en el corte y validación arranca el día siguiente)
    assert stored["validation_period"]["start"] == "2026-06-01"
    assert stored["training_period"]["end"] == "2026-05-31"
    # documentación técnica: origen explícito de las métricas
    doc = result.deliverables["technical_documentation"].read_text(encoding="utf-8")
    assert "backtest real sobre la ventana ANTERIOR" in doc


def test_kpis_accuracy_vienen_del_backtest(tmp_path):
    """Los KPIs de accuracy no quedan en NaN: se rellenan con el backtest."""
    dem, cap, pick = _sources()
    result = run_end_to_end(
        demand=dem,
        capacity=cap,
        pickup=pick,
        run_dir=tmp_path / "futuro_kpis",
        cutoff=pd.Timestamp("2026-05-31"),
        actual=None,
        eval_frame=_eval_frame(dem),
        sample_data=False,
    )
    kpis = result.checks.details["kpis"]
    assert np.isfinite(kpis["forecast_accuracy_wmape"])
    assert np.isfinite(kpis["forecast_accuracy_mape"])
    assert np.isfinite(kpis["forecast_bias"])
    # coincide con las métricas del backtest
    assert abs(kpis["forecast_accuracy_mape"] - result.metrics["mape"]) < 1e-9


def test_eval_frame_requiere_columnas(tmp_path):
    """eval_frame sin columnas requeridas falla alto (no métricas inventadas)."""
    import pytest

    dem, cap, pick = _sources()
    bad = dem.rename(columns={"rooms_sold": "actual"})[["property", "stay_date", "actual"]]

    with pytest.raises(ValueError, match="eval_frame inyectado sin columnas"):
        run_end_to_end(
            demand=dem,
            capacity=cap,
            pickup=pick,
            run_dir=tmp_path / "never",
            cutoff=pd.Timestamp("2026-05-31"),
            actual=None,
            eval_frame=bad,
            sample_data=False,
        )


def test_per_property_backtest_ventana_cubierta_por_reales():
    """El backtest T-horizon devuelve la ventana pedida, toda con reales."""
    dem, _, pick_raw = _sources()
    from src.config import load_config
    from src.data.schema import SourceKind, detect_schema
    from src.forecasting.runner import _normalize_demand, _normalize_pickup

    cfg = load_config(env={})
    dem_n = _normalize_demand(
        dem, detect_schema(dem, SourceKind.DEMAND, cfg.column_hints)
    )
    pick_n = _normalize_pickup(
        pick_raw, detect_schema(pick_raw, SourceKind.PICKUP, cfg.column_hints)
    )
    cutoff = pd.Timestamp("2026-04-30")
    bt = per_property_backtest(dem_n, pick_n, cutoff, horizon=30)
    assert not bt.empty
    assert {"property", "stay_date", "lead_days", "forecast", "actual"} <= set(bt.columns)
    # ventana exacta: cutoff-29 .. cutoff, sin saltos hacia el futuro
    assert bt["stay_date"].min() == cutoff - pd.Timedelta(days=29)
    assert bt["stay_date"].max() == cutoff
    assert (bt["lead_days"] >= 1).all() and (bt["lead_days"] <= 30).all()
    # todo dentro de reales observados
    assert bt["stay_date"].max() <= dem_n["stay_date"].max()
