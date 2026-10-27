"""Reporte mensual de vintages por corte (día 15 de cada mes).

Entregable de negocio: para cada corte (15 de cada mes, ene-oct 2026 — los
posteriores aún no existen), la proyección del mes en curso + los 2
siguientes, contra el real de cada mes proyectado. Formato solicitado:

    Mes proyectado | Propiedad | Fecha corte | RN Real | RN Pronosticado | Nota

- Mes en curso al corte: proyección de cierre de mes = reales 1-15 +
  pronóstico 16-fin (la proyección clásica a mitad de mes).
- Meses futuros al corte: pronóstico del mes completo.
- RN Real: reales del mes completo (vacío si el mes no ha terminado).
- Los modelos son los operativos vigentes, ajustados AS-OF cada corte
  (fit con historia ≤ corte): es una reconstrucción honesta con el motor,
  no lo que se publicó entonces (el motor no existía en enero 2026).

Uso:
    python scripts/monthly_vintage_report.py [salida.xlsx]
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import pandas as pd

from src.config import load_config
from src.data.schema import SourceKind, detect_schema
from src.features.events import load_events
from src.features.holidays import (
    HolidayCalendar,
    calendar_with_events,
    holiday_calendar_from_config,
)
from src.forecasting.core import (
    _apply_aliases,
    _normalize_capacity,
    _normalize_demand,
    _normalize_pickup,
    build_forecasts,
)

MESES = ["ene", "feb", "mar", "abr", "may", "jun",
         "jul", "ago", "sep", "oct", "nov", "dic"]

# selección operativa vigente (corrida futura 2026-10-26)
MODELOS = {
    "CORIN": "pickup_curve",
    "FIESTA": "seasonal_naive",
    "LIREL": "pickup_curve",
    "MARINA": "catboost",
    "SJOSL": "pickup_curve",
    "VILLAS": "pickup_curve",
}


def _load_all():
    cfg = load_config()
    demand = pd.read_csv(cfg.source_paths["demand"])
    capacity = pd.read_csv(cfg.source_paths["capacity"])
    pickup = pd.read_csv(cfg.source_paths["pickup"])
    ds = detect_schema(demand, SourceKind.DEMAND, cfg.column_hints)
    cs = detect_schema(capacity, SourceKind.CAPACITY, cfg.column_hints)
    ps = detect_schema(pickup, SourceKind.PICKUP, cfg.column_hints)
    demand = _apply_aliases(demand, ds, cfg.property_aliases)
    capacity = _apply_aliases(capacity, cs, cfg.property_aliases)
    pickup = _apply_aliases(pickup, ps, cfg.property_aliases)
    excl = [str(p) for p in cfg.excluded_properties]
    for df, col in ((demand, ds.property_col), (capacity, cs.property_col),
                    (pickup, ps.property_col)):
        df.drop(df[df[col].astype(str).isin(excl)].index, inplace=True)
    dem = _normalize_demand(demand, ds)
    _normalize_capacity(capacity, cs)
    pick = _normalize_pickup(pickup, ps)
    cal = holiday_calendar_from_config(
        cfg, min_year=2025, max_year=2027,
    )
    events = load_events(cfg.events_path)
    if events is not None and cal is not None:
        cal = calendar_with_events(cal, events)
    return cfg, dem, pick, cal


def main(out_path: str | None = None) -> int:
    cfg, dem, pick, cal = _load_all()
    curve_params = dict(cfg.pickup_curve)
    ultimo_real = pd.Timestamp(dem["stay_date"].max())

    # reales mensuales por propiedad
    d = dem.copy()
    d["ym"] = d["stay_date"].dt.to_period("M")
    reales = d.groupby(["property", "ym"])["room_nights"].sum()

    rows = []
    for mes_corte in range(1, 11):  # cortes 15-ene ... 15-oct 2026
        corte = pd.Timestamp(2026, mes_corte, 15)
        if corte > ultimo_real:
            continue
        meses_proyectados = [mes_corte + k for k in range(3)]  # mes en curso + 2
        if mes_corte + 2 <= 12:
            fin_horizonte = pd.Timestamp(2026, mes_corte + 2, 1) + pd.offsets.MonthEnd(0)
        else:
            fin_horizonte = pd.Timestamp("2026-12-31")
        horizonte = (fin_horizonte - corte).days

        try:
            raw, _, _ = build_forecasts(
                MODELOS, dem, pick, corte, horizonte, curve_params, cal,
            )
        except Exception as exc:
            print(f"[aviso] corte {corte.date()}: {exc}")
            continue
        raw["ym"] = pd.to_datetime(raw["stay_date"]).dt.to_period("M")
        fc_mes = raw.groupby(["property", "ym"])["raw_forecast_rn"].sum()

        for prop in sorted(dem["property"].unique()):
            for m in meses_proyectados:
                if m > 12:
                    continue
                ym = pd.Period(f"2026-{m:02d}", freq="M")
                fc = fc_mes.get((prop, ym))
                if fc is None or pd.isna(fc):
                    continue
                if m == mes_corte:
                    # proyección de cierre: reales 1-15 + pronóstico 16-fin
                    mtd = dem[(dem["property"] == prop)
                              & (dem["stay_date"] >= ym.to_timestamp())
                              & (dem["stay_date"] <= corte)]["room_nights"].sum()
                    pron = float(mtd) + float(fc)
                    nota = "proyección de cierre de mes (reales 1-15 + pronóstico)"
                else:
                    pron = float(fc)
                    nota = ""
                real = reales.get((prop, ym))
                fin_mes = ym.to_timestamp(how="end").normalize()
                if pd.isna(real):
                    real_val, nota = None, (nota + " · mes sin reales completos").strip(" ·")
                elif fin_mes > ultimo_real:
                    # mes en curso con reales parciales: honestar el corte
                    real_val = float(real)
                    nota = (nota + f" · reales parciales al {ultimo_real.date()}").strip(" ·")
                else:
                    real_val = float(real)
                rows.append({
                    "Mes proyectado": f"{MESES[m - 1]}-2026",
                    "Propiedad": prop,
                    "Fecha corte": f"15-{MESES[mes_corte - 1]}-2026",
                    "RN Real": real_val,
                    "RN Pronosticado": round(pron, 1),
                    "Nota": nota,
                })

    df = pd.DataFrame(rows)
    out = Path(out_path or (_ROOT / "outputs" / "reporte_mensual_vintages_2026.xlsx"))
    out.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(out, engine="openpyxl") as xw:
        df.to_excel(xw, sheet_name="vintages", index=False)
        # hoja de lectura
        lectura = pd.DataFrame({
            "Qué es": [
                "Proyección de cada mes vista desde el corte del día 15 (mes en curso + 2 siguientes), contra el real del mes completo.",
                "Mes en curso al corte: RN Pronosticado = reales 1-15 + pronóstico 16-fin (proyección de cierre).",
                "Meses futuros al corte: RN Pronosticado = pronóstico del mes completo.",
                "RN Real: reales del mes completo; vacío si el mes no ha terminado.",
                "Los modelos son los operativos vigentes del motor, ajustados AS-OF cada corte (sin ver el futuro): reconstrucción honesta, no lo publicado entonces.",
                "Cortes nov-15 y dic-15 aún no existen (reales al 25-oct): nov y dic tienen menos vintages.",
            ],
        })
        lectura.to_excel(xw, sheet_name="lectura", index=False)
    print(f"escrito: {out} ({len(df)} filas)")
    print(df.head(12).to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1] if len(sys.argv) > 1 else None))
