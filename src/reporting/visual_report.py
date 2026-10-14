"""Reporte visual HTML por corrida (interfaz de análisis/presentación).

Genera un archivo ``report.html`` autocontenido dentro de la carpeta de la
corrida, con los gráficos de revisión embebidos en base64 (no requiere
navegador, Jupyter ni conexión). Es una vista: toda la lógica vive en
``src/`` (design D1: los notebooks/reportes solo importan y presentan), y no
altera los 10 entregables del spec (reporting §35).

Secciones (todas defensivas ante datos ausentes):

- **EDA**: volumen diario del portafolio.
- **Pickup**: ocupación de los libros por lead time.
- **Benchmark**: WMAPE por modelo candidato con el campeón destacado.
- **Forecast**: pronóstico por propiedad con banda P10/P50/P90.
- **Forecast por propiedad**: small multiples con banda, techo de capacidad,
  reales cuando existen y el modelo que opera cada hotel.
- **Forecast vs reales**: superposición y dispersión cuando hay ``actual``.
- **Accuracy por propiedad**: barras de WMAPE + tabla con modelo y bias.
- **Revisiones T-90..T-1 por propiedad**: WMAPE del backtest muestreado y
  convergencia del error por banda de lead (una línea por propiedad).
- **Ocupación**: forecast contra el techo de capacidad.
"""

from __future__ import annotations

import base64
import io
import math
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin backend interactivo: seguro en CI/servidor

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.evaluation.breakdowns import lead_band as _lead_band


def _with_lead_band(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve ``df`` con una columna ``lead_band`` derivada de ``lead_days``.

    No muta la entrada. Si no hay ``lead_days`` numérico devuelve el frame
    original para que las secciones dependientes simplemente se omitan.
    """
    if "lead_band" in df.columns or "lead_days" not in df.columns:
        return df
    try:
        out = df.copy()
        out["lead_band"] = [_lead_band(v) for v in out["lead_days"]]
        return out
    except (TypeError, ValueError):
        return df

# paleta fija (accesible en fondos claros)
_COLORS = {
    "forecast": "#2563eb",
    "actual": "#dc2626",
    "band": "#93c5fd",
    "capacity": "#64748b",
    "champ": "#16a34a",
    "other": "#cbd5e1",
    "primary": "#1e293b",
}


def _fig_to_base64(fig: plt.Figure) -> str:
    """Figura -> PNG base64 para embeber en ``<img>`` sin archivos sueltos."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode("ascii")


# ------------------------------------------------------------------
# gráficos
# ------------------------------------------------------------------


def _chart_eda(demand: pd.DataFrame | None) -> str | None:
    """Volumen diario agregado del portafolio."""
    if demand is None or demand.empty:
        return None
    d = demand.copy()
    if not {"stay_date", "room_nights"}.issubset(d.columns):
        return None
    d["stay_date"] = pd.to_datetime(d["stay_date"], errors="coerce")
    daily = d.groupby("stay_date")["room_nights"].sum().sort_index()
    if daily.empty:
        return None

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily.index, daily.values, color=_COLORS["primary"], lw=1.2)
    ax.set_title("EDA · Room nights diarios del portafolio", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_pickup(fc: pd.DataFrame) -> str | None:
    """Ocupación de los libros (current booked) por bandas de lead time."""
    if not {"lead_band", "current_booked_rn"}.issubset(fc.columns):
        return None
    work = fc.copy()
    work["current_booked_rn"] = pd.to_numeric(work["current_booked_rn"], errors="coerce")
    work = work.dropna(subset=["current_booked_rn"])
    if work.empty:
        return None
    order = ["1-2", "3-6", "7-13", "14-29", "30-59", "60-89", "90+"]
    present = [b for b in order if b in set(work["lead_band"])]
    if not present:
        return None
    grouped = work.groupby("lead_band")["current_booked_rn"].mean().reindex(present)

    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(grouped.index, grouped.values, color=_COLORS["forecast"], alpha=0.85)
    ax.set_title("Pickup · RN ya reservados (promedio) por anticipación", loc="left", fontsize=11)
    ax.set_ylabel("RN reservados")
    ax.grid(alpha=0.25, axis="y")
    return _fig_to_base64(fig)


def _chart_benchmark(benchmark: pd.DataFrame | None) -> str | None:
    """WMAPE por modelo candidato, destacando al campeón."""
    if benchmark is None or benchmark.empty or "wmape" not in benchmark.columns:
        return None
    b = benchmark.copy()
    b = b.dropna(subset=["wmape"])
    if b.empty:
        return None
    champion = None
    if "is_champion" in b.columns and b["is_champion"].astype(bool).any():
        champion = b.loc[b["is_champion"].astype(bool), "model"].iloc[0]
    colors = [_COLORS["champ"] if m == champion else _COLORS["other"] for m in b["model"]]

    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.bar(b["model"], b["wmape"], color=colors)
    ax.set_title("Benchmark · WMAPE por modelo (menor = mejor)", loc="left", fontsize=11)
    ax.set_ylabel("WMAPE")
    ax.tick_params(axis="x", rotation=30)
    ax.grid(alpha=0.25, axis="y")
    if champion:
        ax.text(0.99, 0.95, f"campeón: {champion}", transform=ax.transAxes,
                ha="right", va="top", fontsize=9, color=_COLORS["champ"])
    return _fig_to_base64(fig)


def _chart_forecast(fc: pd.DataFrame) -> str | None:
    """Serie de forecast agregada por fecha con banda P10/P50/P90."""
    if "stay_date" not in fc.columns:
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work.get("final_forecast_rn"), errors="coerce")
    if work["final_forecast_rn"].notna().sum() == 0:
        return None
    daily = work.groupby("stay_date").sum(numeric_only=True).sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.4))
    p10 = daily.get("p10_forecast")
    p90 = daily.get("p90_forecast")
    if p10 is not None and p90 is not None and p10.notna().any():
        ax.fill_between(daily.index, p10, p90, color=_COLORS["band"], alpha=0.5, label="P10–P90")
    ax.plot(daily.index, daily["final_forecast_rn"], color=_COLORS["forecast"], lw=1.8, label="Forecast")
    ax.set_title("Forecast · Room nights del portafolio con banda de incertidumbre", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _chart_forecast_vs_actual(fc: pd.DataFrame, actual: pd.Series | None) -> str | None:
    """Superposición forecast vs reales (solo cuando hay reales alineados)."""
    if actual is None or "stay_date" not in fc.columns:
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work.get("final_forecast_rn"), errors="coerce")
    work["_actual"] = pd.to_numeric(actual, errors="coerce").reindex(work.index)
    if work[["_actual", "final_forecast_rn"]].dropna().empty:
        return None
    daily = work.groupby("stay_date")[["final_forecast_rn", "_actual"]].sum().sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(daily.index, daily["final_forecast_rn"], color=_COLORS["forecast"], lw=1.6, label="Forecast")
    ax.plot(daily.index, daily["_actual"], color=_COLORS["actual"], lw=1.4, label="Real")
    ax.set_title("Forecast vs Real · RN del portafolio", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


def _scatter_forecast_vs_actual(fc: pd.DataFrame, actual: pd.Series | None) -> str | None:
    """Dispersión forecast vs real con diagonal de referencia."""
    if actual is None:
        return None
    f = pd.to_numeric(fc.get("final_forecast_rn"), errors="coerce")
    a = pd.to_numeric(actual, errors="coerce").reindex(fc.index)
    mask = f.notna() & a.notna()
    if mask.sum() == 0:
        return None
    x, y = a[mask], f[mask]

    fig, ax = plt.subplots(figsize=(4.6, 4.2))
    lim = max(x.max(), y.max()) * 1.05 or 1
    ax.plot([0, lim], [0, lim], ls="--", c=_COLORS["capacity"], lw=1, label="ideal")
    ax.scatter(x, y, s=12, alpha=0.5, c=_COLORS["forecast"])
    ax.set_xlabel("Real RN")
    ax.set_ylabel("Forecast RN")
    ax.set_title("Dispersión", loc="left", fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


def _chart_occupancy(fc: pd.DataFrame) -> str | None:
    """Forecast contra el techo de capacidad (agregado diario)."""
    if not {"stay_date", "final_forecast_rn", "capacity"}.issubset(fc.columns):
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work["final_forecast_rn"], errors="coerce")
    work["capacity"] = pd.to_numeric(work["capacity"], errors="coerce")
    if work["capacity"].notna().sum() == 0:
        return None
    daily = work.groupby("stay_date")[["final_forecast_rn", "capacity"]].sum().sort_index()

    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.plot(daily.index, daily["capacity"], color=_COLORS["capacity"], lw=1.4, ls="--", label="Capacidad")
    ax.plot(daily.index, daily["final_forecast_rn"], color=_COLORS["forecast"], lw=1.6, label="Forecast")
    ax.fill_between(daily.index, daily["final_forecast_rn"], daily["capacity"],
                    where=daily["final_forecast_rn"] > daily["capacity"],
                    color=_COLORS["actual"], alpha=0.15, label="excede techo")
    ax.set_title("Ocupación · forecast contra techo de capacidad", loc="left", fontsize=11)
    ax.set_ylabel("RN")
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%b-%y"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)
    fig.autofmt_xdate()
    return _fig_to_base64(fig)


# ------------------------------------------------------------------
# gráficos por propiedad (remediación de la vista: la operación es por hotel)
# ------------------------------------------------------------------


def _chart_forecast_by_property(
    fc: pd.DataFrame, actual: pd.Series | None, model_by_property: dict | None
) -> str | None:
    """Small multiples: P50 + banda P10/P90 + capacidad + reales por propiedad."""
    if "property" not in fc.columns or "stay_date" not in fc.columns:
        return None
    work = fc.copy()
    work["stay_date"] = pd.to_datetime(work["stay_date"], errors="coerce")
    work["final_forecast_rn"] = pd.to_numeric(work.get("final_forecast_rn"), errors="coerce")
    work["_actual"] = (
        pd.to_numeric(actual, errors="coerce").reindex(work.index)
        if actual is not None else float("nan")
    )
    props = sorted(work["property"].dropna().unique())
    if not props:
        return None
    ncols = 3 if len(props) > 3 else max(len(props), 1)
    nrows = math.ceil(len(props) / ncols)
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.3 * ncols, 2.9 * nrows), squeeze=False)
    for ax, prop in zip(axes.flat, props):
        g = work.loc[work["property"] == prop].sort_values("stay_date")
        model = (model_by_property or {}).get(str(prop))
        ax.set_title(str(prop) + (f" · {model}" if model else ""), loc="left", fontsize=9)
        if g["final_forecast_rn"].notna().sum() == 0:
            ax.axis("off")
            continue
        if {"p10_forecast", "p90_forecast"} <= set(g.columns):
            p10 = pd.to_numeric(g["p10_forecast"], errors="coerce")
            p90 = pd.to_numeric(g["p90_forecast"], errors="coerce")
            if p10.notna().any() and p90.notna().any():
                ax.fill_between(g["stay_date"], p10, p90, color=_COLORS["band"], alpha=0.5)
        ax.plot(g["stay_date"], g["final_forecast_rn"], color=_COLORS["forecast"], lw=1.4)
        if "capacity" in g.columns:
            cap = pd.to_numeric(g["capacity"], errors="coerce")
            if cap.notna().any():
                ax.plot(g["stay_date"], cap, color=_COLORS["capacity"], lw=1.0, ls="--")
        if g["_actual"].notna().any():
            ax.plot(g["stay_date"], g["_actual"], color=_COLORS["actual"], lw=1.1)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%b"))
    for ax in axes.flat[len(props):]:
        ax.axis("off")
    fig.tight_layout(h_pad=1.2, w_pad=1.2)
    return _fig_to_base64(fig)


def _wmape_color(value: float) -> str:
    """Semáforo de WMAPE: verde <15%, azul <30%, rojo ≥30%."""
    if value < 0.15:
        return _COLORS["champ"]
    if value < 0.30:
        return _COLORS["forecast"]
    return _COLORS["actual"]


def _chart_accuracy_by_property(breakdown: pd.DataFrame | None) -> str | None:
    """Barras horizontales de WMAPE por propiedad, anotadas con bias."""
    if breakdown is None or breakdown.empty or "wmape" not in breakdown.columns:
        return None
    b = breakdown.copy().dropna(subset=["wmape"])
    if b.empty:
        return None
    b = b.sort_values("wmape")
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(b) + 1.3))
    ax.barh(b["property"].astype(str), b["wmape"], color=[_wmape_color(v) for v in b["wmape"]])
    for y, (w, bias) in enumerate(zip(b["wmape"], b.get("bias", pd.Series(dtype=float)))):
        ax.text(w, y, f"  bias {bias:+.3f}", va="center", fontsize=7.5, color="#475569")
    ax.set_title("WMAPE por propiedad (menor = mejor; verde <15%, azul <30%)",
                 loc="left", fontsize=11)
    ax.set_xlabel("WMAPE")
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25, axis="x")
    return _fig_to_base64(fig)


def _table_accuracy_by_property(
    breakdown: pd.DataFrame | None,
    model_by_property: dict | None,
    weekly_breakdown: pd.DataFrame | None = None,
) -> str:
    """Tabla propiedad | modelo | WMAPE | WMAPE sem | MAPE | bias | n_obs.

    El MAPE diario lleva "†" cuando el decil inferior de los reales
    evaluados cae por debajo de 10 RN: con reales de pocas noches, ±5 RN
    de error ya son '+125%' — el MAPE diario es ruido de denominador,
    no señal de precisión (la semana mide la operación real).
    """
    if breakdown is None or breakdown.empty:
        return ""
    b = breakdown.copy().sort_values("wmape")
    weekly: dict[str, float] = {}
    weekly_note = False
    if weekly_breakdown is not None and not weekly_breakdown.empty:
        wb = weekly_breakdown
        if "property" in wb.columns and "wmape" in wb.columns:
            weekly = {
                str(p): float(w) for p, w in zip(wb["property"], wb["wmape"])
                if pd.notna(w)
            }
    rows = []
    for _, r in b.iterrows():
        prop = str(r.get("property"))
        model = (model_by_property or {}).get(prop, "—")
        def _f(key: str, fmt: str) -> str:
            try:
                return format(float(r[key]), fmt)
            except (KeyError, TypeError, ValueError):
                return "—"
        mape_cell = _f("mape", ".1f")
        try:
            p10 = float(r["p10_actual"])
        except (KeyError, TypeError, ValueError):
            p10 = float("nan")
        small_denominator = bool(pd.notna(p10) and p10 < 10.0)
        if small_denominator:
            mape_cell += "†"
            weekly_note = True
        w_sem = weekly.get(prop)
        w_sem_cell = format(w_sem, ".3f") if w_sem is not None else "—"
        if w_sem is not None:
            weekly_note = True
        rows.append(
            f"<tr><td>{prop}</td><td>{model}</td>"
            f"<td>{_f('wmape', '.3f')}</td><td>{w_sem_cell}</td>"
            f"<td>{mape_cell}</td>"
            f"<td>{_f('bias', '+.3f')}</td><td>{_f('n_obs', ',')}</td></tr>"
        )
    nota = ""
    if weekly_note:
        nota = (
            '<p style="font-size:0.78em;color:#64748b;margin:4px 0 0 0">'
            "WMAPE sem = precisión agregando forecast y reales por semana "
            "(grano operativo). †MAPE diario con reales diminutos "
            "(P10 < 10 RN): ler con el grano semanal.</p>"
        )
    return (
        '<table class="tbl"><thead><tr><th>Propiedad</th><th>Modelo</th>'
        "<th>WMAPE</th><th>WMAPE sem</th><th>MAPE</th><th>Bias</th>"
        "<th>n obs</th></tr></thead>"
        "<tbody>" + "".join(rows) + "</tbody></table>" + nota
    )


def _chart_backtest_by_property(backtest_report: dict | None) -> str | None:
    """WMAPE de las revisiones T-90..T-1 por propiedad (campeón operante)."""
    if not backtest_report:
        return None
    rows = backtest_report.get("by_property") or []
    if not rows:
        return None
    df = pd.DataFrame(rows).dropna(subset=["wmape"]).sort_values("wmape")
    if df.empty:
        return None
    labels = [f"{r['property']} ({r.get('model', '?')})" for _, r in df.iterrows()]
    fig, ax = plt.subplots(figsize=(7, 0.55 * len(df) + 1.5))
    ax.barh(labels, df["wmape"], color=[_wmape_color(v) for v in df["wmape"]])
    for y, (w, n) in enumerate(zip(df["wmape"], df.get("n_obs", 0))):
        ax.text(w, y, f"  n={int(n)}", va="center", fontsize=7.5, color="#475569")
    ax.set_title("Revisiones T-90..T-1 · WMAPE por propiedad (backtest muestreado)",
                 loc="left", fontsize=11)
    ax.set_xlabel("WMAPE")
    ax.tick_params(labelsize=8.5)
    ax.grid(alpha=0.25, axis="x")
    return _fig_to_base64(fig)


def _chart_revisions_by_lead(backtest_report: dict | None) -> str | None:
    """Convergencia: WMAPE por banda de lead (una línea por propiedad)."""
    if not backtest_report:
        return None
    rows = backtest_report.get("by_property_lead_band") or []
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if not {"property", "lead_band", "wmape"} <= set(df.columns):
        return None
    order = ["90+", "60-89", "30-59", "14-29", "7-13", "3-6", "1-2"]
    present = [bnd for bnd in order if bnd in set(df["lead_band"])]
    if not present:
        return None
    x = range(len(present))
    fig, ax = plt.subplots(figsize=(8.4, 3.8))
    for prop, g in df.groupby("property"):
        s = g.set_index("lead_band")["wmape"].reindex(present)
        ax.plot(x, s.values, marker="o", ms=3.5, lw=1.4, label=str(prop))
    ax.set_xticks(list(x))
    ax.set_xticklabels(present)
    ax.set_title("Convergencia · WMAPE por anticipación (T-90 → T-1, una línea por propiedad)",
                 loc="left", fontsize=11)
    ax.set_xlabel("Anticipación (banda de lead)")
    ax.set_ylabel("WMAPE")
    ax.legend(fontsize=7.5, ncol=min(3, df["property"].nunique()), loc="upper right")
    ax.grid(alpha=0.25)
    return _fig_to_base64(fig)


def _table_backtest_lead_band(backtest_report: dict | None) -> str:
    """Tabla global del backtest por banda de lead."""
    if not backtest_report:
        return ""
    rows = backtest_report.get("by_lead_band") or []
    if not rows:
        return ""
    order = ["90+", "60-89", "30-59", "14-29", "7-13", "3-6", "1-2"]
    df = pd.DataFrame(rows)
    present = [b for b in order if b in set(df.get("lead_band", []))]
    if present:
        df = df.set_index("lead_band").reindex(present).reset_index()
    cells = []
    for _, r in df.iterrows():
        def _f(key: str, fmt: str) -> str:
            try:
                return format(float(r[key]), fmt)
            except (KeyError, TypeError, ValueError):
                return "—"
        cells.append(
            f"<tr><td>{r.get('lead_band', '—')}</td><td>{_f('n_obs', ',')}</td>"
            f"<td>{_f('wmape', '.3f')}</td><td>{_f('mape', '.1f')}</td></tr>"
        )
    return (
        '<table class="tbl"><thead><tr><th>Banda</th><th>n obs</th>'
        "<th>WMAPE</th><th>MAPE</th></tr></thead><tbody>"
        + "".join(cells) + "</tbody></table>"
    )


# ------------------------------------------------------------------
# ensamblado HTML
# ------------------------------------------------------------------


def _img_tag(b64: str | None, alt: str) -> str:
    if not b64:
        return f'<p class="muted">— {alt}: sin datos en esta corrida.</p>'
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}"/>'


def _kpi_grid(kpis: dict | None, champion: str | None = None) -> str:
    if not kpis and not champion:
        return ""
    labels = {
        "forecast_room_nights": "Forecast RN",
        "forecast_occupancy": "Ocupación",
        "forecast_bias": "Bias",
        "expected_pickup": "Pickup esperado",
        "capacity_utilization": "Uso capacidad",
        "n_properties_evaluated": "Propiedades evaluadas",
        "forecast_accuracy_wmape": "WMAPE",
        "forecast_accuracy_mape": "MAPE",
    }
    cells = []
    if champion:
        cells.append(
            f'<div class="kpi"><span class="kpi-l">Campeón</span>'
            f'<span class="kpi-v">{champion}</span></div>'
        )
    for key, label in labels.items():
        if not kpis or key not in kpis:
            continue
        val = kpis[key]
        try:
            val = float(val)
            if key == "n_properties_evaluated":
                total = float(kpis.get("n_properties_portfolio", 0) or 0)
                shown = f"{val:.0f} / {total:.0f}"
            elif "occupancy" in key or "utilization" in key:
                shown = f"{val:.1%}"
            elif key == "forecast_accuracy_mape":
                shown = f"{val:.1f}%"
            elif key == "forecast_accuracy_wmape":
                shown = f"{val:.3f}"
            else:
                shown = f"{val:,.1f}"
        except (TypeError, ValueError):
            shown = str(val)
        cells.append(f'<div class="kpi"><span class="kpi-l">{label}</span>'
                     f'<span class="kpi-v">{shown}</span></div>')
    if not cells:
        return ""
    return '<div class="kpis">' + "".join(cells) + "</div>"


def _fmt_metrics(metrics: dict | None) -> str:
    if not metrics:
        return '<p class="muted">Sin métricas en esta corrida.</p>'
    rows = []
    for k, v in metrics.items():
        try:
            rows.append(f"<tr><td>{k}</td><td>{float(v):,.4f}</td></tr>")
        except (TypeError, ValueError):
            rows.append(f"<tr><td>{k}</td><td>{v}</td></tr>")
    return ('<table class="tbl"><thead><tr><th>Métrica</th><th>Valor</th></tr></thead>'
            "<tbody>" + "".join(rows) + "</tbody></table>")


def generate_visual_report(
    run_dir: str | Path,
    forecast: pd.DataFrame,
    *,
    demand: pd.DataFrame | None = None,
    benchmark: pd.DataFrame | None = None,
    metrics: dict | None = None,
    kpis: dict | None = None,
    actual: pd.Series | None = None,
    property_breakdown: pd.DataFrame | None = None,
    property_weekly_breakdown: pd.DataFrame | None = None,
    model_by_property: dict | None = None,
    backtest_report: dict | None = None,
    champion: str | None = None,
    title: str = "Reporte de revisión · Forecast de Room Nights",
) -> Path:
    """Escribe ``<run_dir>/report.html`` con los gráficos embebidos.

    Todos los parámetros de datos son opcionales: las secciones ausentes se
    omiten con un mensaje en lugar de fallar la corrida. El detalle por
    propiedad (forecast/bandas/modelo, accuracy y revisiones del backtest
    T-90..T-1) se activa con ``property_breakdown``/``model_by_property``/
    ``backtest_report``/``champion``; ``property_weekly_breakdown`` agrega
    la columna de WMAPE semanal a la tabla de accuracy; sin ellos el
    reporte queda como la vista de portafolio clásica.
    """
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    forecast = _with_lead_band(forecast)
    actual_series = actual
    if actual_series is not None:
        actual_series = pd.Series(actual_series).reset_index(drop=True)
        if len(actual_series) != len(forecast):
            actual_series = None

    sections = [
        ("Forecast", _chart_forecast(forecast)),
        ("Forecast vs Real", _chart_forecast_vs_actual(forecast, actual_series)),
        ("Forecast por propiedad",
         _chart_forecast_by_property(forecast, actual_series, model_by_property)),
        ("Benchmark", _chart_benchmark(benchmark)),
        ("Pickup", _chart_pickup(forecast)),
        ("Ocupación", _chart_occupancy(forecast)),
        ("EDA", _chart_eda(demand)),
    ]
    scatter = _scatter_forecast_vs_actual(forecast, actual_series)

    cards = []
    for name, img in sections:
        if img is None and name in {"Forecast vs Real"}:
            continue
        cards.append(f'<section class="card"><h2>{name}</h2>{_img_tag(img, name)}</section>')
    if scatter:
        cards.append(f'<section class="card narrow"><h2>Dispersión</h2>'
                     f"{_img_tag(scatter, 'dispersión')}</section>")

    # ---- accuracy por propiedad: barras + tabla exacta en la misma tarjeta
    acc_img = _chart_accuracy_by_property(property_breakdown)
    acc_tbl = _table_accuracy_by_property(
        property_breakdown, model_by_property, property_weekly_breakdown
    )
    if acc_img or acc_tbl:
        cards.append(
            '<section class="card"><h2>Accuracy por propiedad</h2>'
            f"{_img_tag(acc_img, 'accuracy por propiedad')}{acc_tbl}</section>"
        )

    # ---- revisiones T-90..T-1: por propiedad + convergencia por lead band
    rev_prop = _chart_backtest_by_property(backtest_report)
    rev_lead = _chart_revisions_by_lead(backtest_report)
    rev_tbl = _table_backtest_lead_band(backtest_report)
    if rev_prop or rev_lead or rev_tbl:
        cards.append(
            '<section class="card"><h2>Revisiones T-90..T-1 · por propiedad</h2>'
            f"{_img_tag(rev_prop, 'revisiones por propiedad')}"
            f"{_img_tag(rev_lead, 'convergencia por anticipación')}"
            f"{rev_tbl}</section>"
        )

    stamp = datetime.now().strftime("%Y-%m-%d %H:%M")
    html = f"""<!DOCTYPE html>
<html lang="es"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
  :root {{ color-scheme: light; }}
  body {{ font-family: -apple-system, Segoe UI, Roboto, sans-serif; margin: 0;
          background: #f8fafc; color: #1e293b; }}
  header {{ background: #0f172a; color: #f8fafc; padding: 20px 28px; }}
  header h1 {{ margin: 0; font-size: 20px; font-weight: 600; }}
  header p {{ margin: 4px 0 0; color: #94a3b8; font-size: 13px; }}
  main {{ padding: 24px 28px; max-width: 1100px; margin: 0 auto; }}
  .kpis {{ display: flex; flex-wrap: wrap; gap: 12px; margin-bottom: 20px; }}
  .kpi {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 10px;
          padding: 12px 18px; min-width: 150px; }}
  .kpi-l {{ display: block; font-size: 12px; color: #64748b; text-transform: uppercase; }}
  .kpi-v {{ display: block; font-size: 22px; font-weight: 650; margin-top: 2px; }}
  .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 20px; }}
  .card {{ background: #fff; border: 1px solid #e2e8f0; border-radius: 12px;
           padding: 18px 20px; }}
  .card.narrow {{ grid-column: span 1; }}
  .card h2 {{ margin: 0 0 12px; font-size: 14px; font-weight: 600; color: #334155;
              text-transform: uppercase; letter-spacing: .03em; }}
  .card img {{ width: 100%; height: auto; margin-bottom: 10px; }}
  .card img:last-child {{ margin-bottom: 0; }}
  .muted {{ color: #94a3a3; font-style: italic; }}
  table.tbl {{ border-collapse: collapse; width: 100%; font-size: 14px; }}
  table.tbl th, table.tbl td {{ text-align: left; padding: 6px 10px;
                                border-bottom: 1px solid #e2e8f0; }}
  table.tbl th {{ color: #64748b; font-weight: 600; }}
  footer {{ color: #94a3b8; font-size: 12px; padding: 16px 28px 32px; text-align: center; }}
  .disc {{ background: #fef9c3; border: 1px solid #fde047; border-radius: 8px;
           padding: 10px 14px; font-size: 13px; margin-bottom: 18px; }}
</style></head>
<body>
<header>
  <h1>{title}</h1>
  <p>Generado {stamp} · vista de análisis (la lógica vive en src/, este reporte solo presenta)</p>
</header>
<main>
  {_kpi_grid(kpis, champion)}
  <div class="grid">
    {"".join(cards)}
    <section class="card"><h2>Métricas</h2>{_fmt_metrics(metrics)}</section>
  </div>
</main>
<footer>Vista generada automáticamente. Los 10 entregables del spec permanecen en la raíz de la corrida.</footer>
</body></html>"""

    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out


__all__ = ["generate_visual_report"]
