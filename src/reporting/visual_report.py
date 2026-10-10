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
- **Forecast vs reales**: superposición y dispersión cuando hay ``actual``.
- **Ocupación**: forecast contra el techo de capacidad.
"""

from __future__ import annotations

import base64
import io
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
# ensamblado HTML
# ------------------------------------------------------------------


def _img_tag(b64: str | None, alt: str) -> str:
    if not b64:
        return f'<p class="muted">— {alt}: sin datos en esta corrida.</p>'
    return f'<img alt="{alt}" src="data:image/png;base64,{b64}"/>'


def _kpi_grid(kpis: dict | None) -> str:
    if not kpis:
        return ""
    labels = {
        "forecast_room_nights": "Forecast RN",
        "forecast_occupancy": "Ocupación",
        "forecast_bias": "Bias",
        "expected_pickup": "Pickup esperado",
        "capacity_utilization": "Uso capacidad",
    }
    cells = []
    for key, label in labels.items():
        if key not in kpis:
            continue
        val = kpis[key]
        try:
            val = float(val)
            shown = f"{val:.1%}" if ("occupancy" in key or "utilization" in key) else f"{val:,.1f}"
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
    title: str = "Reporte de revisión · Forecast de Room Nights",
) -> Path:
    """Escribe ``<run_dir>/report.html`` con los gráficos embebidos.

    Todos los parámetros de datos son opcionales: las secciones ausentes se
    omiten con un mensaje en lugar de fallar la corrida.
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
  .card img {{ width: 100%; height: auto; }}
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
  {_kpi_grid(kpis)}
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
