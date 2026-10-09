# Hotel Room-Nights Forecasting Pipeline

**Enjoy Costa Rica — Revenue Analytics**

Pronóstico de **room nights** (noches vendidas, `rooms_sold`) con **protocolo
honesto** sobre las fuentes reales de producción: snapshots de reservas
(`staging.enjoy_fac_hotel`) + inventario oficial (`public.enjoy_inventory`).

> **Protocolo honesto**: evaluación multi-paso real (recursiva — los reales del
> test jamás son entradas), baseline SeasonalNaive obligatorio, backtesting
> rolling-origin, test Diebold-Mariano con Newey-West, techo de capacidad,
> bandas conformales con cobertura reportada, sesgo/dirección del error,
> cold-start declarado. Datos sintéticos siempre con disclaimer.

## Pipelines

| Pipeline | Target | Datos | Notebook |
|---|---|---|---|
| **PRODUCCIÓN (semanal, ~3 min)** | `rooms_sold` por propiedad + portafolio — **horizonte 365d (12 meses, detalle diario)** | 3 agregados SQL — ver [README_SISTEMAS.md](README_SISTEMAS.md) | `hotel_roomnights_production.ipynb` |
| **Re-validación (trimestral/anual, ~50 min)** | bake-off de 9 modelos + DM + backtest + selección por propiedad (folds 90d) | mismos agregados | `hotel_roomnights_realdata.ipynb` |
| Room nights v1 (archivado) | `rooms_sold` portafolio | `hotel_roomnights_sample.csv` (sintético) | `hotel_roomnights_forecasting.ipynb` |
| Revenue (archivado) | `revenue` / `room_revenue` | `hotel_data.csv` | `hotel_revenue_forecasting.ipynb` |

**Arquitectura campeón/challenger (v0.8)**: producción corre un campeón fijo
(`Prophet(0.01,10)` por propiedad, guard SeasonalNaive <120d, congeladas excluidas)
con pisos honestos en cada corrida (SeasonalNaive y SN365) y gatillos de
re-validación (edge <5% vs SN365, cadencia trimestral, eventos de composición).

## La arquitectura real (v0.4)

```
┌──────────────── staging.enjoy_fac_hotel (~14M filas, NUNCA salen de la base) ┐
│  snap_flag = 0 (realizado)  →  TARGET: room nights por (property, stay_date) │
│  snap_flag = 1 (futuro)     →  PICKUP: libros por anticipación (lead_days)  │
├──────────────── public.enjoy_inventory ──────────────────────────────────────┤
│  oficial_inventory          →  TECHO: capacidad + ocupación implícita       │
└──────────────────────────────────────────────────────────────────────────────┘
        │ 3 queries de agregación (SQL pushdown) → 3 CSVs   [README_SISTEMAS.md]
        ▼
  Pipeline (73 tests): mapeo canónico de propiedades · features as-of seguras
  (rotb_d90/180/365, correlación 0.93 con lo realizado) · recursión incremental
  verificada · tuning honesto (objetivo multi-paso) · backtesting 6 ventanas ·
  bandas conformales · per-propiedad + cold-start (SJOSL)
        ▼
  Pronóstico 90 días: portafolio + por propiedad, con libros actuales como
  covariantes, techo de capacidad, ocupación y cobertura de libros (% ya vendido)
```

## Resultados del último run (agregados reales-schema sintéticos)

- Ganador: **XGBoost** — MAE 33.7 noches/día, WAPE 3.8% (~96% de precisión)
- Backtesting (6 ventanas × 90d): rango medio 1.8/8 — estable en top-2
- Pronóstico 90d: 70,883 noches · ocupación 54.3% · banda 80%: 69,099–75,040
- **43.3% del pronóstico ya está en los libros** (señal forward-looking #1)
- Sesgo −2.24% (sub-pronostica levemente; umbral operativo ±5%)

> ⚠️ Los runs sobre datos sintéticos validan la **metodología**. Con las
> exportaciones reales de sistemas, el mismo notebook corre sin cambios.

## Setup

```bash
python -m venv .venv && .venv\Scripts\activate   # Windows
uv pip install -r requirements.txt              # o pip install -e ".[dev]"
```

Requisito Windows: `cmdstanpy>=1.2,<1.3` (Prophet). TimesFM excluido en
Windows (requiere JAX/Linux) — se reporta su ausencia, nunca su fallback.

## Ejecución

1. Sistemas exporta los 3 agregados → `data/roomnights_real/` (instrucciones
   exactas en [README_SISTEMAS.md](README_SISTEMAS.md))
2. Analítica ejecuta el notebook con el kernel `Hotel Forecast (venv)`
3. Reporte ejecutable: `notebooks/hotel_roomnights_realdata_report.html`

## Estructura

```
src/forecasting/
├── data/            # real_sources (3 queries + loaders), property_map, pickup
│                    # (as-of), aggregation (NULL-safe), synthetic_real (v2)
├── eda/             # EDA orientado a decisiones
├── preprocessing/   # features anti-fuga (shift), splits temporales con gap
├── models/          # Prophet/SARIMAX/ETS · LightGBM/XGBoost/CatBoost · PatchTST
│                    # + tuning.py (objetivo recursivo multi-paso)
├── evaluation/      # métricas + bias · backtesting · conformal · DM Newey-West
└── forecasting/     # recursive (incremental verificado, covariantes por fecha)
tests/               # 73 tests (incl. no-leakage, as-of, determinismo, cold-start)
notebooks/          # 3 pipelines + reportes HTML
README_SISTEMAS.md  # ← contrato de datos con el equipo de sistemas
ASSUMPTIONS.md      # supuestos y limitaciones (leer antes de usar cifras)
CHANGELOG.md        # v0.1 → v0.4
```

## Métricas reportadas

MAE, RMSE, MAPE, sMAPE, WAPE (≡ WMAPE), MASE (vs SeasonalNaive), Bias% y días
sobre/sub-estimados, cobertura empírica de bandas, rango medio en backtesting.
Con benchmarks de la industria (Lighthouse 2025) y su advertencia: miden
horizontes cortos — nuestro protocolo multi-paso es más estricto.

## Licencia

MIT
