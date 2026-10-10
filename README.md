# Enterprise Room Nights Forecasting Engine — Enjoy Costa Rica

Motor de pronóstico de **room nights** (RN) para el portafolio hotelero de Enjoy
Costa Rica. Toma las tres fuentes reales (demanda realizada, reservas en libros
por anticipación y capacidad oficial), valida sin fugas, entrena/cronometra
candidatos, pronostica el horizonte configurado y publica **10 entregables**
listos para Power BI, todo reproducible desde una semilla.

---

## Requisitos

- Python **≥ 3.11** (verificado con 3.11)
- Instalación de dependencias:

```bash
pip install -r requirements.txt
```

---

## Datos

Las tres fuentes reales viven en `data/` (rutas configurables en
`config.yaml → data.sources`):

```
data/
├── demand_daily.csv     # demanda realizada  → property, stay_date, rooms_sold, room_revenue
├── pickup_by_lead.csv   # reservas en libros → property, stay_date, lead_days, rooms_booked
└── capacity_daily.csv   # capacidad oficial   → property, business_date, available_rooms
```

El contrato exacto de exportación (queries SQL para Sistemas, formato,
cadencia y checklist de verificación) está en
[README_SISTEMAS.md](README_SISTEMAS.md). `data/` nunca se commitea
(.gitignore).

---

## Uso

### 1. Corrida de verificación (default)

```bash
python -m src.forecasting.run_real
```

Fija el corte en `último real − horizonte` (90d) y pronostica **esa ventana
que ya tiene reales observados**: así MAPE/WMAPE se miden de verdad contra lo
que ocurrió. Es control de calidad del motor, **no** un pronóstico operativo —
su ventana termina en el último real. El campeón del benchmark (elegido con
folds trimestrales por propiedad, sin solape con la ventana de reporte)
**opera el forecast** — global y por propiedad cuando la evidencia lo
justifica — y el backtest T-90…T-1 muestreado alimenta los residuos por
banda de lead de P10/P50/P90 y el entregable `backtesting_report.json`.
Sale con código `0` solo si los 5 criterios de aceptación pasan.

### 2. Corrida futura (pronóstico operativo)

```bash
python -m src.forecasting.run_real --futuro                  # hasta 31-dic del último año con reales
python -m src.forecasting.run_real --futuro --hasta 2027-03-31
```

**Este es el modo que usa el negocio**: corte = **último real** y horizonte
hasta `--hasta`. Entrena con todo el histórico disponible. La ventana futura
todavía no tiene reales, así que métricas, desgloses y banda provienen de un
**backtest real sobre la ventana anterior** al corte; el manifiesto lo
etiqueta `metrics_source: injected_backtest` y la documentación técnica lo
declara. Jamás se inventan reales futuros.

### Parámetros por entorno (sin tocar código)

| Variable | Efecto |
|---|---|
| `FORECAST_HORIZON` | horizonte en días (default 90) |
| `RETRAIN_FREQUENCY` | política de re-entrenamiento (default monthly) |
| `MIN_HISTORY_DAYS` | histórico mínimo para re-entrenar (default 365) |
| `RANDOM_SEED` | semilla (default 42) |

---

## Salida

Cada corrida esquiva `outputs/runs/<real|futuro>-<YYYYMMDD-HHMM>/` con los
**10 entregables** + manifiesto:

```
outputs/runs/futuro-20261009-2027/
├── data_quality_report.json
├── eda_report.json
├── feature_engineering_report.json
├── model_benchmark.md
├── backtesting_report.json
├── forecast_accuracy_report.json
├── future_forecast_dataset.csv      # ← el pronóstico (P10/P50/P90, ≤ capacidad)
├── model_explainability_report.csv
├── powerbi/                         # ← datasets por área para Power BI
│   ├── forecast.csv · accuracy.csv · pickup.csv · model.csv
├── technical_documentation.md
├── run_manifest.json                # ← semilla, períodos, fingerprint (reproducible)
└── report.html                      # ← vista de revisión con gráficos embebidos
```

### `report.html` — vista de revisión con gráficos

Además de los 10 entregables, cada corrida genera un **reporte HTML
autocontenido** (`report.html`, se abre en el navegador sin instalar nada ni
conexión) con las gráficas de revisión embebidas en base64:

- **Forecast** · serie diaria del portafolio con banda P10/P50/P90
- **Forecast vs Real** · superposición (solo en corridas de verificación, donde hay reales)
- **Benchmark** · WMAPE por modelo, campeón destacado
- **Pickup** · RN ya reservados por banda de anticipación
- **Ocupación** · forecast contra el techo de capacidad
- **EDA** · volumen diario de room nights

Es una **vista**: la lógica vive en `src/` (`reporting/visual_report.py`) y el
reporte solo importa y presenta (spec reporting: los notebooks/reportes no
llevan lógica). No cuenta entre los 10 entregables y nunca rompe la corrida si
faltan datos — las secciones ausentes se omiten con un mensaje. Los KPIs
ejecutivos aparecen como tarjetas arriba.

```bash
# abrir el último reporte (Windows)
start outputs/runs/futuro-20261010-0754/report.html
```

---

## Estructura

```
src/
├── config.py            # config.yaml + overrides de entorno
├── data/                # ingesta, detección de esquema, pipeline
├── validation/          # calidad de datos, gate, suite de leakage (as-of)
├── features/            # calendario, demanda, pickup, capacidad, as-of
├── models/              # baselines + 7 candidatos (SARIMAX, LightGBM, XGBoost, CatBoost, ETS…)
├── backtesting/         # snapshots y backtest expanding window
├── evaluation/          # métricas, ventanas, desgloses, selección de campeón, multi-paso
├── forecasting/         # runner end-to-end, restricciones, CLI run_real.py
├── explainability/      # importancia de features + SHAP
├── monitoring/          # drift, outliers, decisión de re-entrenamiento, manifiesto
└── reporting/           # benchmark, tabla de forecast, KPIs, dataset Power BI, entregables
tests/                   # 197 tests
openspec/                # specs, propuestas y changelog del proyecto
```

---

## Garantías del pipeline

- **Sin fugas (leakage)**: todo se calcula as-of; la suite `validation/leakage`
  corre COMPLETA en la corrida productiva (capacidad incluida en el perímetro)
  y falla la corrida si algún feature lee el futuro o incluye el target de la
  propia fila en su ventana.
- **El campeón opera**: el forecast lo genera el modelo ganador del bake-off
  (global y por propiedad); benchmark, manifiesto, tabla de salida y Power BI
  declaran el mismo campeón.
- **Gate de calidad con dientes**: un status `fail` del Data Quality Report
  detiene la corrida antes de entrenar y no publica forecast.
- **Sin doble conteo**: la demanda pasa por filtros (`snap_flag`, ratecodes de
  ajuste) definidos en el contrato de datos.
- **Propiedades cerradas excluidas**: `data.excluded_properties` en
  `config.yaml` (hoy LIRAK y LAPAS) las saca de toda la corrida; el manifiesto
  y el accuracy report declaran la exclusión.
- **Curva de pickup estacional**: a leads ≥ `pickup_curve.season_min_lead`
  (30) la curva se condiciona por la temporada de la fecha de estancia — el
  forecast a largo plazo converge al nivel estacional del stay; a leads
  cortos se preserva el pooling por propiedad. Defaults validados con A/B.
- **Feriados CR + US como señal de largo plazo**: calendario confiable
  (librería `holidays`: Semana Santa movible, Thanksgiving, Memorial, Día de
  la Madre…) con ventanas de viaje ±`window_days`; la curva de pickup gana un
  nivel de condicionamiento por ventana de feriado (con umbral propio y
  fallback) y los candidatos ML reciben `days_to_nearest_holiday`.
- **`final_forecast ≤ capacity`** en el 100% de los registros, **y también
  `p90 ≤ capacity`**: la banda de incertidumbre jamás publica valores
  físicamente imposibles.
- **Evaluación honesta**: folds de selección trimestrales (cubren estaciones
  distintas, sin solape con la ventana de reporte), prueba Diebold-Mariano
  del campeón contra el subcampeón (empate → parsimonia), cobertura de
  evaluación reportada (propiedades dentro/fuera de la ventana) y desglose
  por propiedad persistido en el accuracy report y Power BI.
- **Reproducible**: misma seed → mismas predicciones, verificado por test.

---

## Tests

```bash
python -m pytest -q          # 223 tests
```

Cobren la cadena completa (e2e), esquema alternativo, as-of/leakage,
restricción de capacidad, calidad de datos, manifiesto reproducible y el modo
`--futuro`. La suite de remediación (`tests/test_remediation.py`) blinda los
hallazgos del diagnóstico: campeón operativo, coherencia entre entregables,
gate de calidad con dientes, lags calendario, rollout ML sin off-by-one,
significancia Diebold-Mariano, bandas ancladas/capadas y cobertura de
evaluación.

---

## Documentación

- [README_SISTEMAS.md](README_SISTEMAS.md) — contrato SQL de exportación para
  el equipo de Sistemas (las 3 queries de los datos reales).
- `openspec/` — specs por capacidad, propuestas y changelog.
- `outputs/runs/<corrida>/technical_documentation.md` — documentación técnica
  de cada corrida (incluye limitaciones y origen de las métricas).
