## Why

El error a largo plazo (60–90 leads: 38–40% MAPE) persiste porque la curva estacional solo distingue alta/media/baja: un stay de Semana Santa (marzo–abril, movible) se promedia con un marzo cualquiera, y Navidad con un enero cualquiera. Los feriados — costarricenses (Semana Santa, Anexión, Día de la Madre, Independencia) y de los mercados emisores (Thanksgiving, Memorial, Labor Day, 4 de julio) — generan picos de reserva que un modelo de calendario fino puede anticipar a CUALQUIER lead, porque son conocidos por adelantado (sin leakage por construcción).

## What Changes

- **Calendario de feriados como fuente confiable** (`src/features/holidays.py`): fechas CR + US vía la librería `holidays` (maneja movibles: Semana Santa por computus, Thanksgiving 4.º jueves, Memorial último lunes), con agrupación semántica (semana_santa, fin_de_ano, thanksgiving…) y ventanas de viaje ±`window_days` configurables.
- **Curva de pickup condicionada por ventana de feriado**: un nivel más de especificidad en la cadena de lookup ya validada — (propiedad, feriado) → (propiedad, temporada) → propiedad → global — activado por `pickup_curve.holiday_conditional` y sujeto al mismo umbral `season_min_lead` y a `min_stays_per_bucket` (feriados con poca evidencia degradan sin daño).
- **Features de feriado para los candidatos ML**: `days_to_nearest_holiday` en la matriz GBM (el bake-off queda parejo: si el campeón gana señal de feriados, los GBM también) y el indicador `is_holiday` en el frame de features de la corrida (spec feature-engineering: "indicador de feriado cuando exista información confiable").
- **Defaults por A/B**: ventana de verificación + backtest dirigido a las ventanas de feriado (Semana Santa/Navidad/Thanksgiving recientes) deciden si `holiday_conditional` arranca activo; sin evidencia de ganancia queda documentado y desactivado.

## Capabilities

### New Capabilities

<!-- Ninguna. -->

### Modified Capabilities

- `feature-engineering`: MODIFIED — Features de calendario y estacionalidad: el indicador de feriado proviene de una fuente confiable configurable (países + ventana) y ya no se omite; MODIFIED — Curvas de pickup no lineales: la cadena de condicionamiento gana el nivel feriado con el mismo fallback controlado.

## Impact

- **Código**: nuevo `src/features/holidays.py`; `src/models/baselines.py` (nivel feriado); `src/forecasting/core.py` (calendario en fit/predict ML); `src/forecasting/runner.py` y `run_real.py` (wiring); `src/config.py` + `config.yaml` (sección `holidays`); `requirements.txt` (+`holidays`).
- **Comportamiento observable**: `is_holiday` presente en el reporte de features; los stays dentro de ventanas de feriado pueden recibir curva propia; el manifiesto registra la configuración de feriados.
- **Riesgo**: fragmentación de buckets por feriado → mitigado por `min_stays_per_bucket` con fallback; efecto mínimo en la ventana de verificación (jul–oct tiene feriados medianos) → A/B dirigido a las ventanas grandes como evidencia complementaria.
