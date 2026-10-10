## Why

La palanca de feriados ya opera (commit `b8b6fd3`: WMAPE 60–89 0.2319→0.2265 en ventana, 0.2075→0.2001 en backtest dirigido). Lo que queda del espacio "feriados/eventos" son los **eventos privados** — conciertos, congresos, torneos, bodas grandes — que generan picos de reserva invisibles para cualquier calendario público y que solo el negocio conoce por adelantado (Revenue/Sistemas). La maquinaria de ventanas ya validada los absorbe sin cambios de arquitectura: un evento es una ventana más con clave propia; la validación de precisión llegará cuando Sistemas entregue el feed, así que este cambio entrega contrato + mecanismo + inertes-sin-datos (cero riesgo).

## What Changes

- **Fuente opcional de eventos** (`data/events.csv`, ruta configurable en `data.events_path`): columnas `event_name, event_date, property` (property opcional/blank = portafolio). Contrato de exportación documentado en `README_SISTEMAS.md`. Sin archivo → el feature se omite y la corrida lo declara (igual que los feriados).
- **Eventos en el calendario compartido**: las fechas de eventos entran a `HolidayCalendar.dates` (la feature ML `days_to_nearest_holiday` los cubre) y sus ventanas ±`window_days` entran al mapa `windows` con clave propia (slug del nombre). `is_holiday` queda reservado a feriados reales (`holiday_dates`).
- **Scoping por propiedad**: eventos con `property` asignada generan ventanas SOLO para esa propiedad (`window_key(date, property)`); un concierto en CORIN no contamina la curva de FIESTA.
- **Nivel de evento en la curva del campeón**: la cadena existente gana las claves de evento sin cambios de modelo — (propiedad, feriado/evento) → temporada → propiedad → global — con el umbral propio `min_stays_per_holiday_bucket` y fallback.
- **Manifiesto**: declara la fuente de eventos (ausente, o activa con nº de eventos/claves) y la e2e lo propaga a benchmark, forecast y backtest.

## Capabilities

### New Capabilities

<!-- Ninguna. -->

### Modified Capabilities

- `data-ingestion`: ADDED — Fuente opcional de eventos: archivo configurable validado (fechas legibles, duplicados por nombre-fecha), ausente → omitido y documentado.
- `feature-engineering`: MODIFIED — Features de calendario: los eventos privados entran a la feature de distancia y a `window_key` con scoping por propiedad; MODIFIED — Curvas de pickup: las ventanas de evento usan el mismo condicionamiento/fallback.

## Impact

- **Código**: nuevo `src/features/events.py`; `src/features/holidays.py` (`holiday_dates`, `scoped`, `window_key(date, property)`, `calendar_with_events`); `src/models/baselines.py` (fit/lookup con scoping); `src/config.py` + `config.yaml` (`data.events_path`); `runner.py`/`run_real.py` (wiring + manifiesto); `README_SISTEMAS.md` (contrato).
- **Comportamiento observable**: sin `events.csv` las corridas son IDÉNTICAS a las actuales (inertes-sin-datos); con eventos, las estancias en sus ventanas pueden recibir curva propia.
- **Riesgo**: eventos duplicados o con typos de fecha → validación dura en la carga; sin feed real la validación de precisión queda pendiente (documentado — la ganancia esperada es la misma que demostraron los feriados donde el bucket activa).
