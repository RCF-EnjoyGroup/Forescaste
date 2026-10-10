# Design — add-event-features

## Context

La maquinaria de ventanas (feriados) ya opera y validó la tesis: condicionar la curva por ventanas de fecha mejora el largo plazo donde el bucket activa. Los eventos privados son el mismo problema con otra fuente de datos — el valor restante del espacio "feriados/eventos" es el mecanismo + el contrato, porque el feed real vive en Sistemas.

## Goals / Non-Goals

**Goals** — Contrato de exportación de eventos (README_SISTEMAS); carga validada con fallo duro; eventos absorbidos por la maquinaria existente (calendario compartido, ventanas, condicionamiento de curva, feature de distancia ML) con scoping opcional por propiedad; inertes-sin-datos verificado.

**Non-Goals** — Validación de precisión con eventos reales (bloqueada por el feed: cualquier A/B sintético sería circular); estimación de uplift de eventos; descubrimiento automático de eventos desde la serie (detección no supervisada — ruido).

## Decisions

### D1 — Un evento es una ventana con clave propia
`calendar_with_events(base, events)` produce un calendario combinado: las fechas de evento entran a `dates` (la feature ML `days_to_nearest_holiday` pasa a cubrir "feriado o evento más cercano" — el nombre queda, la semántica se documenta) y sus ventanas ±`window_days` al mapa `windows` con clave = slug del nombre de evento. Sin niveles nuevos en la curva: el lookup (propiedad, clave) que ya existe los consume. Alternativa descartada: modelo separado de eventos — duplicaría la maquinaria por cero ganancia.

### D2 — Scoping por propiedad en el calendario, no en el modelo
`HolidayCalendar` gana `scoped: dict[prop, Series]` para eventos con propiedad asignada; `window_key(date, property=None)` consulta primero el mapa global y luego el scoped de la propiedad. El modelo pasa su `prop` al asignar buckets (fit) y al predecir — un concierto en CORIN nunca contamina la curva de FIESTA. Alternativa descartada: prefijar la propiedad en la clave y filtrar en el modelo — contamina el mapa global con claves que aplican a quien no deben.

### D3 — `is_holiday` puro, distancia compartida
`holiday_dates` (solo feriados) alimenta el indicador `is_holiday` del frame de features; `dates` (feriados + eventos) alimenta `days_to_nearest_holiday`. Un día de congreso no debe reportarse como "feriado" en los reportes.

### D4 — Validación dura en la carga, inertes-sin-datos garantizado
`load_events(path)` falla con `ValueError` si hay fechas ilegibles o duplicados (nombre+fecha+propiedad) — un evento con fecha corrupta degradaría ventanas silenciosamente. Sin archivo → `None` → el runner construye el calendario de solo feriados y el manifiesto declara `events: {"source": null}`; con archivo → `{"source": path, "n_events": n, "n_scoped": k}`. La prueba de inertes: corrida sin `events.csv` produce métricas idénticas a las del cambio anterior.

### D5 — El A/B de eventos queda explícitamente diferido
Sin feed real no hay A/B honesto. El mecanismo se valida con tests unitarios (bucket de evento, scoping, fallback) y la corrida e2e con un archivo sintético declara la fuente activa. La validación de precisión se ejecuta cuando Sistemas entregue el feed — el protocolo ya existe (mismo A/B de feriados, sustituyendo la fuente).

## Risks / Trade-offs

- [Eventos con typos o fechas pasadas rancias] → validación dura; los eventos muy viejos caen fuera de la ventana de recencia de la curva y no generan bucket.
- [Ventanas de evento solapando feriados] → una sola clave por fecha (la más cercana); el evento más cercano gana — correcto: es la señal más específica.
- [Feed tardío en llegar] → el mecanismo queda documentado y probado; mientras tanto el sistema no pierde nada (idéntico a hoy).

## Migration Plan

1. `events.py` + extensión de `HolidayCalendar` → modelo con scoping → config/wiring/contrato → tests → corrida sin archivo (idéntica) → archive. Rollback: borrar/comentar `data.events_path`.
