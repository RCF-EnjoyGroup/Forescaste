# Proposal — vintage-track-record

## Contexto

El registro acumula 18 corridas con 6 añadas publicadas y 2 graduadas, y ya
guarda por corrida el vintage con la TS por propiedad. Lo que falta es la
vista que convierte ese expediente en decisión de negocio: **el forecast
publicado, ¿está mejorando añada sobre añada?** — la pregunta que un dueño
hace y que hoy no se puede responder en una sola vista.

## Problema

El vintage del reporte muestra UNA añada (la graduable más reciente). El
historial vive en el registro pero nadie lo presenta: cada corrida que
graduó la misma añada duplica el dato, y las añadas viejas (con menos días
materializados al momento de su graduación) deben representarse con su
graduación MÁS RECIENTE (la que más reales acumuló).

## Propuesta

- **`build_track_record(registry_records)`**: reconstruye el expediente
  desde el registro — deduplicado por añada (`published_run_id`, con
  atribución por `cutoff = inicio de la ventana − 1d` para graduaciones
  previas a ese campo), cada añada con su graduación más reciente:
  publicación (cutoff, timestamp), ventana graduada, n, WMAPE, bias,
  cobertura de banda, TS peor por propiedad.
- **Reporte**: nueva tarjeta «Expediente del forecast publicado» (grupo
  Precisión) — tabla por añada + gráfica de WMAPE por añada con su n; fila
  en el panel de validación con la tendencia (info «acumulando» con < 2
  añadas; luego mejora/estable/revisar comparando la última contra las
  previas, con la salvedad del n explícita).
- **Regenerador**: reconstruye el expediente as-of la corrida que
  regenera (solo registros con timestamp ≤ esa corrida) — los reportes
  viejos no mienten con datos del futuro.
- El manifiesto lleva un resumen del expediente (n añadas, última WMAPE)
  para trazabilidad.

## Non-goals

- Pronóstico de la precisión futura (meta-forecast): fuera de alcance.
- Track record por propiedad (matriz añada × hotel): cuando haya ≥ 4
  añadas con n suficiente; la estructura ya lo permite.
