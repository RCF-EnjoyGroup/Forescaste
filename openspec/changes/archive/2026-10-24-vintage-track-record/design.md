# Design — vintage-track-record

## La identidad de una añada

El registro guarda una graduación POR CORRIDA — muchas corridas gradúan la
misma publicación. El expediente necesita UNA fila por AÑADA (publicación):
- identidad explícita: `vintage.published_run_id` (corridas desde el fix de
  selección), con
- fallback por ventana: la publicación que cubre `window.start` tiene corte
  `window.start − 1d` → el registro `futuro` con ese cutoff (permite
  reconstruir las graduaciones previas al campo, incluida la añada 1).

Cada añada queda con su graduación MÁS RECIENTE — la que más reales
acumuló (añada 1: 36 obs en su primera graduación → 78 en la definitiva).

## La tendencia, con honestidad de n

Con < 2 añadas: «Acumulando» (info, no opina). Con ≥ 2: la última contra el
promedio de las previas (mejora ≤ −15%, estable ±15%, revisar > +15%) —
**con el n de la última añada explícito**: una añada con 3 días
materializados (n=18) no pesa igual que una con 13 (n=78), y el criterio
lo dice en voz alta. La gráfica etiqueta cada barra con su n por la misma
razón.

## As-of en el regenerador

`regen_report.py` filtra los registros del expediente a timestamp ≤ la
corrida regenerada: el reporte de una corrida del 21-oct no cita añadas
graduadas el 23 — los reportes viejos no mienten con datos del futuro.
