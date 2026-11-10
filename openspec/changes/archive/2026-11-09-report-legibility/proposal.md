# Proposal — report-legibility

## Contexto

Dos hechos del 6-nov:

1. **Las corridas del 6-nov publicaron su reporte SIN la comparativa
   real-vs-forecast** ("Forecast vs Real" y "Dispersión" desaparecieron).
   La forense descartó data y motor: los reales están completos
   (539/540 estancias) y regenerar el reporte con el código actual
   produce ambas tarjetas. Las corridas se generaron con un estado
   intermedio del archivo del reporte (había ediciones sin commitear en
   `visual_report.py` esa misma mañana) — pero el defecto de fondo es
   real: **las tarjetas dependientes de reales se omiten en silencio**
   cuando no hay `actual`, incluso en corridas de verificación que SÍ
   tienen reales. Eso viola la regla del proyecto: nada silencioso.
2. Las tablas del reporte muestran métricas en formato crudo de
   ingeniería (WMAPE como 0.177, "nan%" cuando no hay reales) — difícil
   de leer para negocio.

## Propuesta

1. **Legibilidad** (mejoras ya validadas en el working copy, con sus
   tests): WMAPE/sesgo como porcentaje, MAPE con %, conteos (n_obs)
   sin decimales, NaN → "—" (coherente con el N/D del panel), nota de
   auditoría explicando la escala de cada métrica.
2. **Blindaje anti-silencio**: cuando una corrida de verificación tiene
   reales en su ventana evaluada pero la comparativa no puede
   dibujarse, la tarjeta se renderiza con el MOTIVO explícito (p. ej.
   "la unión de reales con el horizonte publicado no encontró
   coincidencias — revisar alineación de claves"). Solo cuando
   genuinamente no hay reales (corrida futura: el horizonte aún no
   existe) la tarjeta se omite, que es el comportamiento documented.
3. El runner calcula el motivo: reales existen en `eval_src`
   (`n_real > 0`) pero la unión con el horizonte publicado no produjo
   coincidencias → nota; sin reales en la ventana → sin nota (futuro).

## Validación

- Tests nuevos: (a) con reales + unión vacía → la tarjeta declara el
  motivo; (b) sin reales → la tarjeta se omite (comportamiento
  histórico intacto); (c) las mejoras de formato ya cubiertas por los
  tests del working copy (`_fmt_metrics`, `_kpi_grid`, accuracy semanal).
- Suite completa en verde.

## Non-goals

- No rediseñar el reporte ni tocar los gráficos existentes.
- No cambiar la selección ni el contenido de las secciones.
