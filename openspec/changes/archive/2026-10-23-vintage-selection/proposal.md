# Proposal — vintage-selection

## Contexto

Con la data actualizada (reales al 22-oct) el usuario corrió futura + verificación.
La verificación de las 13:06 y la futura gradúan la añada correcta (4.44% en 18
estancias), pero la verificación de las 14:42 —posterior a una futura con corte
en el último real— reportó vintage `None`: su "corrida futura previa" cubre
estancias desde el 23-oct y aún no tiene reales.

## Problema

La selección del vintage toma SIEMPRE la última corrida de modo `futuro` del
registro. Cuando se publica una futura con el corte en el último real (el flujo
operativo natural tras cada refresh), su horizonte entero es futuro → la añada
gradable es la publicación ANTERIOR, y el tracking se apaga justo cuando
llegaron los reales que debía puntuar. El uso central del feature se rompió en
su primer día de operación real con datos frescos.

## Propuesta

`select_gradable_publication`: itera los registros de modo `futuro` del más
reciente al más antiguo (máx. 5) y gradúa la PRIMERA publicación con estancias
materializadas. La revisión entre corridas sigue comparando contra la última
publicación (semántica de linaje operativo, sin cambio). El reporte y el CSV
declaran qué añada se graduó (`published_run_id`).

## Non-goals

- Track record multi-añada (todas las añadas en una vista): sigue en roadmap;
  esta selección es su prerrequisito.
- Cambiar la semántica de `forecast_vs_previous` (linaje operativo).
