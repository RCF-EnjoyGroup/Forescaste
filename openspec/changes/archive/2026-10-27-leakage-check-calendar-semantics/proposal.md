# Proposal — leakage-check-calendar-semantics

## Contexto

La corrida del 27-oct (data nueva, reales al 26-oct) quedó DETENIDA por el
gate de leakage: `rolling_includes_future` en MARINA 2025-05-09.

## Problema

Diagnóstico con los valores exactos: el nuevo export introdujo un hueco real
en MARINA (2025-05-06). La feature `rolling_mean_7` es **calendario** (spec
feature-engineering/remediación: serie reindexada a grilla diaria; huecos =
NaN que la ventana salta con min_periods=1) — el valor almacenado 7.000 es
EXACTO para la ventana calendario hacia atrás (05-02..05-08 sin el 06:
(10+9+3+3+10+7)/6 = 7.000 ✓, sin la fila propia, sin futuro).

El check `check_rolling_with_future` recalcula **por entradas** (sin
reindexar a grilla): con el hueco, su "ventana centrada por-entradas" dio
7.000 por coincidencia numérica ((3+10+7+9+8+3+9)/7) → falso positivo que
detiene la corrida. El check y la feature hablaban semánticas distintas: la
comparación solo es válida sin huecos.

## Propuesta

El check recalcula sobre la **misma grilla calendario** que la feature
(reindex diario, deduplicación defensiva, comparación solo en fechas
observadas). La detección de fuga real se mantiene: si el almacenado
coincide con la ventana centrada EN GRILLA y no con la retrospectiva EN
GRILLA, sigue disparando.

## Non-goals

- Cambiar la semántica de las features (calendario está en spec y
  documentada desde la remediación).
- Suavizar el gate: sigue con dientes; solo deja de disparar por
  coincidencias numéricas entre semánticas distintas.
