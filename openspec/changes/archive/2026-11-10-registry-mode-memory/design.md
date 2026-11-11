# Design — registry-mode-memory

## La regla: memoria por línea, no por archivo

`latest_record(registry)` devuelve el último registro AGREGADO. Con dos
líneas de corrida compartiendo el archivo (verificación real y
publicación operativa futura), "el último agregado" es con frecuencia de
la otra línea — y los consumidores con memoria toman su métrica, su
incumbente o su estado de re-entrenamiento de ahí.

Cambio mínimo y explícito: los tres consumidores piden
`latest_record(registry, mode=<run_mode>)` (el filtro ya existe en
`run_registry.latest_record`). Sin registro del mismo modo → `None`:
memoria vacía, declarada en el manifiesto — jamás herencia cruzada.

## Dónde (runner.py)

```python
previous_run = latest_record(registry, mode=str(run_mode))
if previous_run is None:
    previous_run = None
```

Consumidores que pasan a ver su propia línea:
- `incumbent=(previous_run or {}).get("model_by_property")` — histéresis;
- `_monitoring_block(..., previous=previous_run)` — memoria de drift;
- `should_retrain(..., previous=previous_run)` — días desde la última
  corrida de la línea.

`previous_operational = latest_record(registry, mode="futuro")` y
`select_gradable_publication` (filtran `mode="futuro"`) NO cambian: ya
son ejemplos del patrón correcto.

## Modo "ad hoc"

Las corridas sin modo real/futuro (por defecto `"ad hoc"`) aíslan su
propia línea: `latest_record(mode="ad hoc")` — cero interferencia con
las líneas productivas y comportamiento determinista en tests.

## Evidencia histórica (replay, no re-corrida)

`scripts/ab_registry_memory.py` recorre el registro real y para cada
corrida imprime el `previous_run` bajo la regla vieja (cualquier modo,
último agregado) vs la nueva (mismo modo):

- `real-20261105-1442` (brazo B del A/B de bias): old →
  `futuro-20261104-1525` (MARINA=sarimax — el incumbente que contaminó
  el A/B) vs new → `real-20261104-1514` (MARINA=lightgbm, comparable).
- resto de corridas: igual o mejor alineado (mismo modo).

## Riesgos

- **Histéresis más corta**: una línea que altera modos (real, futuro,
  real…) ya no encadena histéresis a través del otro modo. Es el punto:
  cada línea arrastra su propio estado; el `property_fallback` del
  manifiesto lo hace visible.
- **Retrocompatibilidad**: con registro vacío o de un solo modo, la
  regla nueva es idéntica a la vieja (tests de regresión).
