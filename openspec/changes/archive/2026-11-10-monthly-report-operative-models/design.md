# Design — monthly-report-operative-models

## Selección por corte (anclaje histórico)

`_modelos_por_corte(records, corte) -> (modelos, fuente)`:

- Candidatos: registros `mode == "futuro"` con `model_by_property` no
  vacío y `cutoff <= corte`.
- Si hay candidatos: el de máximo `(cutoff, timestamp)` — "la selección
  operativa vigente al corte". Fuente: `"registro:<run_id>"`.
- Si no hay candidatos (cortes ene→jul, previos al motor): el registro
  futuro más antiguo disponible — la selección más cercana en el tiempo.
  Fuente: `"registro-mas-antiguo:<run_id>"` (declarada).
- Sin registro legible (archivo ausente/vacío): el snapshot histórico
  `MODELOS_FALLBACK`. Fuente: `"fallback-fijo"` (declarada).

Todas las fuentes viajan al sello de versión — ninguna selección es
silenciosa.

## Sello de versión (hoja "lectura")

- `generado`: timestamp local de la corrida del script.
- `data hasta`: último real de la data cargada (sustituye el texto
  muerto "reales al 25-oct").
- `regla`: "selección operativa por corte desde el registro entre
  corridas (memoria por línea, openspec 2026-11-10-registry-mode-memory)".
- `modelos por propiedad`: el mapa del corte más reciente + cuántos
  mapas distintos hubo entre los cortes del archivo (si es 1, todos los
  cortes compartieron selección; si >1, cada corte usó la suya).

## Cambios en el script

- `MODELOS` → `MODELOS_FALLBACK` (snapshot histórico, solo sin registro).
- `main()`: cargar el registro (`src.monitoring.run_registry.load_registry`
  con la ruta de `config.monitoring.registry_path`), y por corte resolver
  `(modelos, fuente)`.
- Lectura: línea de "reales al" calculada + sección de sello.

## Retrocompatibilidad

Con el registro presente, los cortes existentes (15-ene…15-oct)
reproducen los modelos de época → pronósticos bit a bit. Octubre
(cerrado ya) pasa de parcial al cierre real — único cambio de valor
esperado, consistente con la regla del mes en curso.
