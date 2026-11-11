# Proposal — fix-futuro-registry-access

## Contexto (crash en producción)

`python -m src.forecasting.run_real --futuro` (corrida del 10-nov) falla:

```
File "src\forecasting\run_real.py", line 494, in main
    registro = load_registry(cfg.monitoring.registry_path)
AttributeError: 'dict' object has no attribute 'registry_path'
```

El modo real funciona (usa el acceso correcto
`(cfg.monitoring or {}).get("registry_path")` en la línea 539) — por eso
las corridas de verificación del 10-nov salieron bien y el futuro no.

## Causa raíz

Commit `c255eb4` ("la selección as-of del modo futuro aplica la
histéresis del registro") introdujo el bloque de histéresis del modo
futuro con acceso por ATRIBUTO sobre `cfg.monitoring`, que es un
`dict` (fuente: `src/config.py` — `monitoring` se exponе como mapa).
El bloque quedó fuera del `try` de la corrida y además filtraba
`latest_record(registro)` sin modo.

## Propuesta

1. **Acceso correcto**: helper `_incumbente_previo(cfg, mode="futuro")`
   — ruta desde `(cfg.monitoring or {}).get("registry_path")`, consistent
   con las líneas 539 y con `runner.py`.
2. **Consistencia con la memoria por línea** (openspec
   2026-11-10-registry-mode-memory): la selección as-of del modo futuro
   encadena al último registro **futuro** (`mode="futuro"`), igual que el
   runner — no al último registro de cualquier modo.
3. El helper es testeable de forma directa (dict access, path ausente,
   registro ausente) — el test falla con AttributeError contra el código
   viejo, así queda clavada la regresión.

## Validación

- Tests nuevos sobre `_incumbente_previo` (config con monitoring dict,
  sin ruta, registro vacío, filtro de modo).
- Suite completa en verde.
- Re-corrida `--futuro` de verificación: llega a publicar sin crashear.

## Non-goals

- No cambiar la selección as-of ni el runner.
