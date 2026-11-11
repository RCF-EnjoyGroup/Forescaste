# monitoring — delta

## MODIFIED Requirements

### Requirement: Estado entre corridas (registro)

#### Scenario: Memoria por línea de corrida (modo) — acceso robusto
- **WHEN** cualquier consumidor lee la ruta del registro (`monitoring.registry_path`)
- **THEN** la lee como mapa (`(cfg.monitoring or {}).get("registry_path")`) — jamás por atributo, que revienta con `AttributeError` contra el `dict` de `src/config.py`
- **THEN** la selección as-of del modo futuro (`run_real --futuro`) encadena al último registro **futuro** con el mismo helper que el runner — un solo camino de lectura del registro para toda la memoria entre corridas
