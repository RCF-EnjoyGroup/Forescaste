# forecasting — delta

## MODIFIED Requirements

### Requirement: Intervalos de incertidumbre
#### Scenario: Calibración estacional de la banda (knob, default OFF)
- **WHEN** `bias_correction.seasonal_cells` está activo
- **THEN** la amplitud de la banda y la corrección de bias del punto se estiman por (propiedad, banda de lead, temporada del stay), usando la MISMA definición de temporada publicada en el desglose de estacionalidad (high/shoulder/low) — dimensión conocible a-priori por calendario, sin lookahead
- **THEN** una celda estacional sin evidencia suficiente (n < min_obs o sesgo inconsistente entre folds) cae a la celda (propiedad, banda) o, en su defecto, a la escalera documentada — cada fallback queda registrado
- **THEN** con `seasonal_cells: false` (default) el sistema calibra exactamente como antes (llaves (propiedad, banda)) y las bandas son bit a bit idénticas
- **THEN** el default OFF queda fijado por decisión A/B documentada (2026-10-30): activado, la cobertura del horizonte EMPEORÓ (FIESTA 42.2%→23.3%, CORIN −12.2pp) porque el sesgo del horizonte era drift de nivel YoY — no bias estacional recurrente — y la calibración del año previo corrigió en dirección contraria; la granularidad más fina NO puede corregir un régimen nuevo

### Requirement: Corrección de bias del punto
#### Scenario: Sesgo estacional con gates
- **WHEN** `bias_correction.seasonal_cells` está activo
- **THEN** el factor de corrección por (propiedad, banda, temporada) aplica los mismos gates de siempre (min_obs, consistencia de signo entre folds, shrinkage, clip ±30%)
- **THEN** la fuente aplicada de cada factor (`season`/`band`/`none`) viaja en el reporte del manifiesto — ninguna corrección silenciosa
- **THEN** las combinaciones estacionales sin filas en los folds heredan el factor de banda documentado, sin duplicar correcciones sobre celdas ya decididas
