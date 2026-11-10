# forecasting — delta

## MODIFIED Requirements

### Requirement: Intervalos de incertidumbre

#### Scenario: Calibración de banda por propiedad bajo knob (default OFF)
- **WHEN** `monitoring.interval_calibration.enabled` está activo
- **THEN** cada banda publicada (propiedad, semiancho P10–P90) se calibra con un factor multiplicativo aprendido entre corridas (registro: cobertura observada vs nominal), con shrink hacia 1.0 por pocas observaciones y clip [0.6, 1.5]
- **THEN** el factor por propiedad y la cobertura observada viajan en el manifiesto; `config.yaml` actúa como override del valor aprendido
- **THEN** con `enabled: false` (default) la banda es idéntica a la construcción conformal actual
- **THEN** el default OFF quedó fijado por el A/B de replay del 2026-11-09 (4 corridas reales, brazos OFF/shrink 10/30/90): llevар cada propiedad al 80% nominal EMPEORA la cobertura global (78.1% control vs 74.2–77.6%) porque dos propiedades ya sobre-cubren 93–96% — el target correcto sería global o con costo asimétrico, y queda anotado como revisita futura; el aprendizaje del factor k quedó validado contra medición independiente
