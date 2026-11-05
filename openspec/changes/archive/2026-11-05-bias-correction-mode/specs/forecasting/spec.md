# forecasting — delta

## MODIFIED Requirements

### Requirement: Corrección de bias del punto por (propiedad, banda)

#### Scenario: Capa apagable (kill-switch)
- **WHEN** `monitoring.bias_correction.enabled: false`
- **THEN** no se estiman ni aplican factores: el punto publicado es el del modelo sin corrección y las bandas calibran ese pipeline
- **THEN** el manifiesto publica `bias_correction: {enabled: false}` — la capa jamás se desactiva en silencio
- **THEN** con `enabled: true` (default) el sistema se comporta exactamente como antes

#### Scenario: Evidencia ponderada por recencia de folds
- **WHEN** `monitoring.bias_correction.fold_recency_decay < 1.0`
- **THEN** cada fold pesa `decay^(edad del fold)` en el bias relativo de la celda y en la votación de consistencia de signo — el régimen reciente domina sin anular la evidencia vieja
- **THEN** `min_obs` sigue contando observaciones crudas; shrinkage y clip no cambian
- **THEN** con `fold_recency_decay: 1.0` (default) el estimador es bit a bit el actual
- **THEN** el default de ambos knobs queda fijado por el A/B de 3 brazos del 2026-11-05 (capa ON uniforme vs OFF vs ON con recencia) sobre la misma data, documentado en config
- **THEN** el A/B del 2026-11-05 decidió MANTENER los defaults (capa ON, folds uniformes): apagar la capa empeora SJOSL (sesgo +17.1%→+21.8%, wmape 43.6→45.4) — en SJOSL la capa ayuda; la mejora de MARINA sin capa quedó confundida por un cambio de modelo de la histéresis del registro; la variante recencia 0.7 no dañó a nadie pero no cumplió la mejora estricta y queda como knob documentado
