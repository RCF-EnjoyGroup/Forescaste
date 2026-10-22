# forecasting — delta

## MODIFIED Requirements

### Requirement: Intervalos de incertidumbre
#### Scenario: Cuantiles ponderados por recencia (opcional)
- **WHEN** `monitoring.interval_recency_decay < 1.0`
- **THEN** los residuos de folds recientes dominan los cuantiles de la banda (peso `decay^(edad del fold)`); los pools de la escalera preservan los pesos; Series sin peso siguen válidas (back-compat)
- **WHEN** el valor es 1.0 (default; A/B del 22-oct: la recencia no superó al baseline en desviación media de cobertura — resultado negativo documentado)
- **THEN** la ponderación es uniforme (comportamiento clásico)
