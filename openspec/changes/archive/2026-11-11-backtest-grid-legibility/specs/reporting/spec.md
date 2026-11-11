# reporting — delta

## ADDED Requirements

### Requirement: Backtest de snapshots diagnóstico (T-90..T-1 por propiedad)

#### Scenario: El n desigual se explica en el propio reporte
- **WHEN** el backtest muestreado emite `backtesting_report.json` con `n_obs` desigual entre propiedades (56 baratos vs 16 costosos, guard de costo del campeón que opera)
- **THEN** el reporte incluye `grid_note` explicando de dónde sale el n, que el propósito es diagnóstico (cada propiedad contra sí misma entre corridas, métricas relativas) y que la selección del campeón y las bandas P10/P90 usan la evidencia de rejilla común del benchmark de folds
- **THEN** el HTML renderiza `grid_note` al pie de la tarjeta "Revisiones T-90..T-1 · por propiedad" — la pregunta "¿por qué n distinto?" se responde en el reporte, no en una conversación aparte
