# reporting — delta

## MODIFIED Requirements

### Requirement: Reportes del proyecto

#### Scenario: Excel de vintages con selección operativa anclada
- **WHEN** el reporte mensual de vintajes reconstruye la proyección de cada corte (día 15)
- **THEN** el mapa de modelos por propiedad sale del registro entre corridas (modo `futuro` con `cutoff <= corte`, la selección operativa vigente a esa fecha — memoria por línea, openspec 2026-11-10-registry-mode-memory)
- **THEN** los cortes previos al motor usan el registro futuro más antiguo disponible y la ausencia de registro cae al snapshot histórico — en ambos casos la fuente queda declarada
- **THEN** la hoja de lectura lleva el sello de versión: fecha de generación, fecha de data usada, regla de selección aplicada y mapa de modelos por propiedad, de modo que dos archivos de fechas distintas sean comparables sin adivinar
