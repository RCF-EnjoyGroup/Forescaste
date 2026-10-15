# evaluation — delta

## MODIFIED Requirements

### Requirement: Cobertura empírica de la banda de incertidumbre
El accuracy report SHALL publicar la cobertura empírica de la banda P10–P90 (global, por propiedad y por banda de lead) contra el nominal 80% cuando la ventana evaluada tenga reales y banda disponibles; cuando la banda no exista sobre la ventana evaluada (corrida futura) el reporte SHALL documentarlo en lugar de omitir la métrica en silencio.

#### Scenario: Banda calibrada
- **WHEN** la banda contiene ~80% de los reales de la ventana
- **THEN** la cobertura global se reporta junto al nominal y al n de observaciones

#### Scenario: Celda mal calibrada visible
- **WHEN** una propiedad o banda de lead sub-cubre (p. ej. 67% vs 80% nominal)
- **THEN** el desglose por propiedad/banda expone la celda sin agregación que la oculte

#### Scenario: Sin banda sobre la ventana
- **WHEN** la corrida es futura y la banda vive sobre el horizonte sin reales
- **THEN** el accuracy report publica una nota explicando por qué no hay cobertura medible

### Requirement: Features exógenas de anticipación en candidatos ML
Los candidatos ML SHALL recibir el lead de la estancia respecto al corte (`lead_days`) como feature de primera clase — exógena, conocida a cualquier lead sin fuga (aritmética de calendario), con la misma semántica en panel (lead del snapshot) y horizonte (`stay − cutoff`).

#### Scenario: El modelo distingue anticipaciones
- **WHEN** dos estancias tienen el mismo on-the-books a leads distintos
- **THEN** la matriz de features las distingue por `lead_days` (no solo indirectamente por `current_booked_rn`)
