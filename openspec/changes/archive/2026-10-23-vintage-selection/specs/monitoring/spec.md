# monitoring — delta

## MODIFIED Requirements

### Requirement: Estado entre corridas (registro)
#### Scenario: Tracking vintage del forecast publicado
- **WHEN** la corrida anterior publicó un forecast operativo y parte de su horizonte ya tiene reales
- **THEN** ese forecast se puntúa por vintage (métricas, desglose por lead y cobertura de banda)
- **THEN** la añada graduada es la publicación operativa MÁS RECIENTE cuya ventana ya tiene reales — una futura recién publicada con corte en el último real aún no es graduable (su horizonte entero es futuro); el reporte declara qué añada se graduó (`published_run_id`)
- **WHEN** ninguna estancia del forecast anterior se ha materializado
- **THEN** el CSV no se publica (nunca se inventa) y el manifiesto lo documenta
