# monitoring — delta

## MODIFIED Requirements

### Requirement: Estado entre corridas (registro)
El sistema SHALL persistir un registro append-only entre corridas con: métricas, fuente de métricas, campeón global y por propiedad, cobertura de banda y timestamp de cada corrida; el registro SHALL escribirse atómicamente y un registro ausente o corrupto SHALL jamás romper la corrida.

#### Scenario: Tracking vintage del forecast publicado
- **WHEN** la corrida anterior publicó un forecast operativo y parte de su horizonte ya tiene reales
- **THEN** ese forecast se puntúa por vintage (métricas, desglose por lead y cobertura de banda)
- **THEN** la señal de rastreo (Σ error con signo ÷ desviación media) se calcula POR PROPIEDAD sobre la añada — el control de sesgo sistemático acumulado que el MAPE esconde, con umbral configurable (clásico |TS| ≤ 4)
- **WHEN** ninguna estancia del forecast anterior se ha materializado
- **THEN** el CSV no se publica (nunca se inventa) y el manifiesto lo documenta
