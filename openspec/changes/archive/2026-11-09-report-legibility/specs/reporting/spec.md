# reporting — delta

## MODIFIED Requirements

### Requirement: Reporte auto-explicado

#### Scenario: Comparativa real-vs-forecast nunca silenciosa
- **WHEN** la corrida es de verificación y su ventana evaluada tiene reales, pero la unión con el horizonte publicado no produce coincidencias
- **THEN** las tarjetas "Forecast vs Real" y "Dispersión" se renderizan con el MOTIVO explícito de la indisponibilidad (p. ej. "la unión de reales con el horizonte publicado no encontró coincidencias — revisar alineación de claves")
- **THEN** solo cuando genuinamente no hay reales (corrida futura: el horizonte aún no existe) las tarjetas se omiten, que es el comportamiento documentado

#### Scenario: Métricas en escala legible
- **WHEN** el reporte muestra WMAPE, sesgo, MAPE, n_obs o KPIs de accuracy
- **THEN** WMAPE y sesgo se muestran como porcentaje (se guardan como fracción), MAPE como porcentaje, n_obs y exclusiones como conteos enteros, y los valores inexistentes (NaN) como "—" coherente con el N/D del panel
- **THEN** una nota de auditoría explica la escala de cada métrica
