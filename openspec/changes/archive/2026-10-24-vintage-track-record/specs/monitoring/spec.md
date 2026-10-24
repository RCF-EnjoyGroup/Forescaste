# monitoring — delta

## MODIFIED Requirements

### Requirement: Estado entre corridas (registro)
#### Scenario: Tracking vintage del forecast publicado
- **THEN** el EXPEDIENTE de añadas graduadas se reconstruye desde el registro (deduplicado por publicación — atribución por `published_run_id` o por corte = inicio de la ventana − 1d —, cada añada con su graduación más reciente) y se publica como sección del reporte con tabla y gráfica por añada, más una fila en el panel con la tendencia del expediente y la salvedad explícita del n de cada añada
- **WHEN** la corrida que regenera un reporte es anterior a graduaciones nuevas
- **THEN** el expediente se reconstruye as-of esa corrida (solo registros anteriores) — los reportes viejos no citan añadas graduadas después de su fecha

# reporting — delta

## MODIFIED Requirements

### Requirement: Reporte auto-explicado
#### Scenario: Sección del expediente del forecast publicado
- **WHEN** el registro tiene ≥ 1 añada graduada
- **THEN** el grupo Precisión incluye la tarjeta «Expediente del forecast publicado»: gráfica de WMAPE por añada (con su n), tabla (corte de publicación, ventana graduada, n, WMAPE, bias, cobertura de banda, TS peor) y guía de lectura
- **THEN** el panel de validación incluye la fila «Expediente del forecast publicado» con la tendencia (una añada → «Acumulando»; ≥2 → mejora/estable/revisar contra el promedio de las previas) y el n de cada añada explícito
