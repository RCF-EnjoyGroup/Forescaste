## 1. Configuración del horizonte

- [x] 1.1 En `config.yaml` → `roomnights_real`: cambiar `horizon_days: 90` → `365` y `multi_step_min_lead: 90` → `365`, actualizando los comentarios (bucket único forward-known para el horizonte completo; entregable = pronóstico detallado de 12 meses)
- [x] 1.2 Verificar que `yaml.safe_load` devuelve los nuevos valores y que los tests existentes siguen verdes (`pytest -q`)

## 2. Notebook de producción (365d)

- [x] 2.1 Guardas de slicing: definir `n_cov = min(HORIZON, len(actuals_test))` y usarlo en todas las validaciones de cobertura sobre el test (bandas top-down: `per_horizon_conformal_bands(res[:, :n_cov], preds[:n_cov])`; bandas bottom-up MC: `lo_steps[:n_cov]`/`hi_steps[:n_cov]`)
- [x] 2.2 Cobertura de libros por segmento as-of seguro: días 1-90 con `rotb_d90`, 91-180 con `rotb_d180`, 181-365 con `rotb_d365` — reemplazar la métrica única y actualizar el print del resumen
- [x] 2.3 Etiquetas dinámicas: "TOTAL 90d" → f"{HORIZON}d", título del gráfico dinámico, encabezado MD "horizonte 365 días (12 meses)"
- [x] 2.4 Ejecutar el notebook de producción end-to-end y verificar: 365 fechas futuras por propiedad, bandas con hi > lo en todos los pasos, cobertura empírica reportada sobre los 213d de overlap, tabla de segmentos de libros con % por ventana, consistencia bottom-up vs top-down reportada

## 3. Notebook de re-validación (365d)

- [x] 3.1 Guardas de slicing en las dos celdas de bandas (celda de conformal top-down y celda de MC bottom-up): `n_cov = min(HORIZON, len(...))` y slice de residuales `[:, :n_cov]`
- [x] 3.2 Verificar que la celda de features de libros refleja el set reducido (`rotb_cols_ml` = solo `rotb_d365`) y que el print de cobertura rotb lo reporta
- [x] 3.3 Ejecutar el notebook de re-validación end-to-end (~45 min) y verificar: comparación de modelos con covariantes d365, bandas de 365 pasos, conclusiones con los nuevos totales, 0 errores
- [x] 3.4 Verificar la coherencia del pronóstico futuro a 365d del ganador (bandas asimétricas razonables, sin recortes de capacidad absurdos)

## 4. Reportes y documentación

- [x] 4.1 Regenerar los reportes HTML de ambos notebooks (`hotel_roomnights_production_report.html`, `hotel_roomnights_realdata_report.html`)
- [x] 4.2 Actualizar README (horizonte 365d en la tabla de pipelines) y añadir entrada 0.9.0 al CHANGELOG con los resultados del horizonte completo y las divulgaciones (cobertura validada en overlap 213d, guard de 12 meses para hoteles nuevos)
- [x] 4.3 Correr la suite completa de tests (`pytest -q`, 73 esperados) y `openspec validate hotel-roomnights-365-horizon`
