## 1. Fundación del proyecto

- [x] 1.1 Crear el esqueleto `src/{data,validation,features,forecasting,models,evaluation,backtesting,explainability,monitoring,reporting}` + `tests/` + `config.yaml` y verificar que la estructura de directorios existe y el paquete importa sin errores
- [x] 1.2 Configurar el entorno de dependencias (Python + librerías de modelado/validación) y verificar la instalación con un comando de verificación (p. ej. `python -c "import ..."`) y la ejecución del runner de tests vacío
- [x] 1.3 Implementar el run manifest (versión de dataset, ventanas, hiperparámetros, seed, timestamp, métricas) y verificar con un test que una corrida sintética escribe el manifiesto completo y es reproducible con la misma seed
- [x] 1.4 Definir el config central (`FORECAST_HORIZON` 30/60/90, `RETRAIN_FREQUENCY`, `MIN_HISTORY`, seeds, rutas de fuentes) y verificar que cambiar el horizonte sin tocar código refleja el cambio en la corrida de prueba

## 2. Ingesta y calidad de datos (spec data-ingestion)

- [x] 2.1 Implementar la detección automática de esquema (propiedad, fecha, métrica) para las tres fuentes sin asumir nombres de columna y verificar con tests que reconoce esquemas alternativos
- [x] 2.2 Implementar la validación de calidad (duplicados, negativos, RN > capacidad, capacidad 0, fechas faltantes, lead inválidos, snapshots duplicados, inconsistencias de pickup) y verificar con tests que cada caso se detecta y se clasifica como error o advertencia
- [x] 2.3 Implementar el Data Quality Report (status, registros, errores, advertencias, propiedades y fechas afectadas) y verificar con un test que un dataset con problemas produce el reporte con los conteos esperados
- [x] 2.4 Implementar la detención del pipeline ante problemas críticos y verificar con un test que un dataset críticamente inválido impide entrenar y no escribe forecast

## 3. Anti-leakage (spec leakage-prevention)

- [ ] 3.1 Implementar el as-of snapshot builder como primitiva única de features y verificar con un test que para un corte T ninguna feature proviene de datos > T
- [ ] 3.2 Implementar lags/rollings/expansivos solo hacia atrás y verificar con un test que una ventana rolling de 28 días excluye cualquier observación futura
- [ ] 3.3 Implementar la validación que prohíbe splits aleatorios y verificar con un test que una configuración con split mixto falla antes de entrenar
- [ ] 3.4 Implementar el suite de pruebas de leakage (RN futuros, pickup futuro, capacidad futura, features con dataset completo, rolling con futuro, split aleatorio) y verificar que inyectar leakage deliberado hace fallar la corrida y no publica forecast

## 4. Feature engineering (spec feature-engineering)

- [ ] 4.1 Implementar features de calendario (dow, weekend, semana, mes, trimestre, año, día, fin de mes, feriado si existe fuente, temporada) y verificar con tests que se derivan solo de la fecha de estancia en el horizonte futuro
- [ ] 4.2 Implementar features de demanda histórica (lags 1/7/14/28/365, rolling 7/14/28/56, mediana, desviación, mismo dow, mismo mes) y verificar con tests el orden temporal y el manejo de histórico insuficiente (<1 año omite lag 365)
- [ ] 4.3 Implementar features de capacidad (diaria, día anterior, cambio, rolling, utilización, restante) y verificar con tests sus valores en un dataset con cambio de inventario
- [ ] 4.4 Implementar features de pickup (reserved actuales, lead days, ventanas 1/3/7/14/30, stats al mismo lead, volatilidad, aceleración/desaceleración, acumulado, % de demanda final) y verificar con tests la fórmula de aceleración y el as-of de "pickup al mismo lead"
- [ ] 4.5 Verificar con una prueba de correlación en datos de muestra que las features de pickup/capacidad aportan señal (sin fugas) antes de pasar a modelado

## 5. Modelado (spec forecasting)

- [ ] 5.1 Implementar el baseline Seasonal Naive (mismo día de semana histórico) y verificar con un test que produce forecast por propiedad/fecha
- [ ] 5.2 Implementar el baseline Pickup Curve (`Current Bookings + f(lead)` no lineal) y verificar con un test que la trayectoria T-90→T-1 es decreciente en pickup conforme baja el lead
- [ ] 5.3 Implementar los candidatos ETS y SARIMAX con la config de la sección de modelos y verificar que cada uno ajusta y predice en una serie sintética estacional
- [ ] 5.4 Implementar los candidatos ML LightGBM, XGBoost y CatBoost con features as-of y verificación temprana y verificar que cada uno entrena y produce predicciones en dataset de prueba
- [ ] 5.5 Implementar la capa de restricción de capacidad (`final = min(raw, capacity)`) y verificar con tests que nunca se excede la capacidad, que se conservan `raw/capacity/final` por separado y que se emite la alerta de violación
- [ ] 5.6 Implementar la ocupación implícita con validación 0–100% y verificar con un test que los valores fuera de rango se detectan
- [ ] 5.7 Implementar la jerarquía propiedad/portafolio con reconciliación y verificar con un test que la suma de propiedades se compara contra el portafolio y la brecha se reporta
- [ ] 5.8 Implementar intervalos P10/P50/P90 (nativos o conformal según D7) y verificar con un test que p10 ≤ p50 ≤ p90 en todos los registros y que los modelos sin intervalo quedan marcados

## 6. Evaluación (spec evaluation)

- [ ] 6.1 Implementar la validación rolling/expanding estrictamente temporal y verificar con un test que cada ventana entrena solo con datos anteriores
- [ ] 6.2 Implementar el cálculo de MAPE, WMAPE, MAE, RMSE y Bias y verificar con tests contra valores calculados a mano en un dataset pequeño
- [ ] 6.3 Implementar el manejo de `Actual = 0` en MAPE (exclusión + conteo documentado) y verificar con un test que las observaciones cero se excluyen, se cuentan y MAE/WMAPE las incluyen
- [ ] 6.4 Implementar los desgloses por banda de lead (90+, 60–89, 30–59, 14–29, 7–13, 3–6, 1–2) y verificar con un test que cada observación cae en exactamente una banda con sus métricas
- [ ] 6.5 Implementar los desgloses por propiedad y por estacionalidad (mes, dow, weekend, temporadas) y verificar con tests que producen las tablas completas y detectan un patrón de sesgo inyectado
- [ ] 6.6 Implementar la evaluación multi-step con snapshots congelados y verificar con un test que un forecast de corte T no lee ningún dato > T
- [ ] 6.7 Implementar la selección de campeón (WMAPE, MAE, |Bias|, estabilidad entre ventanas; MAPE no decide) y verificar con un test que un modelo con mejor MAPE pero bias fuerte no gana
- [ ] 6.8 Implementar la selección por propiedad con mínimo de ventanas y fallback a selección global y verificar con un test que las propiedades sin evidencia suficiente caen al modelo global

## 7. Backtesting y revisiones (spec backtesting)

- [ ] 7.1 Implementar el backtester de snapshots en leads 90/60/30/14/7/3/1 con datasets por corte cacheados y verificar con un test que un snapshot histórico no contiene datos posteriores a su corte
- [ ] 7.2 Implementar el almacenamiento de revisiones (forecast_date, lead_days por stay date) y verificar con un test que la serie T-90 → T-1 → Actual es reconstruible
- [ ] 7.3 Implementar el reporte de métricas del backtest (globales + por banda de lead) y verificar con un test que las métricas se calculan sobre los snapshots simulados

## 8. Explicabilidad (spec explainability)

- [ ] 8.1 Implementar explicaciones por predicción y globales para los modelos ML (SHAP/importancia) y verificar con un test que una corrida ML produce explicaciones pobladas
- [ ] 8.2 Implementar la contribución explícita de la restricción de capacidad en la explicación y verificar con un test que cuando hay cap aparece como contribución negativa
- [ ] 8.3 Implementar la versión de negocio de los drivers (reservas, pickup, estacionalidad, capacidad) y verificar con un test que la explicación incluye el resumen legible y que las contribuciones suman el forecast reportado

## 9. Reporting y entregables (spec reporting)

- [ ] 9.1 Implementar la tabla final de forecast con las columnas obligatorias del spec y verificar con un test que todas las columnas existen y están pobladas
- [ ] 9.2 Implementar las columnas extendidas (`historical_same_day_rn`, `expected_pickup_rn`, `pickup_rate`, `forecast_occupancy`, `forecast_bias`, `confidence_level`) cuando la información lo permite y verificar con un test su presencia/ausencia según disponibilidad
- [ ] 9.3 Implementar el dataset Power BI (tablas forecast/accuracy/pickup/modelo con Variance y Variance %) y verificar con un test que cada área del dashboard tiene sus campos cargables
- [ ] 9.4 Implementar los 7 KPIs ejecutivos y verificar con un test que los siete quedan calculados en una corrida de prueba
- [ ] 9.5 Implementar el Model Benchmark con los 7 candidatos (MAPE, WMAPE, MAE, RMSE, Bias, estabilidad) y verificar con una corrida de benchmark que la tabla tiene fila por candidato y campeón declarado con evidencia out-of-sample
- [ ] 9.6 Implementar la generación de los reportes restantes (Data Quality, EDA, Feature Engineering, Backtesting, Forecast Accuracy, Explainability, documentación técnica con limitaciones) y verificar que los 10 entregables se producen en una corrida completa
- [ ] 9.7 Verificar el estándar de ingeniería: la lógica vive en `src/`, los notebooks solo importan y presentan (revisión de estructura + suite de tests en verde)

## 10. Monitoring y retraining (spec monitoring)

- [ ] 10.1 Implementar la detección de drift de MAPE/WMAPE/Bias y verificar con un test que un deterioro inyectado genera alerta con métrica y magnitud
- [ ] 10.2 Implementar el drift de features, curva de pickup y distribución de demanda y verificar con tests que un desplazamiento inyectado se reporta
- [ ] 10.3 Implementar la detección de outliers con clasificación sin eliminación automática y verificar con un test que un pico real se conserva en el entrenamiento y queda clasificado
- [ ] 10.4 Implementar el retraining configurable (`RETRAIN_FREQUENCY`, `MIN_HISTORY`) con omisión documentada cuando no hay datos nuevos suficientes y verificar con un test que ambos caminos (omitir y re-entrenar) se ejercitan
- [ ] 10.5 Ejecutar una corrida end-to-end sobre las tres fuentes reales y verificar: 0 fallos de leakage, `final_forecast ≤ capacity` en el 100% de registros, métricas y desgloses presentes, dataset Power BI y manifiesto reproducibles
