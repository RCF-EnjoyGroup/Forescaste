# Tasks — recency-calibration

- [x] 1.1 Diagnóstico: FIESTA erró unidireccional en la añada con firma visible — perfecto fines de semana, desplome entre semana que `seasonal_naive` no vio (jue +24%/+39%, mar +29%: promedia las últimas 4 semanas y octubre cayó la demanda corporativa); SJOSL sobre-pronóstico sistemático (+22%) = cold start documentado. **Ambos son señales reales, no artefactos** — se re-gravan con la próxima añada.
- [x] 1.2 `src/forecasting/core.py`: `operative_residual_bands` emite celdas como DataFrame (residual, weight) con peso `decay^(edad del fold)`. Verificar: tests de contrato actualizados en `test_per_property_intervals.py`.
- [x] 1.3 `src/forecasting/constraints.py`: cuantiles PONDERADOS (`_wquantile`, interpolación sobre pesos acumulados), `_shifts` acepta celdas Series (back-compat) o DataFrame, escalera preserva pesos en los pools (`_pool_concat`). Verificar: `test_recencia_desplaza_la_banda_hacia_el_regimen_reciente` (la banda se pega al régimen reciente y el p90 honesto bajo el punto se proyecta al contrato DOCUMENTADO), `test_recencia_uniforme_es_el_comportamiento_previo`, `test_build_intervals_acepta_celdas_series_y_dataframe_mixtas`.
- [x] 1.4 `config.yaml` + `src/config.py`: `monitoring.interval_recency_decay`.
- [x] 1.5 Panel: fila TS marca hoteles provisionales (⚠ + criterio que lo explica); filas de benchmark citan la añada publicada reciente cuando cubre los leads del grupo (`_vintage_grupo`, pooling idéntico). Verificar: `test_panel_tracking_signal_marca_hoteles_provisionales`, `test_panel_benchmark_cita_la_annada_reciente`.
- [x] 1.6 Tests de los módulos afectados en verde (137 en los 6 archivos).
- [x] 1.7 **A/B con corrida real** (real-20261022-1506, misma data/corte/seed; WMAPE idéntico al dígito — solo bandas cambian): decay 0.7 mejoró FIESTA 52→57% y VILLAS 92→90% pero degradó LIREL 93→97% y MARINA 63→60%; desviación media |cov−80%| 0.131→0.135. **La regla pre-registrada no se cumple → el default queda en 1.0 (uniforme); la maquinaria y el knob quedan disponibles** con el A/B documentado en config.yaml para recalibrar cuando el expediente acumule vintages.
- [x] 1.8 Suite completa + specs + este change archivado + commit.

## Veredicto de los veredictos (cierre honesto del pedido "solucionar los rojos")

- **Entregado**: marca ⚠ de cold start en la fila TS (SJOSL se lee como lo
  que es), cita de la añada reciente junto al corto ("eval 10.0% · añada
  publicada reciente: 6.9% real"), y la maquinaria de recencia con su A/B.
- **Señales reales que se quedan (correctamente)**: TS de FIESTA/SJOSL
  (erraron unidireccional; el operador y las correcciones ya cambiaron —
  la próxima añada re-grava), cobertura de LIREL/MARINA (la recencia NO
  las arregló — evidencia en contra registrada), drift de MARINA
  (lectura operativa por decisión del negocio).
