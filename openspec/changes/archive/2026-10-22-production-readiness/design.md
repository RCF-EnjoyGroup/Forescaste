# Design — production-readiness

## Por qué cuantiles completos (conformal) y no de-mediaados

La banda de-mediaaba los residuos (q10−q50, q90−q50) bajo el supuesto «el
sesgo del campeón ya vive en el punto». La cobertura empírica lo falsó:
celdas con bias de fold ≠ 0 perdían la masa desplazada (CORIN 67%, FIESTA
68% vs 80% nominal). Con cuantiles completos (−q90, −q10) la banda contiene
el 80% de la muestra de calibración POR CONSTRUCCIÓN, sesgo incluido; el
sesgo sistemático se corrige en el PUNTO (1.5) y la asimetría restante la
absorbe la banda. El contrato monótono p10≤p50≤p90 se proyecta y documenta
cuando > 90% de los residuos comparten signo.

## Corrección de bias con escpticismo de folds

El bias por (propiedad × banda) se aprende de los residuos del forecast
OPERATIVO en los folds, pero la medición real mostró varianza brutal entre
folds (una celda oscilaba +25/−17/−7/+55%). De ahí el **gate de
consistencia**: el bias debe tener el mismo signo en ≥ 75% de los folds para
corregir — sesgo que gira entre ventanas es régimen (p. ej. `seasonal_naive`
rezagando la tendencia), no evidencia. Encima, shrinkage n/(n+30) hacia 1.0
y clip ±30% (correcciones mayores no son bias, son otro régimen). Validación:
el bias a leads 60-90+ pasó de +23%/+30% a −2/−9% sin sobre-corregir.

## PSI calibrado para muestras chicas

Con ventana de 90d, el PSI de deciles tiene media nula 0.23 (todo alertaba).
Medición empírica: 5 bins + suavizado de conteos (máx(count, 0.5)/n) deja el
nulo en p95=0.22; cambios de nivel ≥ 5% dan PSI ≥ 0.58. Umbrales: warning
0.30, alert 0.50. Referencia YoY por propiedad (la "primera mitad del
histórico" incluía la era de 1-2 hoteles y disparaba 22σ siempre);
propiedades con < 365d quedan documentadas como no evaluables.

## Histéresis del campeón

Los empates DM (p=0.29-0.70) hacen que el override por propiedad oscile por
ruido entre corridas. La histéresis retiene al incumbente del registro salvo
ventaja material del 5% — el mismo margen del override global. Con data
nueva el cambio SÍ ocurre (FIESTA→lightgbm con ventaja material el 20-oct),
y con la misma data la selección queda congelada (verificado 2 días
consecutivos). Un modelo que cambia cada corrida erosiona la confianza; con
evidencia material cambia, sin ella no.

## Registro de estado: la memoria que faltaba

El manifiesto es por-corrida e inmutable; el registro es el único estado
entre corridas: append-only, escritura atómica (temp+rename), tolerante a
corrupción (jamás rompe). Habilita: drift de métricas real (solo con
`metrics_source` comparable — verificación vs backtest inyectado miden
ventanas distintas), re-entrenamiento con días/registros reales, incumbente
para la histéresis, y vintage contra la última corrida futura. Sin registro
(`registry_path=None`) el motor queda hermético e idéntico (tests).

## Vintage: medir el artefacto, no la simulación

El tracking vintage puntúa el forecast PUBLICADO por la corrida anterior
contra los reales que se materializaron — con la banda que se publicó y el
detalle por estancia en `vintage_tracking_detail.csv` (P50/banda/reservado
al publicar vs real, error, in_band, `published_run_id`). `forecast_date`
no "se actualiza" por diseño: es la añada — evaluar una publicación con
información posterior sería medir con la respuesta en la mano.

## Reporte auto-explicado

Los veredictos del panel son interpretativos (ayuda de lectura con criterio
explícito); los gates duros siguen en los criterios de aceptación del
runner y el gate de calidad. Identidad visual: membrete de corrida con
metadatos, KPIs serif tabulares, secciones por orden de lectura, estilo de
casa unificado para todos los gráficos (petróleo/terracota, sin adorno),
lámparas de veredicto imprimibles. Sin movimiento deliberadamente: es un
artefacto de archivo, no una demo.

## Non-goals documentados

- SHAP para campeones ML: maquinaria lista, no cableada al runner (el
  campeón nunca ha sido ML; el método sustituto está declarado).
- Orquestación/scheduler: excluida por el negocio; el registro la habilita.
