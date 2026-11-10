# Proposal — band-calibration

## Contexto

La corrida publicada `real-20261106-1323` (n=539) muestra que la banda
P10-P90 NO es un intervalo del 80% para ninguna propiedad (verificado
leyendo `outputs/runs/real-20261106-1323/forecast_accuracy_report.json`):

| propiedad | cobertura publicada (nominal 80%) | dirección |
|---|---|---|
| SJOSL | 67.8% | sub-cubre |
| MARINA | 66.3% | sub-cubre |
| FIESTA | 71.1% | sub-cubre |
| CORIN | 74.4% | sub-cubre |
| LIREL | 93.3% | sobre-cubre |
| VILLAS | 95.6% | sobre-cubre |

Cobertura global 78.1%. Dos celdas propiedad×banda caen a 14-25% justo
donde el negocio decide (CORIN 14-29 = 25%, FIESTA 7-13 = 14.3%). Lo que
publica el motor como banda del 80% en realidad es 66-96% según la
propiedad, en la dirección del error de nivel (FIESTA con nivel +10% YoY
tiende a sub-cobertura; VILLAS 0.71x y LIREL 0.47x de nivel a
sobre-cobertura).

## Propuesta

Un knob `interval_calibration` (default **OFF**, como los knobs previos de
corrección de bias/nivel: nadie cambia el comportamiento publicado hasta
tener A/B positivo) que aplica un factor de escala multiplicativo por
propiedad al semiancho de la banda, aprendido de la ventana reciente:

- ventana de entrenamiento: el backtest más reciente (mismos folds que
  alimentan los residuos por banda de lead que ya usa el motor);
- target: cobertura empírica por propiedad ≈ nominal (80%);
- shrink hacia 1.0 cuando la evidencia es poca (pocas obs por propiedad),
  para no sobre-ajustar propiedades chicas;
- publicación: el factor por propiedad queda en el manifiesto (auditable),
  junto con la cobertura observada que lo justifica.

What-if medido sobre la ventana publicada (in-sample, etiquetado como tal
en el diseño): un factor k_p por propiedad (SJOSL 1.94x, FIESTA 1.52x,
CORIN 1.35x, MARINA 1.18x más anchas; LIREL 0.70x, VILLAS 0.66x más
angostas) lleva las 6 a ~80% y la global de 78.1% a ~80%. Eso es el
límite superior esperado; el A/B de replay decidirá si se enciende.

## Non-goals

- No cambiar el método de construcción de la banda base (residuales por
  banda de lead), solo escalarla por propiedad.
- No tocar el punto del forecast (p50) — esto es solo incertidumbre.
- No cambiar defaults de otros knobs.

## Validación

- Test nuevo: con knob ON sobre el backtest etiquetado, la cobertura por
  propiedad se acerca al nominal más que sin el knob; con knob OFF, la
  banda es idéntica a la actual (byte a byte en el entregable de test).
- Suite completa en verde.
- A/B de replay real (3 brazos: OFF / shrink débil / shrink fuerte) antes
  de proponer default ON; hasta entonces el default queda OFF.
