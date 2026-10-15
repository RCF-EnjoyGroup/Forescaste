# Design — external-review-remediation

## Contexto

Revisión externa de data science (2026-10-15) con 8 hallazgos. Cada uno se
validó contra el código y los datos ANTES de actuar — dos afirmaciones de la
revisión resultaron imprecisas y se documentan abajo.

## Decisiones

### D1 — La evaluación futura mide el pipeline completo, as-of (hallazgo 1)

`per_property_backtest` hardcodeaba `PickupCurveModel`. Tres niveles posibles
de fix:

- **Nivel 1**: operar el campeón seleccionado al corte (cero costo extra).
  Descartado: los folds de selección teselan el año previo al corte y el
  último fold SOLAPA la ventana de backtest — la selección habría visto
  parcialmente los datos sobre los que se reporta (sesgo optimista, sobre
  todo en asignaciones por propiedad).
- **Nivel 2 (elegido)**: repetir el benchmark con corte `cutoff − horizon`
  (as-of al inicio de la ventana de backtest), seleccionar campeón +
  por-propiedad con esa evidencia, y operar esa ventana con `build_forecasts`
  (fit/predict/blend de producción) + piso de reservas + techo de capacidad.
  Las métricas publicadas son literalmente "lo que el motor habría publicado
  hace `horizon` días", espejo exacto del modo verificación.
  Costo: el modo futuro corre `real_fold_metrics` dos veces (~4.4 min medidos
  por llamada). Aceptable para una corrida operativa mensual.

La métrica evalúa el forecast PUBLICADO (con restricciones), igual que en
modo verificación (`eval_src["forecast"] = final_forecast_rn`).

### D2 — `lead_days` como feature ML, capacidad fuera (hallazgo 2)

El lead es exógeno y sin fuga en ambos planos (panel: lead del snapshot;
horizonte: `stay − cutoff` — con snapshots diarios el par (lead, booked) es
consistente). Se agrega a `ML_FEATURES` y a `_ml_row`.

La capacidad como feature se DIFIERE: `fit_candidate`/`predict_candidate`
no reciben capacidad y pasarla exige cambiar la interfaz de candidatos +
folds + runner. El `lead_days` es la mitad barata y más informativa del
hallazgo; la capacidad queda como trabajo futuro documentado.

### D3 — Bordes de bins densos con catch-all (hallazgo 3)

`bin_edges: list[int] | None` en `PickupCurveModel` (precede a `n_bins`).
El último borde es catch-all (`[90, ∞)`): leads largos jamás quedan fuera de
rango (mejora sobre linspace, cuyo extremo superior dejaba fuera de rango a
leads > max_lead+1). Validación de config en `config.py` (creciente, ≥ 2,
primer borde ≥ min_lead) y en el modelo.

A/B sobre la ventana de verificación (`real-20261015-1202` vs
`real-20261014-1520`): banda 1-2 bias +1.85% → **−0.01%** y WMAPE 3.1% →
**1.8%** (el defecto medido por la revisión, eliminado); global WMAPE
12.79% → 12.73%; 5/6 propiedades mejoran; banda 30-59 +0.01 (ruido). Se
adopta el ladder corto `[1,3,7,14,21,30,45,60,90]` en `config.yaml`.

### D4 — Corrección de sesgo diferida (hallazgo 4)

El what-if de la revisión (WMAPE 12.36% → 11.76%) calibra y evalúa en la
MISMA ventana: mejora casi tautológica (el shift mediano minimiza el MAE
in-sample). Una corrección honesta debe calibrarse as-of en una ventana
previa y evaluarse fuera de muestra — con el D1 aplicado, las métricas
futuras ahora SÍ incluyen el sesgo verdadero del campeón operante, que es
el prerrequisito para diseñar esa corrección. Diferido, no descartado.

### D5 — Rollout recursivo diferido (hallazgo 5)

Eliminar los lags autorregresivos del ML o entrenar multi-horizon directo es
un cambio de investigación que invalidaría la evidencia de selección
acumulada. El blend + perfil DOW lo mitigan en producción. Nota: con
`lead_days` + `current_booked_rn` el GBM puede aprender el pickup restante
como función del lead — reduce la dependencia en los lags realimentados,
atacando parcialmente la causa.

### D6 — Margen de materialidad 5% para overrides (hallazgo 6)

Con 4 folds × 90d las diferencias pequeñas entre candidatos están dentro del
ruido muestral (el DM solo corre campeón vs subcampeón global). Un ganador
local solo opera si supera al campeón global por > 5% relativo del composite
en esa propiedad; si no, retención documentada en `property_fallback`.
`override_margin=0` restaura el comportamiento anterior (probado en test).

### D7 — Baseline suavizado (hallazgo 7)

`SeasonalNaive.dow_window = 4`: media de las últimas K ocurrencias del
mismo DOW. El benchmark solo es creíble si ganarle al baseline cuesta algo.
Efecto observado en la corrida de verificación: el benchmark movió sus
números (catboost mejora con lead_days a 0.223, el naive a 0.310) y FIESTA —
gran propiedad, estable — pasa a operar seasonal_naive por evidencia de
folds (su ventana verificada queda en 9.9% WMAPE, dentro del ruido frente
al 9.3% previo, sin la maquinaria de blend).

### D8 — Cobertura empírica publicada (hallazgo 8)

`interval_coverage` en `breakdowns.py`: global + por propiedad + por banda
de lead, contra el nominal 80%. En la corrida de verificación: global 81.1%
(calibrada), FIESTA 67.8% y CORIN 66.7% (sub-cobren — ahora VISIBLE). La
recalibración fina (umbral `min_residuals` por celda) queda como trabajo
futuro: primero hay que medir, luego ajustar.

## Precisiones a la revisión externa

- La cobertura global citada (80.7%, 450 obs) omitía CORIN por no resolver el
  alias "Hotel Royal Corin": el valor correcto es 78.3% (540 obs); CORIN
  también sub-cubría (66.7%).
- El fix propuesto para el hallazgo 1 (pasar `model_by_property` del campeón
  vigente) habría dejado sesgo de selección; el diseño as-of es la versión
  honesta del mismo fix.

## Riesgos y mitigaciones

- **Doble benchmark en futuro**: +4.4 min. Aceptado; corridas futuras ~11 min.
- **Cambio de candidatos altera selecciones por propiedad**: esperado y
  deseado (la evidencia mide modelos distintos ahora); el margen D6 acota el
  churn por ruido.
- **`bin_edges` cambia el campeón operante**: validado con A/B antes de
  adoptarse; el ladder queda en `config.yaml` con el resultado documentado.
