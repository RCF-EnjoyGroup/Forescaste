## Why

Revisión del `report.html` (real-20261014-1224): en el forecast por propiedad, **las bandas P10-P90 son idénticas para las 6 propiedades** (13.2/18.4/32.7/47.6 RN en las mismas bandas de lead) porque `build_intervals` calibra con residuos del PORTAFOLIO agrupados solo por banda de lead. Consecuencias medidas en la ventana de verificación:

- **CORIN (54 hab, media 30 RN)**: banda de 41 RN a leads 61-90 — más ancha que el hotel mismo; sobre-cobertura 97-100% (objetivo 80%). El usuario lo ve como "rango muy dilatado".
- **FIESTA (408 hab)**: banda de 30 RN a leads 30-59 cuando su error real es de 40-60 RN; sub-cobertura 25-37%. Hasta el P90 queda por debajo del real de los findes.

La fuente actual de residuos (backtest T-90..T-1 muestreado) no puede alimentar buckets por propiedad: solo 4-8 observaciones por (propiedad, banda) — bajo `min_residuals=20`, cualquier keying por propiedad caería al pool de hoy. Hace falta una fuente de residuos del **forecast operativo** con muestra suficiente.

Esa fuente ya existe y se descarta: los folds del benchmark predicen los 7 candidatos por propiedad por ventana (4 × 90 días = ~360 obs/propiedad). Ensamblando el forecast operativo (campeón + empalme del blend + perfil DOW) por fold se obtienen residuos por (propiedad, banda) con 28-120 observaciones.

**Contrafactual validado** (residuos operativos de folds, aplicados a la corrida real-20261014-1445):

| Propiedad/Banda | Ancho actual | Ancho por propiedad | Cobertura actual | Cobertura nueva |
|---|---|---|---|---|
| CORIN 30-59 | 32.2 | **15.5** | 97% | 73% |
| CORIN 60-89 | 40.8 | **21.0** | 97% | 73% |
| FIESTA 14-29 | 17.5 | **80.3** | 25% | 62% |
| FIESTA 30-59 | 30.4 | **72.4** | 37% | 57% |
| VILLAS 60-89 | 47.0 | **110.4** | 57% | 100% |
| LIREL 30-59 | 32.2 | **32.8** | 53% | 83% |

Las anchuras pasan a ser proporcionales a la escala de error de cada propiedad y la cobertura se acerca al objetivo 80% en la mayoría de celdas.

## What Changes

- **`real_fold_metrics` retiene las predicciones de los folds**: además de las métricas, devuelve un DataFrame `fold_predictions` (property, fold, model, stay_date, lead_days, forecast, actual) — datos que ya computa y hoy descarta.
- **Ensamble del forecast operativo por fold**: nueva función en core que, dadas las predicciones de folds + campeones + blend (modelo largo, split, perfil DOW), reconstruye el forecast operativo por (propiedad, fold) y emite sus residuos — sin re-entrenar nada.
- **Bandas por (propiedad, banda)**: `build_intervals` acepta claves (propiedad, banda) y `property_col`; escalera de fallback: (propiedad, banda) con ≥ 20 residuos → pool de la propiedad → pool de la banda (portafolio, comportamiento actual) → global.
- **Cableado**: `run_real` pasa las predicciones de folds al runner; el runner construye las bandas operativas tras decidir el blend y las usa para los intervalos; el manifiesto declara `interval_source: operative_fold_residuals` (o el fallback actual).
- Sin fold predictions (corridas de muestra/tests): comportamiento actual (pool del portafolio desde el backtest).

## Capabilities

### Modified Capabilities
- `forecasting`: MODIFIED — Intervalos de incertidumbre: la amplitud se calibra por (propiedad, banda de lead) con residuos del forecast operativo; el ancho refleja la escala de error de cada propiedad.

## Impact

- **Código**: `src/forecasting/run_real.py` (retener predicciones), `src/forecasting/core.py` (ensamble operativo + residuos), `src/forecasting/constraints.py` (build_intervals por propiedad), `src/forecasting/runner.py` (cableado + interval_source), tests.
- **Comportamiento observable**: CORIN ve sus bandas reducirse ~50%; FIESTA/VILLAS las ven crecer a su escala real; el manifiesto declara la fuente de los intervalos; el forecast puntual NO cambia en nada.
- **Riesgo**: bajo — sin fold predictions el comportamiento es el actual; las celdas con pocas observaciones caen en la escalera documentada.
