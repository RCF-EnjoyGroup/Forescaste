## Why

FIESTA tiene un swing de 200+ RN entre jueves (190-260) y sábado (380-440). SARIMAX, esencialmente plano (sensibilidad día-de-semana: 4 RN), gana el composite porque la media de WMAPE sobre ventanas de 90d "diluye" los desastres de los jueves de corto lead entre 60 días de leads largos donde ser plano "no es terrible". El resultado: se opera el modelo que NO ve el patrón real de la propiedad (WMAPE 0.141) sobre el que sí lo ve (pickup_curve, WMAPE 0.130, DOW sens 179 RN).

## What Changes

- **Composite ponderado por lead**: el WMAPE del score de selección se desagrega en `wmape_short` (leads 1-30) y `wmape_long` (leads 31-89) por fold; el composite usa `0.65·wmape_short + 0.35·wmape_long` en lugar del WMAPE plano de 90d. Los leads operativos —donde vive el staffing y el revenue management— pesan ~2× por día que los de planeación.
- **Backward compatible**: sin las columnas nuevas (corridas de muestra, tests antiguos), el composite cae al WMAPE plano actual. Sin cambios en DM, parsimonia ni selección por propiedad (que hereda el composite nuevo).
- **Constantes declaradas**: `LEAD_ALPHA = 0.65`, split en lead 30, documentadas y ajustables.

## Capabilities

### Modified Capabilities

- `evaluation`: MODIFIED — Integridad del benchmark de candidatos: el composite pondera los leads operativos (1-30) con peso 0.65 sobre los de planeación (31-89, peso 0.35); sin columnas de desagregación, cae al WMAPE plano.

## Impact

- **Código**: `src/forecasting/run_real.py` (wmape_short/long por fila de fold), `src/evaluation/selection.py` (composite ponderado), tests.
- **Comportamiento observable**: campeones por propiedad pueden cambiar (FIESTA: sarimax → pickup_curve esperado); los benchmarks MD cambian (el composite ya no es el WMAPE plano).
- **Riesgo**: propiedad con buen largo-lead y mal corto-lead puede perder la selección — es el objetivo del cambio, pero se verifica contra regresión en TODAS las propiedades antes de cerrar.
