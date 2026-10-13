# Design — gbm-blend-long-leads

## Decisions

### D1 — Blend en `build_forecasts`, no en `predict_candidate`
El blending necesita DOS modelos fitted por propiedad (el campeón GBM + la curva). `build_forecasts` es donde ya se hace el fitting por propiedad: ahí se ajusta la curva adicional cuando el campeón es GBM, y se combinan las predicciones por lead. `predict_candidate` no cambia — cada modelo sigue produciendo su predicción pura.

### D2 — Split duro en lead 30, no interpolación
`BLEND_LEAD_SPLIT = 30` (misma constante que `LEAD_SPLIT` del composite). En leads 1-30, el GBM tiene lags mayormente reales (solo los primeros pasos son sintéticos) y captura el DOW. En leads 31+, el rollout del GBM es 100% sintético y degrada. Un split duro es más simple, determinístico y auditable que una transición gradual; la transición ya existe de facto porque el GBM y la curva convergen en el rango 25-35.

### D3 — La curva usa los mismos parámetros del config
`curve_params` del config (temporada, feriados, recencia) — misma configuración que operaría si fuera campeona. No hay parámetros "especiales" para la curva de blend.

### D4 — El blend NO aplica al backtest muestreado
`_sampled_backtest` evalúa el CAMPEÓN a cada lead. El blend es una decisión de producción (cómo operar el campeón), no de evaluación. El backtest sigue midiendo el campeón puro.

### D5 — Manifiesto: `per_property_blend` declara el split
En `hyperparameters`, un dict `{prop: {"blend_split": 30, "short_model": "lightgbm", "long_model": "pickup_curve"}}` para propiedades con blend GBM. Propiedades sin GBM: ausentes del dict.

## Risks / Trade-offs

- [Curva sobre-estima fines de semana a largo lead] — el techo de capacidad la contiene; el piso de booked la protege en corto. Documentado en el A/B: sáb de sept +17% para la curva vs −13% del GBM.
- [Doble fitting agrega runtime] — solo para propiedades con campeón GBM (hoy: FIESTA, MARINA); ~1-2s por propiedad adicional.
- [El fold evaluation no ve el blend] — el benchmark mide los modelos puros; el blend es post-selección. El composite ponderado por lead (0.65 corto) ya favorece modelos que ganan en corto.
