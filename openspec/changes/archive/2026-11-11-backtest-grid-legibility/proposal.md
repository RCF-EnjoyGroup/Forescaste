# Proposal — backtest-grid-legibility

## Contexto (pregunta de comparabilidad del reporte)

El usuario leyó la sección "Revisiones T-90..T-1 · por propiedad" del
reporte y preguntó por qué el `n` difiere entre propiedades (56 vs 16):
"¿se debe tener n cantidad igual para todas las propiedades? ¿cómo
comparar papas con papas?".

## Verificación en código (hallazgo, no suposición)

1. **Selección de campeón**: `real_fold_metrics` (run_real.py) NO usa
   rejillas por modelo — los 7 candidatos se ajustan por fold y predicen
   el MISMO horizonte completo sobre las MISMAS fechas y propiedades.
   Papas con papas.
2. **Bandas P10/P90 publicadas**: `operative_residual_bands` se alimenta
   de `fold_predictions` (la evidencia de rejilla común), filtrada al
   campeón que opera; `interval_source = operative_fold_residuals` en el
   manifiesto de `futuro-20261111-0712`. El snapshot desigual NO alimenta
   la banda publicada.
3. El `n` desigual vive SOLO en el backtest de snapshots diagnóstico
   (`_sampled_backtest`, runner.py): baratos 7 leads × 8 estancias = 56,
   caros 4 × 4 = 16 (guard de costo documentado). Cada propiedad se mide
   contra SÍ MISMA entre corridas; las métricas son relativas (WMAPE),
   comparables entre propiedades aunque el n difiera.

## Conclusión

NO se necesita un A/B de "rejilla común": la maquinaria que decide
(selección y bandas) ya opera sobre rejilla común; unificar la rejilla
del snapshot diagnóstico no cambiaría ningún número publicado y solo
agregaría cómputo. Lo accionable es **legibilidad**: el reporte muestra
el `n` sin explicar por qué difiere (la confusión del usuario es la
evidencia — "nada silencioso").

## Propuesta

1. `runner.py::_sampled_backtest`: añadir `grid_note` al
   `backtesting_report.json` — explica el n por tipo de modelo (con las
   constantes reales de la corrida), el propósito diagnóstico y que la
   selección y las bandas usan la rejilla común del benchmark de folds.
2. `visual_report.py`: la tarjeta "Revisiones T-90..T-1 · por propiedad"
   renderiza `grid_note` como nota al pie (mismo estilo que los demás
   explainers).
3. Test: el reporte de una corrida mini contiene `grid_note` con las
   dos rejillas; el HTML la incluye cuando el reporte la trae.

## Non-goals

- No unificar las rejillas del snapshot (sin efecto publicado, ver arriba).
- No tocar selección, bandas ni ningún número.
