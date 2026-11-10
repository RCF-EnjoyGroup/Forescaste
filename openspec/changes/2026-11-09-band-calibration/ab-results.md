# A/B de replay — band-calibration (2026-11-10)

4 corridas reales de verificación (`python -m src.forecasting.run_real`), mismo
corte, misma data; config y registry respaldados y restaurados tras cada
brazo. WMAPE idéntica en los 4 (0.1369): el knob solo toca la banda.

- **A (control)**: knob OFF — el comportamiento publicado.
- **B**: ON, `shrinkage_obs=10` (shrink débil), k aprendido del registro (corrida A).
- **C**: ON, `shrinkage_obs=90` (shrink fuerte).
- **D**: ON, `shrinkage_obs=30`, k crudos de A como override de config.

## Cobertura P10-P90 observada (nominal 80%)

| propiedad | A | B | C | D |
|---|---|---|---|---|
| CORIN | 74.4% | 78.9% | 74.4% | 78.9% |
| FIESTA | 71.1% | 80.0% | 71.1% | 80.0% |
| LIREL | 93.3% | 75.6% | 91.1% | 77.8% |
| MARINA | 66.3% | 66.3% | 66.3% | 66.3% |
| SJOSL | 67.8% | 67.8% | 67.8% | 67.8% |
| VILLAS | 95.6% | 76.7% | 94.4% | 77.8% |
| **global** | **78.1%** | **74.2%** | **77.6%** | **74.8%** |
| dispersión por propiedad | 29.3pp | 13.7pp | 24.5pp | 13.7pp |

## Veredicto: default sigue OFF

1. **Ningún brazo mejora la cobertura global** (78.1% control vs 74.2-77.6%
   con knob). Llevar cada propiedad al 80% es aritméticamente negativo
   cuando dos propiedades ya sobre-cubren 93-96%: bajarlas quita más
   cobertura global de la que agrega subir las sub-cubiertas.
2. El what-if in-sample de la propuesta (~82%) **sobrestimó**: se midió
   sobre la misma muestra que calibró. El A/B con k aplicado da 74-78%.
   Lección registrada: los what-if in-sample de este motor sesgan al alza.
3. Las dos propiedades más descallibradas (MARINA 66.3%, SJOSL 67.8%) no
   se mueven en ningún brazo: sus celdas quedan fuera de los gates
   (SJOSL pool 90 < 120 por cold-start) — el knob no toca a quien más
   lo necesita.
4. Sí logra lo que promete donde aplica: dispersión por propiedad
   29.3pp → ~14pp y las sub-cubiertas al nominal. Eso tiene valor de
   previsibilidad por hotel, pero no el que justifica cambiar el default
   publicado.

## Si se revisita en el futuro

- Target correcto: cobertura global o con costo asimétrico (sub-cobertura
  cuesta inventario no vendido / overbooking; sobre-cobertura desperdicia
  inventario), no 80% equiprobable por propiedad.
- Incluir a MARINA/SJOSL: requieren pool suficiente (más histórico o
  pool de portafolio con k del portafolio), no gates que las excluyan.
- El k crudo por propiedad quedó validado como aprendizaje (coincide
  ±0.05 con la medición independiente); lo que falló es el TARGET, no
  la mecánica.
