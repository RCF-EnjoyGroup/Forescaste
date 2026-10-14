## Why

Revisión del `report.html` de la corrida `real-20261014-1224` (2026-10-14): en el forecast por propiedad, **FIESTA no toma los picos de fin de semana a largo lead** — a leads > 30 opera el sarimax del blend accuracy-aware, que es plano por construcción (la componente estacional SARIMA decae a 60-90 pasos y el forecast converge al nivel medio). El P50 largo es ~337 RN plano contra reales de finde de 398-410 RN: gap de +68 RN en vie/sab, y hasta el P90 (364-379) queda por DEBAJO del real del finde.

El nivel de sarimax es correcto (por eso ganó la evidencia: wmape_long 10.2% vs 26.3% de la curva) — lo que falta es el PATRÓN DOW. El patrón semanal de una propiedad es su ritmo operativo, estable y conocible a cualquier lead (viene de historia ≤ corte, sin lookahead).

**Contrafactual validado** (nivel de sarimax × perfil DOW normalizado de los últimos 84 días as-of del corte):

| Escenario | sarimax puro | sarimax × DOW | delta |
|---|---|---|---|
| fold0 | 9.2% | 7.9% | −1.3 |
| fold1 | 7.2% | 6.9% | −0.3 |
| fold2 | 8.3% | 7.1% | −1.2 |
| fold3 | 16.2% | 13.1% | −3.1 |
| T0 (jul-oct) | 14.1% | 10.3% | −3.8 |
| **promedio** | **10.98%** | **9.07%** | **−1.9** |

Mejora el wmape largo en **las 5 ventanas** — nunca daña — y restaura los picos (gap finde +68 → +15 RN). Correlación del patrón DOW con el real: 0.96 con perfil vs plano.

## What Changes

- **Perfil DOW para reemplazos planos del blend**: cuando el blend empalma un candidato de kind `stats` (SARIMAX/ETS — planos a largo horizonte) en los leads > `BLEND_LEAD_SPLIT`, el segmento empalmado se multiplica por el perfil DOW de la propiedad: nivel del mejor modelo × patrón de la historia reciente.
- El perfil se calcula **as-of** (solo historia ≤ corte): media de RN por día-de-semana en los últimos 84 días (12 semanas), normalizada a media 1. Guardas: historia insuficiente (< 28 días o < 3 obs por DOW) → sin perfil (segmento plano, comportamiento actual).
- La selección del reemplazo NO cambia: sigue siendo el mejor `wmape_long` por evidencia con margen. El perfil es una **capa de patrón** sobre el nivel elegido, no una re-selección.
- El manifiesto declara la decisión: `per_property_blend` gana `long_pattern` ("dow_profile(84d)" o "flat").

## Capabilities

### Modified Capabilities
- `forecasting`: MODIFIED — Champion/challenger con blend por lead: los reemplazos planos reciben el patrón DOW de la propiedad sobre su nivel.

## Impact

- **Código**: `src/forecasting/core.py` (perfil DOW + aplicación al segmento empalmado), tests.
- **Comportamiento observable (FIESTA con los datos actuales)**: P50 largo pasa de plano ~337 a patrones vie/sab ~385-395; gap de picos +68 → ~+15 RN; wmape largo esperado 11.0 → 9.1% según folds.
- **Riesgo**: bajo — solo aplica al segmento empalmado con modelos stats (hoy: FIESTA); guardas ante historia corta; validado en 5 ventanas sin contracara.
- El problema de las BANDAS (anchas para CORIN, estrechas para FIESTA) es un cambio separado (`per-property-intervals`): la calibración actual agrupa residuos de todas las propiedades en un solo pool por banda de lead.
