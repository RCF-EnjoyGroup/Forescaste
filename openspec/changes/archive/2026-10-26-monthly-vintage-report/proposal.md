# Proposal — monthly-vintage-report

## Contexto

El negocio solicitó un Excel de seguimiento mensual: para cada corte del día
15, la proyección del mes en curso + los 2 siguientes, contra el real de cada
mes proyectado (formato de la plantilla de referencia: matriz de vintages por
corte, ene-dic 2026).

## Propuesta

`scripts/monthly_vintage_report.py`: genera el Excel (hoja `vintages` con
Mes proyectado / Propiedad / Fecha corte / RN Real / RN Pronosticado / Nota,
y hoja `lectura` con la metodología).

- Cortes 15-ene → 15-oct 2026 (los posteriores aún no existen); cada corte
  proyecta el mes en curso + 2 siguientes → cada mes proyectado queda con
  1-3 vintages.
- Mes en curso: **proyección de cierre** = reales 1-15 + pronóstico 16-fin
  (la proyección clásica a mitad de mes).
- Modelos operativos vigentes ajustados **as-of cada corte** (fit con
  historia ≤ corte, sin ver el futuro) — reconstrucción honesta con el
  motor, no lo publicado entonces (el motor no existía a comienzos de
  2026); la hoja `lectura` lo declara.
- Validación: reales mensuales exactos contra la fuente; nov/dic sin
  reales; WMAPE del detalle 8.3% (grano mensual, todas las vintages).

## Non-goals

- Forecast de revenue/ADR: mismo detalle en $ requiere su propio pipeline.
- Selección de modelos as-of histórica por corte (lo que se habría
  elegido entonces): usa los operativos vigentes — documentado en la
  hoja `lectura`.
