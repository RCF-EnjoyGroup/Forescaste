# Tasks — monthly-vintage-report

- [x] 1.1 `scripts/monthly_vintage_report.py`: cortes 15-ene→15-oct 2026, cada uno proyecta mes en curso + 2 siguientes con los modelos operativos ajustados as-of el corte (`build_forecasts` con fit ≤ corte); mes en curso = reales 1-15 + pronóstico 16-fin.
- [x] 1.2 Excel (openpyxl): hoja `vintages` (Mes proyectado, Propiedad, Fecha corte, RN Real, RN Pronosticado, Nota) + hoja `lectura` con la metodología y las salvedades.
- [x] 1.3 Validación: 180 filas (10 cortes × 6 propiedades × 3 meses); reales de ene-2026 exactos contra demanda por propiedad; nov/dic con RN Real vacío; estructura de vintages correcta (ene 1 corte, feb 2, mar-oct 3, nov 2, dic 1); WMAPE del detalle 8.3%.
- [x] 1.4 Registro en openspec + commit.
