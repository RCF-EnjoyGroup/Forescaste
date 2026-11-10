# Design — report-legibility

## El blindaje: de tarjeta fantasma a tarjeta declarada

Hoy las dos tarjetas dependientes de reales se llaman con
`placeholder=False` y sus funciones devuelven `None` cuando `actual`
es `None` (o está vacía) → la tarjeta desaparece sin explicación. El
defecto no es que desaparezca (en modo futuro es correcto: no hay
reales) sino que también desaparece cuando DEBERÍA haber reales y algo
falló — indistinguible para el lector.

Nuevo parámetro `actual_note: str | None = None` en
`generate_visual_report`, calculado por el runner:

- corrida de verificación con reales en `eval_src` (`n_real > 0`) Y unión
  vacía → nota con el motivo accionable ("la unión de reales con el
  horizonte publicado no encontró coincidencias — revisar alineación de
  claves");
- cualquier otro caso → `None` (sin reales: la omisión sigue siendo el
  comportamiento documentado).

Helper `_actual_card(name, img, note)`: imagen si existe; tarjeta con la
nota si no hay imagen pero hay nota; omisión solo si no hay ninguna de
las dos. Se usa en "Dispersión" (grupo Precisión) y "Forecast vs Real"
(grupo Pronóstico).

## Legibilidad

- `_fmt_metrics`: escala por clave — wmape/bias (fracción) → %,
  mape (×100) → %, n_obs/n_mape_excluded → conteo entero, mae/rmse → 1
  decimal; más una nota de auditoría que explica la escala.
- `_kpi_grid`: NaN → "—" (coherente con el N/D del panel);
  `forecast_room_nights` → conteo sin decimales; wmape/mape/bias en %.
- Tablas de accuracy y backtest: wmape/bias como porcentaje, mape con
  "%", n_obs como conteo; MAPE con NaN → "—" + marca † cuando aplica.
- Tests del working copy ya fijan estos formatos (`test_fmt_metrics_escala_por_clave`,
  `test_kpi_grid_nan_se_ve_como_raya`, y las aserciones actualizadas del
  grano semanal).

## Riesgos

- Cambios de formato rompen aserciones de tests que esperaban el formato
  viejo — ya actualizadas en `tests/test_reporting.py`.
- La nota solo aparece en el caso anomalía (reales sin unión): en la
  operación normal el reporte se ve idéntico al actual.
