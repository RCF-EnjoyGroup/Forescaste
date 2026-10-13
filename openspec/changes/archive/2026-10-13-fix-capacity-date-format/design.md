# Design — fix-capacity-date-format

## Decisions

### D1 — Detección por patrón con evidencia, no por conteo (ni por configuración)
El primer diseño comparaba conteos de parseo entre default y ``dayfirst=True``.
Los experimentos lo invalidaron: **``dayfirst=True`` corrompe fechas ISO**
("2026-10-05" se lee como 2026-05-10 — dateutil trata el guion como
ambiguo), así que no es un superset seguro. La detección final clasifica
la columna por patrón de strings:

* ISO puro (≥90%): ``format='%Y-%m-%d'`` estricto — jamás se cruza.
* Barras (≥90%): evidencia explícita — algún día > 12 en la PRIMERA
  posición prueba DD/MM; en la SEGUNDA prueba MM/DD (evidencia mutuamente
  excluyente: en MM/DD la primera posición es mes ≤ 12 siempre).
* Columna 100% ambigua (todo día ≤ 12, p. ej. un ``events.csv`` cargado
  a mano): el conteo no prueba nada → ``prefer_dayfirst`` desempata por
  el estándar del negocio. Las fuentes diarias (capacity) siempre
  tienen días 13-31, así que el caso no ocurre en las tres principales.
* Mixto/otro: comparación por conteo como mejor esfuerzo (documentado).

Sin configuración nueva: el motor se adapta a la fuente.

### D2 — El fix vive solo en la frontera de normalización
Los consumidores internos (features, backtest, evaluación, reporting) reciben frames ya datetime; sus ``pd.to_datetime`` son no-ops. Los puntos de entrada del string crudo son exactamente cinco: ``_normalize_demand/_capacity/_pickup`` (core), ``SourceSchema.coerce`` (pipeline de ingesta), ``_is_date_like`` (detección), los checks de calidad que leen frames crudos, y el loader de ``events.csv``.

### D3 — Filas completamente vacías = higiene, no corrupción
Las 6 líneas ``,,`` del capacity actual (edit accidental 14:36, posterior a la corrida de las 09:32) son ruido de exportación: se reportan como ``empty_rows`` (warning) y se filtran antes de los demás checks. Sin esto, ``duplicated`` las cuenta como duplicados NaN==NaN (error → gate fail) y el análisis de huecos crashea en ``date_range(NaT, NaT)``. Un grupo con filas reales pero fechas ilegibles sigue siendo **error** (corrupción genuina), emitido sin crash vía guarda por grupo. La detección contempla el placeholder ``'nan'`` stringificado que deja ``_apply_aliases``.

### D4 — El gate de la corrida real ve el capacity CRUDO
``run_real`` pasaba el capacity pre-normalizado al runner: el ``dropna``
de la normalización eliminaba la basura ANTES del gate y el reporte de
producción nunca la mostraba. Ahora ``load_real_sources`` también
devuelve el capacity crudo (post alias/exclusión) y ``run_end_to_end``
lo recibe a él: el gate reporta las filas vacías con su conteo real de
registros, y el runner normaliza de forma idempotente (mismo forecast).

### D5 — Sin cambios de semántica del gate
Las severidades quedan igual; el fix hace que los checks operen sobre el 100% de la serie. ``check_rn_over_capacity`` pasa de 3,499 a 8,851 filas cruzadas: los falsos positivos de LIREL (10) y SJOSL (7, 6.5%) desaparecen y queda el hallazgo real: 5 días de FIESTA en ago-2026 con RN > 408 (0.5% → warning, el gate pasa).

### D6 — El "accidente" era el orden de filas del export anterior
Hallazgo forense: la corrida de las 09:32 reportó capacity con 30,289
registros — parseo ~100% CORRECTO. El archivo viejo, por el orden de sus
filas, hacía que pandas INFIRIERA DD/MM desde el primer elemento (su
primera fecha tenía día > 12). El re-export de las 14:36 empieza con
``CORIN,01/01/2013`` (ambiguo → pandas infiere MM/DD → 39% de fechas,
día/mes cruzados en el resto). Es decir: el motor estuvo correcto por
accidente de ordenamiento, y el nuevo export ROMPE el accidente. El fix
hace el resultado determinista e independiente del orden de filas; los
deltas de métricas exactamente 0.00 entre ambas corridas lo confirman.

## Risks / Trade-offs

- [Columna mezcla ISO y barras] — la clasificación ≥90% manda la minoría a NaT (reportado como ilegible por calidad); la detección por conteo queda de respaldo. Los datos reales no mezclan; el contrato pide un formato por columna.
- [Un MM/DD real con toda fecha ambigua] — imposible de distinguir de DD/MM ambiguo sin metadatos; ``prefer_dayfirst`` solo se usa en events (estándar del negocio) y el default sigue siendo conservador. No existe tal fuente en los datos.
- [El reporte de calidad cambia] — issues recalculados con fechas correctas (SJOSL: 371 días capacity=0 en 2025 = cierre real; 4 cambios de capacidad = apertura CORIN 2013-09-23 y rampa SJOSL ene-2026). Es el objetivo; verificado con corrida real.
- [Métricas] — deltas 0.00 en las seis propiedades: los techos del horizonte son constantes y la demanda está debajo; el valor del fix es eliminar la mina latente (cualquier capacidad cambiante futura habría recibido el techo equivocado), no mover números.
