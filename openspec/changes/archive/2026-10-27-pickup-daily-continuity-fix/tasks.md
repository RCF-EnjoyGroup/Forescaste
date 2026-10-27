# Tasks — pickup-daily-continuity-fix

- [x] 1.1 Diagnóstico de la revisión openspec: los "VILLAS 332/CORIN 213/SJOSL 71/FIESTA 66 faltantes" provienen del PICKUP, no de demanda (demanda limpia: solo LIREL con su gap COVID real). El pickup es un registro disperso: stays sin reservas legítimamente no traen fila — exigirles continuidad diaria es falso positivo por diseño. (Corrige también el mecanismo de mi diagnóstico del 27-oct: el chequeo SÍ usa la grilla propia por propiedad; el problema es la semántica del pickup, no la grilla.)
- [x] 1.2 `src/validation/quality.py`: `check_missing_dates` respeta el `SourceKind` — continuidad diaria solo para series físicas (demanda/capacidad); el pickup solo reporta higiene (fechas ilegibles = error), nunca huecos por stays vacíos.
- [x] 1.3 Tests: `test_fechas_faltantes_pickup_es_disperso_no_advierte` (gaps legítimos → sin avisos), `test_fechas_ilegibles_pickup_si_reportan` (higiene aplica a todas las fuentes); el test existente de demanda/capacidad con hueco real sigue pasando.
- [x] 1.4 Verificado en datos reales: `validate_sources` sobre las tres fuentes actuales emite SOLO 'LIREL: 93 fechas faltantes' (el gap COVID, real) — los 4 falsos positivos del pickup desaparecen.
- [x] 1.5 Suite completa + spec data-ingestion actualizada + este change archivado + commit.
