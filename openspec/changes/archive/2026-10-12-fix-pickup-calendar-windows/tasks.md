# Tasks — fix-pickup-calendar-windows

- [x] 1.1 `src/features/pickup.py`: ventanas de pickup a días calendario (grilla diaria + ffill + incrementos); aceleración 7d vs 7d previos; volatilidad 7d; cumulative documentado por snapshot; NaT de snapshot fuera. Verificar: tests reescritos + nuevos.
- [x] 1.2 Tests: (a) cadencia esparcida (gaps de 30/15/10/4 días) → ventanas cuentan días reales; (b) era diaria → equivalencia con la fórmula por snapshots; (c) aceleración day-based (7d − 7d previos). Verificar: suite en verde.
- [x] 1.3 Corrida real de verificación: métricas IDÉNTICAS a la corrida anterior (MAPE 25.10%, WMAPE 0.1577, mismos campeones) — neutralidad de producción probada. Verificar: manifiestos comparados.
- [x] 1.4 README_SISTEMAS: nota del régimen de cadencia (diarios desde 2024, esparcidos antes; el loader/modelo maneja ambos). Commit.
