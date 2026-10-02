# Hotel Revenue & Room-Nights Forecasting Pipeline

**Enjoy Costa Rica — Revenue Analytics**

Pipeline completo de pronóstico de series temporales para hoteles. El objetivo primario actual es **room nights** (`rooms_sold` — demanda), con el pipeline de ingresos monetarios (`revenue`/`room_revenue`) preservado y funcional.

## Overview

Este proyecto implementa un framework modular de forecasting con **protocolo honesto**: evaluación multi-paso real (sin usar reales del test como entradas), baseline SeasonalNaive en toda comparación, test Diebold-Mariano con corrección Newey-West, y métricas en unidades reales.

### Pipeline Room Nights (objetivo actual)

- **Target**: `rooms_sold` (noches de habitación vendidas)
- **Datos**: `data/hotel_roomnights_sample.csv` (ficticio, para desarrollo offline — identidades físicas exactas) o SQL real
- **Agregación NULL-safe**: filas financieras (NULL target) excluidas ANTES de imputar
- **Regla forward-known**: features monetarias excluidas; `available_rooms` (capacidad planeada) incluida
- **Techo de capacidad**: pronósticos recortados a capacidad disponible + ocupación implícita
- **Notebook**: `notebooks/hotel_roomnights_forecasting.ipynb` (+ reporte HTML)

### Modelos Implementados

| Familia | Modelo | Descripción |
|---------|--------|-------------|
| **Estadístico** | Prophet | Facebook Prophet con festivos de Costa Rica |
| **Estadístico** | SARIMAX | Auto ARIMA con exógenas |
| **Estadístico** | ETS | Holt-Winters con selección automática |
| **ML** | LightGBM | Gradient boosting con early stopping |
| **ML** | XGBoost | Gradient boosting regularizado |
| **ML** | CatBoost | Gradient boosting con categóricas nativas |
| **DL** | TimesFM | Foundation model de Google (zero-shot) |
| **DL** | PatchTST | Transformer con patching |

### Métricas de Evaluación

MAE, RMSE, MAPE, sMAPE, MASE, WAPE + test de Diebold-Mariano para significancia estadística.

## Project Structure

```
hotel-revenue-forecasting/
├── config.yaml                    # Configuración centralizada
├── pyproject.toml                 # Dependencias y build config
├── requirements.txt               # pip-compatible dependencies
├── notebooks/
│   └── hotel_revenue_forecasting.ipynb  # Notebook principal
├── src/forecasting/
│   ├── data/                      # Data loading & validation
│   ├── eda/                       # Exploratory data analysis
│   ├── preprocessing/             # Feature engineering & splits
│   ├── models/
│   │   ├── statistical/           # Prophet, SARIMAX, ETS
│   │   ├── ml/                    # LightGBM, XGBoost, CatBoost
│   │   └── dl/                    # TimesFM, PatchTST
│   ├── evaluation/                # Metrics & comparison
│   └── forecasting/               # Future forecast generation
├── tests/                         # Unit tests
├── README.md
├── ASSUMPTIONS.md
└── CHANGELOG.md
```

## Setup

### 1. Create virtual environment

```bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
# .venv\Scripts\activate   # Windows
```

### 2. Install dependencies

```bash
pip install -r requirements.txt

# For development
pip install -e ".[dev]"

# For deep learning models (optional)
pip install -e ".[dl]"
```

### 3. Configure

Edit `config.yaml` to set:
- SQL connection (or use CSV fallback)
- Model hyperparameters
- Split ratios
- Forecast horizon

### 4. Run

```bash
# Open the notebook
jupyter notebook notebooks/hotel_revenue_forecasting.ipynb
```

## Data Format

The pipeline expects data from a SQL query combining two tables:

```sql
(SELECT * FROM test_enjoy_fac_hotel)
UNION ALL
(SELECT * FROM enjoy_fac_hotel_financial)
```

**Required columns:**
- `date` — Transaction date
- `hotel_id` — Hotel identifier
- `revenue` — Total revenue
- `room_revenue` — Room-specific revenue

**Optional columns:** `rooms_sold`, `available_rooms`, `occupancy_rate`, `adr`, `revpar`

## Key Design Decisions

1. **Temporal splits only** — No random splits; 70/15/15 train/val/test with 7-day gap
2. **No data leakage** — All lag/rolling features shifted by 1; scaler fitted on train only
3. **Modular package** — Reusable components in `src/forecasting/`
4. **Notebook-first** — Deliverable is the Jupyter notebook; code in package for testing

## License

MIT
