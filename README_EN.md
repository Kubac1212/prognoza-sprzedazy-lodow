# 🍦 Ice Cream Sales Forecasting

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green.svg)](https://xgboost.readthedocs.io/)

Advanced application for forecasting weekly ice cream sales based on **Machine Learning models** (XGBoost + Prophet), **real-time weather data** (Open-Meteo API), and intelligently selected **calendar features**.

The application runs **in your browser** using Streamlit — no installation required beyond Python.

---

## 📋 Table of Contents

- [✨ Features](#-features)
- [🚀 Quick Start](#-quick-start)
- [📋 Requirements](#-requirements)
- [🔧 Installation](#-installation)
- [📖 Usage](#-usage)
- [🏗️ Architecture](#-architecture)
- [📦 Dependencies](#-dependencies)
- [🌐 API](#-api)
- [📊 Project Structure](#-project-structure)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [✍️ Authors](#-authors)

---

## ✨ Features

✅ **Hybrid models** — XGBoost for accuracy + Prophet for seasonal trends  
✅ **Confidence intervals** — Quantiles p10/p50/p90 for each forecast  
✅ **Smart recommendations** — Automated order suggestions with risk assessment  
✅ **Feature analysis** — Feature importance and ABC/XYZ assortment classification  
✅ **Real-time weather** — Open-Meteo API integration (no API key required)  
✅ **Calendar intelligence** — Polish holidays, Easter, school breaks  
✅ **Manual corrections** — Adjust forecasts with automatic saving  
✅ **Backtesting** — Compare actual vs predicted sales  

---

## 🚀 Quick Start

### Windows

```bash
# 1. Clone repository
git clone https://github.com/your-username/ice-cream-sales-forecasting.git
cd ice-cream-sales-forecasting

# 2. Run script
run.bat
```

**That's it!** The app will install dependencies and open in your browser.

### macOS / Linux

```bash
# 1. Clone repository
git clone https://github.com/your-username/ice-cream-sales-forecasting.git
cd ice-cream-sales-forecasting

# 2. Run script
bash run.sh
```

App will be available at **http://localhost:8501**

---

## 📋 Requirements

| Requirement | Version |
|-------------|---------|
| Python | 3.10+ |
| Internet connection | Required (Open-Meteo API) |
| Disk space | ~500 MB |
| RAM | Minimum 2 GB (recommended 4 GB) |

---

## 🔧 Installation

### Manual Installation (Advanced Users)

```bash
# 1. Clone repository
git clone https://github.com/your-username/ice-cream-sales-forecasting.git
cd ice-cream-sales-forecasting

# 2. Create virtual environment
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run application
streamlit run app_prognoza_lodow.py
```

### Troubleshooting

**Problem**: "Python not found"  
**Solution**: Ensure Python is in PATH → [Python Installation Guide](https://docs.python.org/3/using/windows.html#finding-the-python-executable)

**Problem**: Prophet library errors  
**Solution**: `pip install --upgrade cmdstanpy`

**Problem**: No internet / Open-Meteo API unavailable  
**Solution**: Check your internet connection; API is free and requires no registration

---

## 📖 Usage

### Workflow

1. **Tab 1 — Data Upload**
   - Upload Excel file with sales data (columns: `date`, `product`, `sales_qty`)
   - App automatically aggregates to weeks and merges with weather

2. **Tab 2 — Train Models**
   - Select impulse/family products (for optimal Prophet setup)
   - Click "Train XGBoost" and "Train Prophet"
   - Review metrics: MAPE, MAE, R² and feature importance

3. **Tab 3 — Forecast**
   - Forecast for T+1…T+8 weeks
   - Scenarios: conservative/average/optimistic
   - Order recommendation with stockout risk assessment

4. **Tab 4 — Recommendations**
   - Global feature importance (average across all models)
   - ABC (volume) / XYZ (volatility) assortment classification

5. **Tab 5 — Actual vs Forecast**
   - Upload actual sales
   - Compare model predictions with reality (MAPE, MAE, bias)

---

## 🏗️ Architecture

### Data Flow

```
┌──────────────────────────────────────────────────────┐
│ User uploads Excel file (date, product, sales)      │
└────────────────────┬─────────────────────────────────┘
                     ↓
         ┌───────────────────────────┐
         │ Aggregate to weeks        │
         │ + Fetch historical weather│
         └────────────┬──────────────┘
                      ↓
       ┌──────────────────────────┐  ┌──────────────────┐
       │ Add calendar features    │  │ Add weather data │
       │ (holidays, vacations)    │  │ (temp, rain)     │
       └──────────┬───────────────┘  └────────┬─────────┘
                  └─────────────┬──────────────┘
                                ↓
                  ┌──────────────────────────────┐
                  │ Train ML models:             │
                  │ • XGBoost (3 quantiles)      │
                  │ • Prophet (seasonality)      │
                  │ • TimeSeriesSplit CV         │
                  │ • Feature importance         │
                  └────────┬─────────────────────┘
                           ↓
            ┌──────────────────────────────────┐
            │ Forecast T+1…T+8 (Tab 3)        │
            │ • Fetch weather forecast        │
            │ • Create scenarios              │
            │ • Order recommendation          │
            └──────────────────────────────────┘
```

### Key Components

| Module | Responsibility |
|--------|----------------|
| `pobierz_pogode_*` | Open-Meteo Archive/Forecast API integration |
| `dodaj_cechy_kalendarza` | Polish holidays, school breaks, Easter, May holidays |
| `dodaj_lagi_sprzedazy` | Yearly (lag_52), monthly (lag_4), MA4 sales lags |
| `trenuj_z_ci` | XGBoost with hyperparameter tuning (TimeSeriesSplit) |
| `trenuj_prophet` | Prophet with multiplicative seasonality and regressors |
| `prognozuj_*` | Generate forecasts and confidence intervals |

---

## 📦 Dependencies

```txt
streamlit==1.57.0           # UI and interactivity
pandas==3.0.2              # Data processing
numpy==2.4.4               # Numerical operations
xgboost==3.2.0             # Quantile regression models
prophet==1.3.0             # Time series + seasonality
scikit-learn==1.8.0        # TimeSeriesSplit for CV
plotly==6.7.0              # Interactive charts
holidays==0.96             # Polish holidays
requests==2.33.1           # API requests
openpyxl==3.1.5            # Excel file handling
```

Full list: see `requirements.txt`

---

## 🌐 API

App uses **free, public APIs**:

### Open-Meteo Archive API
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Purpose**: Historical weather data (training)
- **No API key required** ✅

### Open-Meteo Forecast API
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Purpose**: 16-day weather forecast
- **No API key required** ✅

### Optional
- **OpenWeatherMap** (optional) — add your API key in sidebar
- **Polish holidays** — `holidays` library (offline)

---

## 📊 Project Structure

```
ice-cream-sales-forecasting/
│
├── app_prognoza_lodow.py        # Main application (~2700 lines)
│                                # Sections:
│                                # - UI (Streamlit)
│                                # - Weather fetching
│                                # - Feature engineering
│                                # - Model training
│                                # - Forecast generation
│                                # - Recommendations
│
├── requirements.txt             # Python dependencies with versions
│
├── run.bat                      # Windows startup script
│                                # - Checks Python 3.10+
│                                # - Creates venv
│                                # - Installs dependencies
│                                # - Runs Streamlit
│
├── run.sh                       # macOS/Linux startup script
│                                # Same as run.bat
│
├── README.md                    # Documentation (this file)
│
├── LICENSE                      # MIT License
│
├── .gitignore                   # Standard Python ignores
│
└── manual_corrections.xlsx      # Auto-created after first correction
                                 # Stores user edits between sessions
```

---

## 🎯 Usage Example

### Prepare Data

Create Excel file (`sales.xlsx`) with columns:

```
| date       | product        | sales_qty |
|------------|----------------|-----------|
| 2024-01-01 | Vanilla ice    | 150       |
| 2024-01-02 | Vanilla ice    | 160       |
| 2024-01-01 | Chocolate ice  | 80        |
| ...        | ...            | ...       |
```

### Run

1. **Windows**: Double-click `run.bat`
2. **macOS/Linux**: `bash run.sh`
3. Upload Excel in Tab 1
4. Train models in Tab 2
5. Review forecasts in Tab 3
6. Act on recommendations in Tab 4

---

## 🚀 Performance

### Accuracy Metrics

App reports for each product:

- **MAPE** (Mean Absolute Percentage Error) — % error
- **MAE** (Mean Absolute Error) — average error magnitude
- **R²** — explained variance of model

### Operation Times

| Operation | Time |
|-----------|------|
| Load data (1000 rows) | ~1 s |
| Fetch historical weather | ~5-10 s |
| Train XGBoost (10 products) | ~30-60 s |
| Train Prophet (10 products) | ~20-40 s |
| Generate forecast T+8 | ~5-10 s |

---

## 🤝 Contributing

Contributions are welcome! To propose changes:

1. **Fork** the repository
2. Create branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open **Pull Request**

### Development Roadmap

- [ ] Shopify / WooCommerce API integration
- [ ] Google Sheets export
- [ ] Volatility modeling (GARCH)
- [ ] Storage cost optimization
- [ ] Stockout prediction (alert system)
- [ ] Multi-location support

---

## 📄 License

Project is licensed under **MIT License**. See `LICENSE` file for details.

---

## ✍️ Authors

- **Grupa IV** — AIFINC 2026 · ALK
- **Kuba** — Lead Developer

---

## 📞 Support

If you encounter issues or have questions:

1. **Issues** — open [GitHub Issue](https://github.com/your-username/ice-cream-sales-forecasting/issues)
2. **Documentation** — read [Usage](#-usage) section
3. **Email** — your-email@example.com

---

## 🙏 Acknowledgments

- **Open-Meteo** — for free weather API
- **Streamlit** — for amazing framework
- **XGBoost & Prophet** — for powerful ML models
- Python & open-source community!

---

## 📈 Roadmap (v2.0)

- ✅ **v1.0** — Core XGBoost + Prophet
- 🔄 **v1.1** — Hyperparameter optimization (Q4 2026)
- 🚀 **v2.0** — REST API + PostgreSQL database
- 📱 **v2.1** — Mobile app (React Native)

---

<div align="center">

**[⬆ Back to top](#-ice-cream-sales-forecasting)**

Made with ❤️ by Grupa IV AIFINC 2026

</div>
