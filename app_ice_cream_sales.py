"""
╔════════════════════════════════════════════════════════════════╗
║    Ice Cream Sales Forecasting Application — Group IV ALK 2026 ║
║                                                                ║
║  Advanced weekly ice cream sales forecasting system based on   ║
║  Machine Learning (XGBoost + Prophet) with real-time weather  ║
║  data integration (Open-Meteo API) and smart calendar features.║
╚════════════════════════════════════════════════════════════════╝
"""

import io
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime, timedelta
from pathlib import Path
import warnings
import requests
import pickle
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_percentage_error, mean_absolute_error, r2_score

try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False

import holidays

warnings.filterwarnings("ignore")

# ============================================================================
# SECTION 1: STREAMLIT CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Ice Cream Sales Forecasting",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SECTION 2: CONSTANTS AND FEATURE DEFINITIONS
# ============================================================================

# Weather, holiday, time, and lag features
WEATHER_FEATURES = [
    "Temp_Max_C", "Temp_Max_Peak_C", "heat_wave", "first_warm_week",
    "post_heat_wave", "Temp_Mean_C", "Rain_Sum_mm", "Rain_Days",
    "Wind_Speed_10m_Mean_kmh"
]

HOLIDAY_FEATURES = [
    "is_holiday", "days_to_holiday", "days_since_holiday",
    "easter_week", "may_long_weekend", "school_holiday"
]

TIME_FEATURES = ["ISO_Week", "Month"]

LAG_FEATURES = ["lag_52", "lag_4", "ma4"]

MODEL_FEATURES = WEATHER_FEATURES + HOLIDAY_FEATURES + TIME_FEATURES + LAG_FEATURES

# Feature sets for impulse vs family products (Prophet)
PROPHET_IMPULSE_FEATURES = [
    "Temp_Max_C", "Temp_Max_Peak_C", "Rain_Days", "is_holiday",
    "easter_week", "may_long_weekend", "school_holiday"
]

PROPHET_FAMILY_FEATURES = [
    "Temp_Mean_C", "Temp_Max_Peak_C", "Rain_Sum_mm", "Wind_Speed_10m_Mean_kmh",
    "is_holiday", "easter_week", "may_long_weekend", "school_holiday"
]

# XGBoost parameters
XGBOOST_PARAMS = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}

# Geographic coordinates (Poland center)
LAT, LON = 51.9, 19.1

# Color palette
COLOR_RED = "#E8231A"
COLOR_YELLOW = "#FFB800"
COLOR_BLUE = "#1B4FCC"
COLOR_LIGHT = "#FFF8E1"
COLOR_GRAY = "#616161"

# Feature descriptions
FEATURE_DESCRIPTIONS = {
    "Temp_Max_C": "Max temperature (avg week) [°C]",
    "Temp_Max_Peak_C": "Max temperature (hottest day) [°C]",
    "Temp_Mean_C": "Mean temperature [°C]",
    "Rain_Sum_mm": "Total precipitation [mm]",
    "Rain_Days": "Days with rainfall",
    "Wind_Speed_10m_Mean_kmh": "Wind speed [km/h]",
    "heat_wave": "Heat wave (T_max > 30°C for ≥3 days)",
    "first_warm_week": "First warm week of season",
    "post_heat_wave": "Week immediately after heat wave",
    "is_holiday": "Week contains public holiday",
    "days_to_holiday": "Days to nearest holiday",
    "days_since_holiday": "Days since last holiday",
    "easter_week": "3-week Easter window",
    "may_long_weekend": "May long weekend (weeks 17–19)",
    "school_holiday": "School vacation / breaks",
    "ISO_Week": "ISO week number (1–52)",
    "Month": "Month (1–12)",
    "lag_52": "📅 Sales from 1 year ago (lag 52 weeks)",
    "lag_4": "📅 Sales from 1 month ago (lag 4 weeks)",
    "ma4": "📊 4-week moving average",
}

# ============================================================================
# SECTION 3: EXCEL LOADING FUNCTIONS
# ============================================================================

def list_sheets(file_bytes: bytes) -> list:
    """Returns list of sheet names from Excel file."""
    import openpyxl
    try:
        wb = openpyxl.load_workbook(io.BytesIO(file_bytes), read_only=True, data_only=True)
        names = wb.sheetnames
        wb.close()
        return names
    except Exception as e:
        st.error(f"❌ Error reading sheets: {e}")
        return []

@st.cache_data(show_spinner=False)
def load_sheet(file_bytes: bytes, sheet: str) -> pd.DataFrame:
    """Load sheet to DataFrame with caching."""
    try:
        return pd.read_excel(io.BytesIO(file_bytes), sheet_name=sheet)
    except Exception as e:
        st.error(f"❌ Error loading sheet '{sheet}': {e}")
        return pd.DataFrame()

# ============================================================================
# SECTION 4: DATA STANDARDIZATION
# ============================================================================

def standardize_data(df, col_date, col_product, col_sales, col_channel=None, col_metric=None,
                     filter_channel=None, filter_metric=None) -> pd.DataFrame:
    """
    Standardizes DataFrame columns to internal application names.
    
    Returns DataFrame with columns: Date, Product, Sales_Value, Channel_Group (optional)
    """
    df = df.copy()
    
    # Column mapping
    df["Date"] = pd.to_datetime(df[col_date], dayfirst=True, errors="coerce")
    df["Product"] = df[col_product].astype(str).str.strip()
    df["Sales_Value"] = pd.to_numeric(df[col_sales], errors="coerce")
    
    # Filter by channel and metric (optional)
    if col_channel and filter_channel:
        df = df[df[col_channel].astype(str).str.strip() == filter_channel].copy()
    
    if col_metric and filter_metric:
        df = df[df[col_metric].astype(str).str.strip() == filter_metric].copy()
    
    # Remove rows with missing values
    df = df.dropna(subset=["Date", "Product", "Sales_Value"])
    
    # Sort
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df[["Date", "Product", "Sales_Value"]]

# ============================================================================
# SECTION 5: WEATHER FETCHING
# ============================================================================

@st.cache_data(show_spinner=False)
def fetch_historical_weather(lat, lon, date_start, date_end):
    """
    Fetch weather data from Open-Meteo Archive API.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": date_start.strftime("%Y-%m-%d"),
        "end_date": date_end.strftime("%Y-%m-%d"),
        "daily": "temperature_2m_max,temperature_2m_mean,precipitation_sum,precipitation_days,wind_speed_10m_max",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "timezone": "Europe/Warsaw",
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        df = pd.DataFrame({
            "Date": pd.to_datetime(data["daily"]["time"]),
            "Temp_Max_C": data["daily"]["temperature_2m_max"],
            "Temp_Mean_C": data["daily"]["temperature_2m_mean"],
            "Rain_Sum_mm": data["daily"]["precipitation_sum"],
            "Rain_Days": data["daily"]["precipitation_days"],
            "Wind_Speed_10m_Mean_kmh": data["daily"]["wind_speed_10m_max"],
        })
        return df
    except Exception as e:
        st.error(f"❌ Error fetching historical weather: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def fetch_weather_forecast(lat, lon):
    """
    Fetch weather forecast from Open-Meteo Forecast API (16 days).
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_mean,precipitation_sum,precipitation_days,wind_speed_10m_max",
        "temperature_unit": "celsius",
        "wind_speed_unit": "kmh",
        "timezone": "Europe/Warsaw",
        "forecast_days": 16,
    }
    
    try:
        resp = requests.get(url, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        
        df = pd.DataFrame({
            "Date": pd.to_datetime(data["daily"]["time"]),
            "Temp_Max_C": data["daily"]["temperature_2m_max"],
            "Temp_Mean_C": data["daily"]["temperature_2m_mean"],
            "Rain_Sum_mm": data["daily"]["precipitation_sum"],
            "Rain_Days": data["daily"]["precipitation_days"],
            "Wind_Speed_10m_Mean_kmh": data["daily"]["wind_speed_10m_max"],
        })
        return df
    except Exception as e:
        st.error(f"❌ Error fetching weather forecast: {e}")
        return pd.DataFrame()

def aggregate_weather_weekly(df_weather):
    """
    Aggregate daily weather data to weekly level.
    Adds features: heat_wave, first_warm_week, post_heat_wave.
    """
    df = df_weather.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["ISO_Week"] = df["Date"].dt.isocalendar().week
    df["Year"] = df["Date"].dt.isocalendar().year
    
    # Basic aggregation
    df_w = df.groupby(["Year", "ISO_Week"], as_index=False).agg({
        "Temp_Max_C": "mean",
        "Temp_Mean_C": "mean",
        "Rain_Sum_mm": "sum",
        "Rain_Days": "sum",
        "Wind_Speed_10m_Mean_kmh": "mean",
    })
    
    # Calculate Temp_Max_Peak_C
    df_w["Temp_Max_Peak_C"] = df.groupby(["Year", "ISO_Week"])["Temp_Max_C"].max().values
    
    # Seasonal features
    df_w["heat_wave"] = (df_w["Temp_Max_Peak_C"] > 30).astype(int)
    
    # first_warm_week — first where Temp_Mean > 15°C
    df_w["first_warm_week"] = 0
    for year in df_w["Year"].unique():
        year_data = df_w[df_w["Year"] == year].sort_values("ISO_Week")
        first_warm_idx = (year_data["Temp_Mean_C"] > 15).idxmax()
        if first_warm_idx in df_w.index:
            df_w.loc[first_warm_idx, "first_warm_week"] = 1
    
    # post_heat_wave — week after heat wave
    df_w["post_heat_wave"] = 0
    heat_waves = df_w[df_w["heat_wave"] == 1].index.tolist()
    for hw_idx in heat_waves:
        next_idx = df_w.index[df_w.index > hw_idx].min()
        if pd.notna(next_idx):
            df_w.loc[next_idx, "post_heat_wave"] = 1
    
    return df_w.sort_values(["Year", "ISO_Week"]).reset_index(drop=True)

# ============================================================================
# SECTION 6: CALENDAR FEATURES
# ============================================================================

def add_calendar_features(df, year_range=None):
    """
    Add calendar features to DataFrame:
    - is_holiday, days_to_holiday, days_since_holiday
    - easter_week, may_long_weekend, school_holiday
    """
    df = df.copy()
    
    if "Date" not in df.columns and "ISO_Week" in df.columns:
        # If we have ISO_Week, reconstruct approximate date
        df["Date"] = df.apply(
            lambda row: datetime.strptime(f"{int(row['Year'])}-W{int(row['ISO_Week'])}-1", "%Y-W%W-%w"),
            axis=1
        )
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Polish holidays
    pl_holidays = holidays.Poland(years=year_range or range(df["Date"].dt.year.min(), df["Date"].dt.year.max() + 1))
    
    # Map holidays to weeks
    df["is_holiday"] = df["Date"].apply(
        lambda d: 1 if any(pd.to_datetime(d.date()) == pd.to_datetime(h) for h in pl_holidays.keys()) else 0
    )
    
    # Distance to/from holidays
    holidays_dates = sorted(pl_holidays.keys())
    df["days_to_holiday"] = df["Date"].apply(
        lambda d: min([(h - d).days for h in holidays_dates if (h - d).days > 0], default=999)
    )
    df["days_since_holiday"] = df["Date"].apply(
        lambda d: min([(d - h).days for h in holidays_dates if (d - h).days >= 0], default=999)
    )
    
    # Easter (3-week window)
    df["easter_week"] = 0
    for year in df["Date"].dt.year.unique():
        try:
            easter_date = holidays.Easter(year=year)
            if easter_date:
                easter_iso_week = easter_date.isocalendar()[1]
                mask = (df["Year"] == year) & (df["ISO_Week"].between(easter_iso_week - 1, easter_iso_week + 1))
                df.loc[mask, "easter_week"] = 1
        except:
            pass
    
    # May long weekend (weeks 17–19)
    df["may_long_weekend"] = df["ISO_Week"].between(17, 19).astype(int)
    
    # School holidays
    df["school_holiday"] = 0
    summer_mask = (df["ISO_Week"] >= 24) & (df["ISO_Week"] <= 35)  # mid June - end August
    winter_mask = (df["ISO_Week"] >= 3) & (df["ISO_Week"] <= 8)    # mid January - end February
    autumn_mask = (df["ISO_Week"] >= 42) & (df["ISO_Week"] <= 43)  # October
    
    df.loc[summer_mask | winter_mask | autumn_mask, "school_holiday"] = 1
    
    return df

# ============================================================================
# SECTION 7: SALES LAGS
# ============================================================================

def add_sales_lags(df):
    """
    Add sales lags (lag_52, lag_4, ma4).
    """
    df = df.copy()
    df = df.sort_values(["Product", "Year", "ISO_Week"]).reset_index(drop=True)
    
    df["lag_52"] = df.groupby("Product")["Sales_Value"].shift(52)
    df["lag_4"] = df.groupby("Product")["Sales_Value"].shift(4)
    df["ma4"] = df.groupby("Product")["Sales_Value"].rolling(window=4, min_periods=1).mean().values
    
    return df

# ============================================================================
# SECTION 8: MODEL TRAINING
# ============================================================================

def train_xgboost_with_ci(df_product, features, quantiles=[0.1, 0.5, 0.9]):
    """
    Train XGBoost with quantiles (confidence intervals p10/p50/p90).
    Hyperparameter tuning with TimeSeriesSplit.
    """
    df = df_product.dropna(subset=features + ["Sales_Value"]).copy()
    
    if len(df) < 30:
        return None, None, None, None
    
    # Prepare data
    X = df[features].values
    y = df["Sales_Value"].values
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = XGBOOST_PARAMS.copy()
    best_score = float("inf")
    
    # Grid search (simplified)
    for max_depth in [2, 3, 4]:
        for min_child_weight in [3, 8]:
            params = best_params.copy()
            params["max_depth"] = max_depth
            params["min_child_weight"] = min_child_weight
            
            scores = []
            for train_idx, test_idx in tscv.split(X):
                X_train, X_test = X[train_idx], X[test_idx]
                y_train, y_test = y[train_idx], y[test_idx]
                
                model = xgb.XGBRegressor(**params)
                model.fit(X_train, y_train, verbose=0)
                score = mean_absolute_percentage_error(y_test, model.predict(X_test))
                scores.append(score)
            
            avg_score = np.mean(scores)
            if avg_score < best_score:
                best_score = avg_score
                best_params = params
    
    # Train final models (3 quantiles)
    models = {}
    evals = {}
    
    for q in quantiles:
        params = best_params.copy()
        params["objective"] = "reg:quantilehubererror"
        params["quantile_alpha"] = q
        
        model = xgb.XGBRegressor(**params)
        model.fit(X, y, verbose=0)
        models[q] = model
        
        y_pred = model.predict(X)
        mape = mean_absolute_percentage_error(y, y_pred)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        evals[q] = {"MAPE": mape, "MAE": mae, "R2": r2}
    
    # Feature importance
    fi = models[0.5].get_booster().get_score(importance_type="weight")
    fi = {features[int(k.split("_")[1])]: v for k, v in fi.items() if k.startswith("f_")}
    
    return models, evals, fi, best_params

def train_prophet(df_product, regressor_features):
    """
    Train Prophet model with multiplicative seasonality and regressors.
    """
    if not PROPHET_AVAILABLE:
        st.warning("⚠️ Prophet not available. Install: pip install prophet")
        return None, None
    
    df = df_product.dropna(subset=regressor_features + ["Sales_Value"]).copy()
    
    if len(df) < 30:
        return None, None
    
    # Prepare data
    df["ds"] = pd.to_datetime(df[["Year", "ISO_Week"]].apply(
        lambda x: datetime.strptime(f"{int(x['Year'])}-W{int(x['ISO_Week'])}-1", "%Y-W%W-%w"), axis=1
    ))
    df["y"] = df["Sales_Value"]
    
    try:
        model = Prophet(interval_width=0.8, yearly_seasonality=True, seasonality_mode="multiplicative")
        
        # Add regressors
        for reg in regressor_features:
            if reg in df.columns:
                model.add_regressor(reg)
        
        model.fit(df[["ds", "y"] + regressor_features])
        
        # Evaluate on full dataset
        forecast = model.make_future_dataframe(periods=0, freq="W")
        forecast = model.predict(forecast[forecast["ds"].dt.day <= 7])
        
        y_actual = df["y"].values
        y_pred = forecast["yhat"].values[:len(y_actual)]
        
        mape = mean_absolute_percentage_error(y_actual, np.maximum(y_pred, 1))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        return model, {"MAPE": mape, "MAE": mae, "R2": r2}
    except Exception as e:
        st.warning(f"⚠️ Error training Prophet: {e}")
        return None, None

# ============================================================================
# SECTION 9: MAIN APPLICATION (UI)
# ============================================================================

def main():
    st.markdown("""
    <style>
        .main { max-width: 100%; }
        .stTabs [data-baseweb="tab-list"] button { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🍦 Ice Cream Sales Forecasting")
    st.markdown("**Advanced ML forecasting system** — XGBoost + Prophet | Open-Meteo API | Calendar Intelligence")
    
    # ========== SIDEBAR: DATA CONFIGURATION ==========
    with st.sidebar:
        st.header("📥 Configuration")
        
        uploaded_file = st.file_uploader("Upload Excel file with sales data", type=["xlsx", "csv"])
        
        if uploaded_file:
            file_bytes = uploaded_file.read()
            
            # Sheet selection
            sheets = list_sheets(file_bytes)
            if sheets:
                selected_sheet = st.selectbox("Select sheet", sheets)
                df_raw = load_sheet(file_bytes, selected_sheet)
                
                if not df_raw.empty:
                    st.success(f"✅ Loaded {len(df_raw)} rows")
                    
                    # Column mapping
                    st.subheader("Column Mapping")
                    cols = df_raw.columns.tolist()
                    
                    col_date = st.selectbox("Date column", cols, index=0)
                    col_product = st.selectbox("Product column", cols, index=min(1, len(cols)-1))
                    col_sales = st.selectbox("Sales [qty] column", cols, index=min(2, len(cols)-1))
                    
                    # Standardization
                    df_std = standardize_data(df_raw, col_date, col_product, col_sales)
                    
                    if not df_std.empty:
                        # Weekly aggregation
                        df_std["Year"] = df_std["Date"].dt.isocalendar().year
                        df_std["ISO_Week"] = df_std["Date"].dt.isocalendar().week
                        
                        df_weekly = df_std.groupby(["Year", "ISO_Week", "Product"], as_index=False)["Sales_Value"].sum()
                        
                        # Fetch weather
                        st.subheader("☀️ Weather")
                        if st.button("Fetch Historical Weather"):
                            date_min = pd.to_datetime(df_std["Date"]).min()
                            date_max = pd.to_datetime(df_std["Date"]).max()
                            
                            with st.spinner("Fetching..."):
                                df_weather = fetch_historical_weather(LAT, LON, date_min, date_max)
                            
                            if not df_weather.empty:
                                df_weather_w = aggregate_weather_weekly(df_weather)
                                st.session_state.df_weather_w = df_weather_w
                                st.success("✅ Weather fetched")
        
        st.divider()
        st.markdown("**Group IV AIFINC 2026 · ALK**")
    
    # ========== MAIN TABS ==========
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Data", "🤖 Training", "🔮 Forecast", "📈 Recommendations", "📉 Backtest"])
    
    with tab1:
        st.header("📊 Loaded Data")
        if "df_weather_w" in st.session_state:
            st.write("✅ Weather loaded")
        else:
            st.info("ℹ️ Upload data and fetch weather in the left panel")
    
    with tab2:
        st.header("🤖 Train Models")
        st.info("ℹ️ XGBoost and Prophet training module")
    
    with tab3:
        st.header("🔮 Forecast for 8 Weeks")
        st.info("ℹ️ Future sales forecasting module")
    
    with tab4:
        st.header("📈 Strategic Recommendations")
        st.info("ℹ️ Feature analysis and ABC/XYZ classification")
    
    with tab5:
        st.header("📉 Actual vs Forecast")
        st.info("ℹ️ Backtesting and model accuracy assessment")

if __name__ == "__main__":
    main()
