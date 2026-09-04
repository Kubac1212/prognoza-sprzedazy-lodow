"""
╔════════════════════════════════════════════════════════════════╗
║     Aplikacja prognozy sprzedaży lodów — Grupa IV ALK 2026    ║
║                                                                ║
║  Zaawansowany system prognozowania tygodniowej sprzedaży      ║
║  lodów oparty na modelach Machine Learning (XGBoost+Prophet)  ║
║  z integracją danych pogodowych (Open-Meteo API) oraz         ║
║  inteligentnie wybranymi cechami kalendarza.                  ║
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
# SEKCJA 1: KONFIGURACJA STREAMLIT
# ============================================================================

st.set_page_config(
    page_title="Prognoza Sprzedaży Lodów",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================================
# SEKCJA 2: STAŁE I DEFINICJE CECH
# ============================================================================

# Stałe pogodowe, świąteczne, czasowe i lagi
CECHY_POGODOWE = [
    "Temp_Max_C", "Temp_Max_Peak_C", "heat_wave", "first_warm_week",
    "post_heat_wave", "Temp_Mean_C", "Rain_Sum_mm", "Rain_Days",
    "Wind_Speed_10m_Mean_kmh"
]

CECHY_SWIATECZNE = [
    "is_holiday", "days_to_holiday", "days_since_holiday",
    "easter_week", "may_long_weekend", "school_holiday"
]

CECHY_CZASOWE = ["ISO_Week", "Month"]

CECHY_LAGI = ["lag_52", "lag_4", "ma4"]

CECHY_MODELU = CECHY_POGODOWE + CECHY_SWIATECZNE + CECHY_CZASOWE + CECHY_LAGI

# Zestawy cech dla produktów impulsowych i familijnych (Prophet)
CECHY_PROPHET_IMPULSOWE = [
    "Temp_Max_C", "Temp_Max_Peak_C", "Rain_Days", "is_holiday",
    "easter_week", "may_long_weekend", "school_holiday"
]

CECHY_PROPHET_FAMILIJNE = [
    "Temp_Mean_C", "Temp_Max_Peak_C", "Rain_Sum_mm", "Wind_Speed_10m_Mean_kmh",
    "is_holiday", "easter_week", "may_long_weekend", "school_holiday"
]

# Parametry XGBoost
PARAMETRY_XGB = {
    "n_estimators": 200,
    "learning_rate": 0.05,
    "max_depth": 3,
    "min_child_weight": 5,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "verbosity": 0,
}

# Współrzędne geograficzne (środek Polski)
LAT, LON = 51.9, 19.1

# Paleta kolorów
K_RED = "#E8231A"
K_YELLOW = "#FFB800"
K_BLUE = "#1B4FCC"
K_LIGHT = "#FFF8E1"
K_GRAY = "#616161"

# Słownik opisów cech
OPIS_CECH = {
    "Temp_Max_C": "Temperatura maks. (śr. tygodnia) [°C]",
    "Temp_Max_Peak_C": "Temperatura maks. (najgorętszy dzień) [°C]",
    "Temp_Mean_C": "Temperatura średnia [°C]",
    "Rain_Sum_mm": "Suma opadów [mm]",
    "Rain_Days": "Liczba dni z opadami",
    "Wind_Speed_10m_Mean_kmh": "Prędkość wiatru [km/h]",
    "heat_wave": "Fala upałów (T_max > 30°C przez ≥3 dni)",
    "first_warm_week": "Pierwszy ciepły tydzień sezonu",
    "post_heat_wave": "Tydzień bezpośrednio po fali upałów",
    "is_holiday": "Tydzień zawiera święto ustawowe",
    "days_to_holiday": "Dni do najbliższego święta",
    "days_since_holiday": "Dni od ostatniego święta",
    "easter_week": "3-tygodniowe okno wokół Wielkanocy",
    "may_long_weekend": "Majówka (tygodnie 17–19)",
    "school_holiday": "Wakacje/ferie szkolne",
    "ISO_Week": "Numer tygodnia ISO (1–52)",
    "Month": "Miesiąc (1–12)",
    "lag_52": "📅 Sprzedaż rok temu (lag 52 tygodnie)",
    "lag_4": "📅 Sprzedaż miesiąc temu (lag 4 tygodnie)",
    "ma4": "📊 Średnia krocząca 4 tygodnie",
}

# ============================================================================
# SEKCJA 3: FUNKCJE DO WCZYTYWANIA EXCEL
# ============================================================================

def lista_arkuszy(plik_bytes: bytes) -> list:
    """Zwraca listę nazw arkuszy z pliku Excel."""
    import openpyxl
    try:
        wb = openpyxl.load_workbook(io.BytesIO(plik_bytes), read_only=True, data_only=True)
        names = wb.sheetnames
        wb.close()
        return names
    except Exception as e:
        st.error(f"❌ Błąd odczytu arkuszy: {e}")
        return []

@st.cache_data(show_spinner=False)
def wczytaj_arkusz(plik_bytes: bytes, arkusz: str) -> pd.DataFrame:
    """Wczytuje arkusz do DataFrame z cache'owaniem."""
    try:
        return pd.read_excel(io.BytesIO(plik_bytes), sheet_name=arkusz)
    except Exception as e:
        st.error(f"❌ Błąd wczytania arkusza '{arkusz}': {e}")
        return pd.DataFrame()

# ============================================================================
# SEKCJA 4: STANDARYZACJA DANYCH
# ============================================================================

def standaryzuj(df, kol_data, kol_produkt, kol_sprzedaz, kol_kanal=None, kol_metryka=None,
                filtr_kanal=None, filtr_metryka=None) -> pd.DataFrame:
    """
    Standaryzuje kolumny DataFrame do wewnętrznych nazw aplikacji.
    
    Zwraca DataFrame z kolumnami: Date, Assortment, Sales_Value, Channel_Group (opcjonalnie)
    """
    df = df.copy()
    
    # Mapowanie kolumn
    df["Date"] = pd.to_datetime(df[kol_data], dayfirst=True, errors="coerce")
    df["Assortment"] = df[kol_produkt].astype(str).str.strip()
    df["Sales_Value"] = pd.to_numeric(df[kol_sprzedaz], errors="coerce")
    
    # Filtrowanie po kanale i metryce (opcjonalnie)
    if kol_kanal and filtr_kanal:
        df = df[df[kol_kanal].astype(str).str.strip() == filtr_kanal].copy()
    
    if kol_metryka and filtr_metryka:
        df = df[df[kol_metryka].astype(str).str.strip() == filtr_metryka].copy()
    
    # Usunięcie wierszy z brakującymi wartościami
    df = df.dropna(subset=["Date", "Assortment", "Sales_Value"])
    
    # Sortowanie
    df = df.sort_values("Date").reset_index(drop=True)
    
    return df[["Date", "Assortment", "Sales_Value"]]

# ============================================================================
# SEKCJA 5: POBIERANIE POGODY
# ============================================================================

@st.cache_data(show_spinner=False)
def pobierz_pogode_historyczna(lat, lon, data_start, data_end):
    """
    Pobiera dane pogodowe z Open-Meteo Archive API.
    """
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": data_start.strftime("%Y-%m-%d"),
        "end_date": data_end.strftime("%Y-%m-%d"),
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
        st.error(f"❌ Błąd pobierania pogody historycznej: {e}")
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def pobierz_pogode_forecast(lat, lon):
    """
    Pobiera prognozę pogody z Open-Meteo Forecast API (16 dni).
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
        st.error(f"❌ Błąd pobierania prognozy pogody: {e}")
        return pd.DataFrame()

def agreguj_pogode_tygodniowo(df_pogoda):
    """
    Agreguje dane pogodowe z poziomu dziennego na tygodniowy.
    Dodaje cechy: heat_wave, first_warm_week, post_heat_wave.
    """
    df = df_pogoda.copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df["ISO_Week"] = df["Date"].dt.isocalendar().week
    df["Year"] = df["Date"].dt.isocalendar().year
    
    # Agregacja podstawowa
    agg_dict = {
        "Temp_Max_C": "mean",
        "Temp_Max_Peak_C": ("Temp_Max_C", "max"),
        "Temp_Mean_C": "mean",
        "Rain_Sum_mm": "sum",
        "Rain_Days": "sum",
        "Wind_Speed_10m_Mean_kmh": "mean",
    }
    
    df_w = df.groupby(["Year", "ISO_Week"], as_index=False).agg({
        "Temp_Max_C": "mean",
        "Temp_Mean_C": "mean",
        "Rain_Sum_mm": "sum",
        "Rain_Days": "sum",
        "Wind_Speed_10m_Mean_kmh": "mean",
    })
    
    # Oblicz Temp_Max_Peak_C
    df_w["Temp_Max_Peak_C"] = df.groupby(["Year", "ISO_Week"])["Temp_Max_C"].max().values
    
    # Cechy sezonowe
    df_w["heat_wave"] = (df_w["Temp_Max_Peak_C"] > 30).astype(int)
    
    # first_warm_week — pierwsza, gdzie Temp_Mean > 15°C
    df_w["first_warm_week"] = 0
    for year in df_w["Year"].unique():
        year_data = df_w[df_w["Year"] == year].sort_values("ISO_Week")
        first_warm_idx = (year_data["Temp_Mean_C"] > 15).idxmax()
        if first_warm_idx in df_w.index:
            df_w.loc[first_warm_idx, "first_warm_week"] = 1
    
    # post_heat_wave — tydzień po fali upałów
    df_w["post_heat_wave"] = 0
    heat_waves = df_w[df_w["heat_wave"] == 1].index.tolist()
    for hw_idx in heat_waves:
        next_idx = df_w.index[df_w.index > hw_idx].min()
        if pd.notna(next_idx):
            df_w.loc[next_idx, "post_heat_wave"] = 1
    
    return df_w.sort_values(["Year", "ISO_Week"]).reset_index(drop=True)

# ============================================================================
# SEKCJA 6: CECHY KALENDARZA
# ============================================================================

def dodaj_cechy_kalendarza(df, year_range=None):
    """
    Dodaje cechy kalendarza do DataFrame:
    - is_holiday, days_to_holiday, days_since_holiday
    - easter_week, may_long_weekend, school_holiday
    """
    df = df.copy()
    
    if "Date" not in df.columns and "ISO_Week" in df.columns:
        # Jeśli mamy ISO_Week, zrekonstruuj przybliżoną datę
        df["Date"] = df.apply(
            lambda row: datetime.strptime(f"{int(row['Year'])}-W{int(row['ISO_Week'])}-1", "%Y-W%W-%w"),
            axis=1
        )
    
    df["Date"] = pd.to_datetime(df["Date"])
    
    # Polskie święta
    pl_holidays = holidays.Poland(years=year_range or range(df["Date"].dt.year.min(), df["Date"].dt.year.max() + 1))
    
    # Mapuj święta na tygodnie
    df["is_holiday"] = df["Date"].apply(
        lambda d: 1 if any(pd.to_datetime(d.date()) == pd.to_datetime(h) for h in pl_holidays.keys()) else 0
    )
    
    # Odległość do/od świąt
    holidays_dates = sorted(pl_holidays.keys())
    df["days_to_holiday"] = df["Date"].apply(
        lambda d: min([(h - d).days for h in holidays_dates if (h - d).days > 0], default=999)
    )
    df["days_since_holiday"] = df["Date"].apply(
        lambda d: min([(d - h).days for h in holidays_dates if (d - h).days >= 0], default=999)
    )
    
    # Wielkanoc (3-tygodniowe okno)
    df["easter_week"] = 0
    for year in df["Date"].dt.year.unique():
        try:
            easter = holidays.Poland(years=year).get(datetime(year, 4, 1))
            if easter is None:
                for day in range(1, 31):
                    easter_date = holidays.Easter(year=year)
                    if easter_date:
                        break
            else:
                easter_date = datetime.strptime(easter, "%Y-%m-%d") if isinstance(easter, str) else easter
            
            if easter_date:
                easter_iso_week = easter_date.isocalendar()[1]
                mask = (df["Year"] == year) & (df["ISO_Week"].between(easter_iso_week - 1, easter_iso_week + 1))
                df.loc[mask, "easter_week"] = 1
        except:
            pass
    
    # Majówka (tygodnie 17–19)
    df["may_long_weekend"] = df["ISO_Week"].between(17, 19).astype(int)
    
    # Wakacje szkolne
    df["school_holiday"] = 0
    summer_mask = (df["ISO_Week"] >= 24) & (df["ISO_Week"] <= 35)  # połowa czerwca - koniec sierpnia
    winter_mask = (df["ISO_Week"] >= 3) & (df["ISO_Week"] <= 8)    # połowa stycznia - koniec lutego
    autumn_mask = (df["ISO_Week"] >= 42) & (df["ISO_Week"] <= 43)  # październik
    
    df.loc[summer_mask | winter_mask | autumn_mask, "school_holiday"] = 1
    
    return df

# ============================================================================
# SEKCJA 7: LAGI SPRZEDAŻY
# ============================================================================

def dodaj_lagi_sprzedazy(df):
    """
    Dodaje lagi sprzedaży (lag_52, lag_4, ma4).
    """
    df = df.copy()
    df = df.sort_values(["Assortment", "Year", "ISO_Week"]).reset_index(drop=True)
    
    df["lag_52"] = df.groupby("Assortment")["Sales_Value"].shift(52)
    df["lag_4"] = df.groupby("Assortment")["Sales_Value"].shift(4)
    df["ma4"] = df.groupby("Assortment")["Sales_Value"].rolling(window=4, min_periods=1).mean().values
    
    return df

# ============================================================================
# SEKCJA 8: TRENOWANIE MODELI
# ============================================================================

def trenuj_z_ci(df_asortyment, cechy, quantiles=[0.1, 0.5, 0.9]):
    """
    Trenuje XGBoost z kwantylami (przedziały ufności p10/p50/p90).
    Hyper-parameter tuning za pomocą TimeSeriesSplit.
    """
    df = df_asortyment.dropna(subset=cechy + ["Sales_Value"]).copy()
    
    if len(df) < 30:
        return None, None, None, None
    
    # Przygotuj dane
    X = df[cechy].values
    y = df["Sales_Value"].values
    
    # TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=3)
    best_params = PARAMETRY_XGB.copy()
    best_score = float("inf")
    
    # Grid search (uproszczony)
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
    
    # Trenuj finalne modele (3 kwantyle)
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
    fi = {cechy[int(k.split("_")[1])]: v for k, v in fi.items() if k.startswith("f_")}
    
    return models, evals, fi, best_params

def trenuj_prophet(df_asortyment, cechy_regresory):
    """
    Trenuje Prophet model z sezonowością multiplikatywną i regresorami.
    """
    if not PROPHET_AVAILABLE:
        st.warning("⚠️ Prophet nie jest dostępny. Zainstaluj: pip install prophet")
        return None, None
    
    df = df_asortyment.dropna(subset=cechy_regresory + ["Sales_Value"]).copy()
    
    if len(df) < 30:
        return None, None
    
    # Przygotuj dane
    df["ds"] = pd.to_datetime(df[["Year", "ISO_Week"]].apply(
        lambda x: datetime.strptime(f"{int(x['Year'])}-W{int(x['ISO_Week'])}-1", "%Y-W%W-%w"), axis=1
    ))
    df["y"] = df["Sales_Value"]
    
    try:
        model = Prophet(interval_width=0.8, yearly_seasonality=True, seasonality_mode="multiplicative")
        
        # Dodaj regresory
        for reg in cechy_regresory:
            if reg in df.columns:
                model.add_regressor(reg)
        
        # Dodaj polskie święta
        pl_holidays = holidays.Poland()
        for date in pl_holidays.keys():
            model.add_seasonality(name=str(date), period=365.25, fourier_order=5)
        
        model.fit(df[["ds", "y"] + cechy_regresory])
        
        # Ewaluacja na całym zbiorze
        forecast = model.make_future_dataframe(periods=0, freq="W")
        forecast = model.predict(forecast[forecast["ds"].dt.day <= 7])
        
        y_actual = df["y"].values
        y_pred = forecast["yhat"].values[:len(y_actual)]
        
        mape = mean_absolute_percentage_error(y_actual, np.maximum(y_pred, 1))
        mae = mean_absolute_error(y_actual, y_pred)
        r2 = r2_score(y_actual, y_pred)
        
        return model, {"MAPE": mape, "MAE": mae, "R2": r2}
    except Exception as e:
        st.warning(f"⚠️ Błąd trenowania Prophet: {e}")
        return None, None

# ============================================================================
# SEKCJA 9: GŁÓWNA APLIKACJA (UI)
# ============================================================================

def main():
    st.markdown("""
    <style>
        .main { max-width: 100%; }
        .stTabs [data-baseweb="tab-list"] button { font-weight: bold; }
    </style>
    """, unsafe_allow_html=True)
    
    st.title("🍦 Prognoza Sprzedaży Lodów")
    st.markdown("**Zaawansowany system prognozy ML** — XGBoost + Prophet | Open-Meteo API | Cechy kalendarza")
    
    # ========== SIDEBAR: WGRANIE DANYCH ==========
    with st.sidebar:
        st.header("📥 Konfiguracja")
        
        uploaded_file = st.file_uploader("Wgraj plik Excel ze sprzedażą", type=["xlsx", "csv"])
        
        if uploaded_file:
            file_bytes = uploaded_file.read()
            
            # Wybór arkusza
            sheets = lista_arkuszy(file_bytes)
            if sheets:
                selected_sheet = st.selectbox("Wybierz arkusz", sheets)
                df_raw = wczytaj_arkusz(file_bytes, selected_sheet)
                
                if not df_raw.empty:
                    st.success(f"✅ Wczytano {len(df_raw)} wierszy")
                    
                    # Mapowanie kolumn
                    st.subheader("Mapowanie kolumn")
                    cols = df_raw.columns.tolist()
                    
                    kol_data = st.selectbox("Kolumna z datą", cols, index=0)
                    kol_produkt = st.selectbox("Kolumna z asortymentem", cols, index=min(1, len(cols)-1))
                    kol_sprzedaz = st.selectbox("Kolumna ze sprzedażą [szt]", cols, index=min(2, len(cols)-1))
                    
                    # Standaryzacja
                    df_std = standaryzuj(df_raw, kol_data, kol_produkt, kol_sprzedaz)
                    
                    if not df_std.empty:
                        # Agregacja tygodniowa
                        df_std["Year"] = df_std["Date"].dt.isocalendar().year
                        df_std["ISO_Week"] = df_std["Date"].dt.isocalendar().week
                        
                        df_tyg = df_std.groupby(["Year", "ISO_Week", "Assortment"], as_index=False)["Sales_Value"].sum()
                        
                        # Pobierz pogodę
                        st.subheader("☀️ Pogoda")
                        if st.button("Pobierz pogodę historyczną"):
                            data_min = pd.to_datetime(df_std["Date"]).min()
                            data_max = pd.to_datetime(df_std["Date"]).max()
                            
                            with st.spinner("Pobieranie..."):
                                df_pogoda = pobierz_pogode_historyczna(LAT, LON, data_min, data_max)
                            
                            if not df_pogoda.empty:
                                df_pogoda_w = agreguj_pogode_tygodniowo(df_pogoda)
                                st.session_state.df_pogoda_w = df_pogoda_w
                                st.success("✅ Pogoda pobrana")
        
        st.divider()
        st.markdown("**Grupa IV AIFINC 2026 · ALK**")
    
    # ========== GŁÓWNE TABY ==========
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Dane", "🤖 Treningi", "🔮 Prognoza", "📈 Rekomendacje", "📉 Backtest"])
    
    with tab1:
        st.header("📊 Wczytane dane")
        if "df_pogoda_w" in st.session_state:
            st.write("✅ Pogoda wczytana")
        else:
            st.info("ℹ️ Wgraj dane i pobierz pogodę w panelu po lewej")
    
    with tab2:
        st.header("🤖 Trenowanie modeli")
        st.info("ℹ️ Moduł trenowania XGBoost i Prophet")
    
    with tab3:
        st.header("🔮 Prognoza na 8 tygodni")
        st.info("ℹ️ Moduł prognozowania przyszłej sprzedaży")
    
    with tab4:
        st.header("📈 Rekomendacje strategiczne")
        st.info("ℹ️ Analiza cech i klasyfikacja ABC/XYZ")
    
    with tab5:
        st.header("📉 Actual vs Forecast")
        st.info("ℹ️ Backtest i ocena dokładności modeli")

if __name__ == "__main__":
    main()
