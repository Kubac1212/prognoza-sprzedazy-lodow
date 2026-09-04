# 🍦 Prognoza Sprzedaży Lodów

Aplikacja do prognozowania tygodniowej sprzedaży lodów oparta na modelach Machine Learning (XGBoost + Prophet), danych pogodowych (Open-Meteo API) i cechach kalendarza. Zbudowana w Streamlit — działa lokalnie w przeglądarce, bez potrzeby instalowania czegokolwiek poza Pythonem.

---

## ▶️ Uruchomienie (dla użytkownika końcowego)

### Windows

1. Upewnij się, że masz zainstalowany **Python 3.10 lub nowszy** → [python.org/downloads](https://www.python.org/downloads/)  
   ⚠️ Podczas instalacji zaznacz opcję **"Add Python to PATH"**
2. Otwórz folder z aplikacją
3. **Kliknij dwukrotnie `run.bat`**
4. Przy pierwszym uruchomieniu skrypt zainstaluje wszystkie zależności (może potrwać 2–5 minut)
5. Aplikacja otworzy się automatycznie w przeglądarce pod adresem **http://localhost:8501**

### macOS / Linux

1. Upewnij się, że masz zainstalowany **Python 3.10 lub nowszy**
2. Otwórz Terminal w folderze z aplikacją
3. Wpisz: `bash run.sh` i naciśnij Enter
4. Przy pierwszym uruchomieniu skrypt zainstaluje wszystkie zależności
5. Aplikacja otworzy się automatycznie pod adresem **http://localhost:8501**

> **Aby zatrzymać aplikację:** zamknij okno terminala lub naciśnij `Ctrl+C`

---

## 📁 Struktura plików

```
projekt/
├── app_prognoza_lodow.py   # Główny kod aplikacji (~2700 linii)
├── requirements.txt        # Lista zależności Python z wersjami
├── run.bat                 # Skrypt uruchamiający dla Windows
├── run.sh                  # Skrypt uruchamiający dla macOS/Linux
├── README.md               # Ten plik
└── korekty_manualne.xlsx   # Tworzony automatycznie po pierwszej korekcie
```

---

## 🏗️ Architektura aplikacji

### Przepływ działania

```
Użytkownik wgrywa plik Excel (sprzedaż)
        ↓
Wczytanie i mapowanie kolumn (data, asortyment, sprzedaż)
        ↓
Agregacja do tygodni → połączenie z pogodą historyczną (Open-Meteo Archive API)
        ↓
Dodanie cech kalendarza (święta, Wielkanoc, majówka, wakacje szkolne)
        ↓
Dodanie cech sezonowych (first_warm_week, post_heat_wave)
        ↓
Trening modeli ML (XGBoost / Prophet) per asortyment
  ├─ Strojenie hiperparametrów (TimeSeriesSplit CV)
  ├─ Ewaluacja: MAPE, MAE, R²
  └─ Feature importance
        ↓
Tab 3 — Prognoza T+1…T+8:
  ├─ Pobierz prognozę pogody (Open-Meteo Forecast API)
  ├─ Fallback: historyczne średnie dla T+4–T+8
  ├─ Prognoza modelu + przedziały ufności (p10/p50/p90)
  └─ Rekomendacja zamówienia + ryzyko stockoutu
        ↓
Tab 4 — Rekomendacje:
  ├─ Zbiorcza ważność cech
  └─ Klasyfikacja ABC/XYZ asortymentów
        ↓
Tab 5 — Actual vs Forecast (opcjonalny backtest)
```

---

## 🔧 Kluczowe sekcje kodu

### Stałe i listy cech (`CECHY_*`)

Definicje zestawów cech używanych przez modele. Różne zestawy dla:
- **XGBoost** (`CECHY_MODELU`) — pełny zestaw z lagami sprzedaży
- **Prophet impulsowy** (`CECHY_PROPHET_IMPULSOWE`) — temperatura maks, deszcz, święta
- **Prophet familijny** (`CECHY_PROPHET_FAMILIJNE`) — temperatura średnia, opady, wiatr

### Pobieranie pogody

| Funkcja | Co robi |
|---|---|
| `pobierz_pogode_historyczna()` | Open-Meteo Archive API — pogoda przeszła (trening) |
| `pobierz_pogode_forecast()` | Open-Meteo Forecast API — prognoza 16 dni |
| `agreguj_pogode_tygodniowo()` | Agreguje dzienne dane do tygodni + `heat_wave`, `first_warm_week`, `post_heat_wave` |
| `rozszerz_prognoza_pogody()` | Rozszerza prognozę do 8 tygodni (API → historyczne średnie → domyślne) |

### Cechy kalendarza (`dodaj_cechy_kalendarza`)

Dodaje do każdego tygodnia:
- `is_holiday` — czy tydzień zawiera polskie święto ustawowe
- `days_to_holiday` / `days_since_holiday` — bliskość świąt
- `easter_week` — 3-tygodniowe okno wokół Wielkanocy (rzeczywiste daty 2019–2028)
- `may_long_weekend` — majówka (tygodnie 17–19)
- `school_holiday` — wakacje/ferie (letnie 20VI–31VIII, zimowe tyg.3–8, wielkanocne, jesienne tyg.42–43)

### Lagi sprzedaży (`dodaj_lagi_sprzedazy`, `dodaj_lagi_do_prognozy`)

- `lag_52` — sprzedaż rok temu w tym samym tygodniu (główna cecha sezonowości)
- `lag_4` — sprzedaż miesiąc temu
- `ma4` — średnia krocząca z 4 tygodni poprzedzających

Dla prognozy `lag_52` jest wyszukiwany w historii per data, a `lag_4`/`ma4` to stałe z ostatnich znanych tygodni.

### Trening XGBoost (`trenuj_z_ci`)

Trenuje 3 modele kwantylowe (p10, p50, p90) z przedziałami ufności:
1. Dodaje lagi sprzedaży
2. **Stroi hiperparametry** — `TimeSeriesSplit(n_splits=3)` × siatka `max_depth ∈ {2,3,4}` × `min_child_weight ∈ {3,8}`
3. Trenuje finalne modele na najlepszych parametrach
4. Zwraca MAPE, MAE, R², feature importance, `best_params`

### Trening Prophet (`trenuj_prophet`)

Prophet z sezonowością multiplikatywną, polskimi świętami i regresorami pogodowymi. Osobne zestawy cech dla produktów impulsowych vs familijnych (wybór w sidebarze przez `st.multiselect`).

Prognoza Prophet (`prognozuj_prophet`) — wypełnia cechy z `df_fc_tyg` dla przyszłych tygodni.

### Tab 3 — Prognoza (`with tab3`)

1. Pobiera pogodę na nadchodzące tygodnie
2. Oblicza cechy sekwencyjne (`first_warm_week`, `post_heat_wave`) używając ogona historii + prognozy
3. Dla każdego asortymentu: dodaje lagi → uruchamia model → skaluje scenariuszem popytu
4. Wyświetla wykres (deterministyczny T+1–T+N | orientacyjny T+N+1–T+8)
5. Rekomendacja zamówienia tylko dla tygodni z prognozą API, trend sezonowy osobno (zwijany)

### Tab 4 — Rekomendacje (`with tab4`)

- Zbiorcza ważność cech (uśredniona po wszystkich modelach)
- Klasyfikacja ABC (wolumen) i XYZ (zmienność / CV) asortymentów

### Tab 5 — Actual vs Forecast (`with tab5`)

Użytkownik wgrywa Excel z uzupełnioną kolumną `actual_szt`. Aplikacja porównuje prognozy modelu z rzeczywistością i wylicza MAPE, MAE, bias per asortyment.

---

## 🌐 Zewnętrzne API

| Serwis | Endpoint | Do czego |
|---|---|---|
| Open-Meteo Archive | `archive-api.open-meteo.com/v1/archive` | Pogoda historyczna do treningu |
| Open-Meteo Forecast | `api.open-meteo.com/v1/forecast` | Prognoza pogody na 16 dni |

Oba są **bezpłatne i nie wymagają klucza API**. Klucz OpenWeatherMap jest opcjonalny (pole w sidebarze).

---

## 📦 Zależności

| Pakiet | Wersja | Rola |
|---|---|---|
| `streamlit` | 1.57.0 | Framework webowy — UI, wykresy, interaktywność |
| `pandas` | 3.0.2 | Przetwarzanie danych tabelarycznych |
| `numpy` | 2.4.4 | Obliczenia numeryczne |
| `xgboost` | 3.2.0 | Model regresji kwantylowej |
| `prophet` | 1.3.0 | Model szeregów czasowych z sezonowością |
| `scikit-learn` | 1.8.0 | `TimeSeriesSplit` do strojenia hiperparametrów |
| `plotly` | 6.7.0 | Interaktywne wykresy |
| `holidays` | 0.96 | Polskie święta ustawowe |
| `requests` | 2.33.1 | Zapytania do Open-Meteo API |
| `openpyxl` | 3.1.5 | Odczyt/zapis plików Excel |

---

## ⚙️ Wymagania systemowe

- Python 3.10 lub nowszy
- Połączenie z internetem (pobieranie pogody z Open-Meteo)
- ~500 MB miejsca na dysku (głównie środowisko venv + Prophet/Stan)
- RAM: minimum 2 GB (zalecane 4 GB przy wielu asortymentach)

---

*Aplikacja stworzona w ramach projektu Grupa IV · AIFINC 2026 · ALK*
