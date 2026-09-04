# 📘 Objaśnienie kodu — `app_prognoza_lodow.py`

Plik opisuje **każdą sekcję i funkcję** aplikacji prognozy sprzedaży lodów.
Dokument podzielony jest zgodnie z kolejnością pojawiania się kodu w pliku.

---

## 1. Docstring i importy (linie 1–22)

```python
"""
Aplikacja prognozy sprzedaży lodów — Grupa IV ALK 2026
...
"""
import io, streamlit, pandas, numpy, xgboost, plotly, holidays, requests
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
```

**Co robi:**
- `io` — obsługa plików w pamięci (BytesIO), potrzebne do wczytywania Excel bez zapisywania na dysk.
- `streamlit` (`st`) — framework webowy: buduje UI, obsługuje upload plików, rysuje wykresy, zarządza stanem sesji.
- `pandas` (`pd`) — przetwarzanie danych tabelarycznych: DataFramy, operacje na datach, grupowania, merge.
- `numpy` (`np`) — obliczenia numeryczne: tablice, percentyle, operacje wektorowe.
- `xgboost` (`xgb`) — model ML (XGBRegressor z regresją kwantylową) do prognozowania sprzedaży.
- `plotly` (`go`, `px`, `make_subplots`) — interaktywne wykresy wyświetlane w przeglądarce.
- `holidays` — biblioteka z polskim kalendarzem świąt ustawowych.
- `requests` — zapytania HTTP do Open-Meteo API (pobieranie danych pogodowych).
- `pathlib.Path` — wygodna obsługa ścieżek plików niezależna od systemu operacyjnego.
- `datetime` — pobieranie aktualnej daty i godziny (m.in. do nazw plików eksportowanych).
- `warnings.filterwarnings("ignore")` — wycisza ostrzeżenia bibliotek (np. deprecation warnings z Prophet/Stan) żeby nie zaśmiecać interfejsu.

---

## 2. Konfiguracja strony Streamlit (linie 28–33)

```python
st.set_page_config(
    page_title="Prognoza Sprzedaży Lodów",
    page_icon="🍦",
    layout="wide",
    initial_sidebar_state="expanded",
)
```

**Co robi:**
- Musi być **pierwszym wywołaniem Streamlit** w pliku — ustawia globalną konfigurację strony.
- `layout="wide"` — aplikacja zajmuje pełną szerokość okna przeglądarki (zamiast domyślnego wąskiego układu).
- `initial_sidebar_state="expanded"` — sidebar (panel konfiguracji) jest domyślnie widoczny.

---

## 3. Stałe — zestawy cech (`CECHY_*`, linie 38–76)

### 3.1 `CECHY_POGODOWE`, `CECHY_SWIATECZNE`, `CECHY_CZASOWE`, `CECHY_LAGI`

```python
CECHY_POGODOWE  = ["Temp_Max_C", "Temp_Max_Peak_C", "heat_wave", "first_warm_week",
                   "post_heat_wave", "Temp_Mean_C", "Rain_Sum_mm", "Rain_Days",
                   "Wind_Speed_10m_Mean_kmh"]
CECHY_SWIATECZNE = ["is_holiday", "days_to_holiday", "days_since_holiday",
                    "easter_week", "may_long_weekend", "school_holiday"]
CECHY_CZASOWE    = ["ISO_Week", "Month"]
CECHY_LAGI       = ["lag_52", "lag_4", "ma4"]
CECHY_MODELU     = CECHY_POGODOWE + CECHY_SWIATECZNE + CECHY_CZASOWE + CECHY_LAGI
```

**Co robi:**
Definicje zestawów cech wejściowych dla modeli. Podzielone tematycznie:
- **Pogodowe** — zmienne z Open-Meteo API: temperatura max (średnia tygodnia i najgorętszy dzień), fala upałów, pierwszy ciepły tydzień sezonu, tydzień po fali upałów, temperatura średnia, suma/liczba dni z opadami, wiatr.
- **Świąteczne** — zmienne kalendarza: czy tydzień zawiera święto, odległość w dniach do/od święta, Wielkanoc, majówka, wakacje/ferie szkolne.
- **Czasowe** — numer tygodnia ISO (1–52) i miesiąc — kodują sezonowość.
- **Lagi sprzedaży** — historyczna sprzedaż: rok temu (lag_52), miesiąc temu (lag_4), średnia krocząca 4 tygodnie (ma4). Kluczowe dla XGBoost, bo model nie „widzi" czasu — trzeba mu podać historię jawnie.
- `CECHY_MODELU` — pełny zestaw (suma wszystkich) dla modelu XGBoost z domyślnym zestawem.

### 3.2 `CECHY_IMPULSOWE` i `CECHY_FAMILIJNE`

```python
CECHY_IMPULSOWE = ["Temp_Max_C", "Temp_Max_Peak_C", ..., "Rain_Days", ...]
CECHY_FAMILIJNE = ["Temp_Mean_C", "Temp_Max_Peak_C", ..., "Rain_Sum_mm", "Wind_Speed_10m_Mean_kmh", ...]
```

**Co robi:**
Warianty zestawów cech dostosowane do **typu produktu**:
- **Impulsowe** (single, rożki, lody na patyku) — reagują głównie na **temperaturę maksymalną** i **liczbę dni z deszczem** (decyzja zakupowa jest spontaniczna, powiązana z upałem i słoneczną pogodą).
- **Familijne** (opakowania 900 ml+) — reagują na **temperaturę średnią**, **sumę opadów** i **wiatr** (zakup planowany na zapas do domu, mniej wrażliwy na chwilowy szczyt temperatury).
- Oba zestawy zawierają takie same cechy kalendarza i lagi sprzedaży.

### 3.3 `CECHY_PROPHET_IMPULSOWE` i `CECHY_PROPHET_FAMILIJNE`

```python
CECHY_PROPHET_IMPULSOWE = ["Temp_Max_C", ..., "Rain_Days", "is_holiday", "easter_week", ...]
CECHY_PROPHET_FAMILIJNE = ["Temp_Mean_C", ..., "Rain_Sum_mm", "Wind_Speed_10m_Mean_kmh", ...]
```

**Co robi:**
Zestawy cech dla modelu **Prophet** — celowo **bez lagów sprzedaży** (`lag_52`, `lag_4`, `ma4`), bo Prophet ma własny wbudowany mechanizm trendu i sezonowości rocznej. Dodawanie lagów byłoby redundancją i mogłoby zakłócić wewnętrzną dekompozycję modelu.

---

## 4. Parametry XGBoost i stałe pomocnicze (linie 78–122)

### 4.1 `PARAMETRY_XGB`

```python
PARAMETRY_XGB = {
    "n_estimators": 200,       # liczba drzew decyzyjnych
    "learning_rate": 0.05,     # krok uczenia (mały = ostrożny, mniej overfittingu)
    "max_depth": 3,            # domyślna głębokość drzewa (nadpisywana przez CV)
    "min_child_weight": 5,     # min. waga węzła (regularyzacja)
    "subsample": 0.8,          # 80% próbek na każde drzewo (losowanie bez zwracania)
    "colsample_bytree": 0.8,   # 80% cech na każde drzewo (regularyzacja kolumnowa)
    "random_state": 42,        # ziarno losowości — wyniki powtarzalne
    "verbosity": 0,            # wyciszenie logów XGBoost
}
```

**Co robi:**
Domyślna siatka hiperparametrów dla XGBoost. Parametry `max_depth` i `min_child_weight` są **nadpisywane przez strojenie CV** (`trenuj_z_ci`), jeśli danych jest wystarczająco dużo (≥ 30 próbek treningowych). Pozostałe parametry są stałe — dobrane empirycznie jako dobre wartości domyślne dla tygodniowych szeregów czasowych sprzedaży.

### 4.2 Współrzędne geograficzne i paleta kolorów

```python
LAT, LON = 51.9, 19.1   # środek Polski (używane do pobierania pogody z Open-Meteo)

K_RED    = "#E8231A"   # czerwień Koral — główny akcent (linie prognozy)
K_YELLOW = "#FFB800"   # żółty Koral — drugi akcent (przedziały)
K_BLUE   = "#1B4FCC"   # niebieski — historia, tekst
K_LIGHT  = "#FFF8E1"   # kremowy — tła kart
K_GRAY   = "#616161"   # szary — elementy pomocnicze
```

**Co robi:**
`LAT/LON` to geograficzny środek Polski — Open-Meteo API wymaga podania lokalizacji. Przy produkcji warto zmienić na lokalizację konkretnego rynku (np. magazynu).
Paleta `K_*` zapewnia spójność wizualną zgodną z identyfikacją firmy.

### 4.3 `OPIS_CECH`

```python
OPIS_CECH = {
    "Temp_Max_C": "Temperatura maks. (śr. tygodnia) [°C]",
    "lag_52": "📅 Sprzedaż rok temu (lag 52 tygodnie)",
    ...
}
```

**Co robi:**
Słownik tłumaczący wewnętrzne nazwy cech (np. `"lag_52"`) na czytelne etykiety wyświetlane na wykresach ważności cech. Zapewnia spójne nazewnictwo w całej aplikacji.

---

## 5. Wczytywanie pliku Excel (linie 129–139)

### 5.1 `lista_arkuszy(plik_bytes)`

```python
def lista_arkuszy(plik_bytes: bytes) -> list[str]:
    import openpyxl
    wb = openpyxl.load_workbook(io.BytesIO(plik_bytes), read_only=True, data_only=True)
    names = wb.sheetnames
    wb.close()
    return names
```

**Co robi:**
Otwiera plik Excel **tylko do odczytu** w pamięci (`io.BytesIO`) i zwraca listę nazw arkuszy. Tryb `read_only=True` jest szybszy i zużywa mniej RAM — nie wczytuje zawartości komórek. `data_only=True` pomija formuły (zwraca wartości). Plik nie jest zapisywany na dysku.

### 5.2 `wczytaj_arkusz(plik_bytes, arkusz)`

```python
@st.cache_data(show_spinner=False)
def wczytaj_arkusz(plik_bytes: bytes, arkusz: str) -> pd.DataFrame:
    return pd.read_excel(io.BytesIO(plik_bytes), sheet_name=arkusz)
```

**Co robi:**
Wczytuje wybrany arkusz do DataFrame. Dekorator `@st.cache_data` powoduje, że wynik jest **cachowany** — jeśli użytkownik zmieni tylko inną konfigurację (np. wybór asortymentu), Excel nie jest ponownie parsowany. Klucz cache = `(plik_bytes, arkusz)`.

---

## 6. Standaryzacja danych — `standaryzuj()` (linie 146–178)

```python
def standaryzuj(df, kol_data, kol_produkt, kol_sprzedaz,
                kol_kanal, kol_metryka, filtr_kanal, filtr_metryka) -> pd.DataFrame:
```

**Co robi:**
Przemapowuje kolumny z pliku użytkownika na **wewnętrzne nazwy aplikacji**. Przyjmuje surowy DataFrame (z dowolnymi nazwami kolumn) i zwraca DataFrame ze standaryzowanymi kolumnami:
- `Date` — data (parse z wielu formatów dzięki `dayfirst=True, errors="coerce"`).
- `Assortment` — nazwa produktu.
- `Sales_Value` — wartość sprzedaży (konwertowana do liczby, błędy → NaN).
- `Channel_Group` — kanał dystrybucji (opcjonalne; jeśli brak → "WSZYSTKIE").
- `Metric` — metryka (szt./PLN; jeśli brak → "szt").
- Filtruje wiersze jeśli podano `filtr_kanal` lub `filtr_metryka`.
- Usuwa wiersze z brakami w dacie lub sprzedaży (`dropna`).
- Sortuje chronologicznie.

---

## 7. Agregacja tygodniowa — `agreguj_sprzedaz_tygodniowo()` (linie 185–202)

```python
def agreguj_sprzedaz_tygodniowo(df: pd.DataFrame) -> pd.DataFrame:
```

**Co robi:**
Jeśli dane wejściowe są **dzienne** (jeden wiersz na dzień na produkt), sumuje je do **tygodnia ISO**. Grupuje po `(Assortment, Channel_Group, year_ISO, week_ISO)` i oblicza:
- `Sales_Value` → suma sprzedaży w tygodniu.
- `Date` → data pierwszego dnia (poniedziałku) tygodnia.
- `Month` → miesiąc z daty poniedziałku (do cech modelu).

Wynik: jeden wiersz per (asortyment, tydzień). Jeśli dane były już tygodniowe, suma jest trywialna (jeden element).

---

## 8. Pobieranie pogody historycznej — `pobierz_pogode_historyczna()` (linie 209–227)

```python
@st.cache_data(ttl=86400, show_spinner=False)
def pobierz_pogode_historyczna(data_od, data_do, lat, lon):
```

**Co robi:**
Pobiera dzienne dane pogodowe z **Open-Meteo Archive API** (bezpłatne, bez klucza API). Parametry:
- `temperature_2m_max` — dobowa temperatura maksymalna.
- `temperature_2m_mean` — dobowa temperatura średnia.
- `precipitation_sum` — dobowa suma opadów [mm].
- `wind_speed_10m_max` — dobowa prędkość wiatru maks. [km/h].
- Strefa czasowa: `Europe/Warsaw` (ważne dla prawidłowego przypisania dat).

Cache `ttl=86400` — dane odświeżane raz na dobę (pogoda historyczna nie zmienia się częściej). Zwraca `(DataFrame, None)` przy sukcesie lub `(None, komunikat_błędu)`.

---

## 9. Pobieranie prognozy pogody — `pobierz_pogode_forecast()` (linie 230–247)

```python
@st.cache_data(ttl=3600, show_spinner=False)
def pobierz_pogode_forecast(lat, lon, dni=16):
```

**Co robi:**
Pobiera **prognozę pogody na kolejne 16 dni** z Open-Meteo Forecast API. Te same zmienne co archiwum, ale cache `ttl=3600` — odświeżane co godzinę (prognoza zmienia się często). 16 dni to maksymalny darmowy horyzont Open-Meteo — pokrywa około 2 tygodnie pełnych.

---

## 10. Agregacja pogody do tygodni — `agreguj_pogode_tygodniowo()` (linie 250–300)

```python
def agreguj_pogode_tygodniowo(df_dzien: pd.DataFrame) -> pd.DataFrame:
```

**Co robi — krok po kroku:**

1. **Grupowanie po tygodniu ISO** — grupuje dni w tygodnie i oblicza:
   - `Temp_Max_C` — średnia z dobowych maksimów w tygodniu.
   - `Temp_Max_Peak_C` — maksimum z dobowych maksimów (najgorętszy dzień tygodnia).
   - `Temp_Mean_C` — średnia temperatur średnich.
   - `Rain_Sum_mm` — suma opadów w tygodniu.
   - `Rain_Days` — liczba dni z opadami > 5 mm (intensywny deszcz).
   - `Wind_Speed_10m_Mean_kmh` — średnia prędkości wiatru.
   - `n_dni` — ile dni weszło do tygodnia (filtr: ≥ 5 dni, bo krótsze tygodnie to efekty graniczne roku).

2. **`heat_wave`** — flaga `1` jeśli najgorętszy dzień tygodnia > 30°C. Fale upałów dramatycznie zwiększają sprzedaż lodów impulsowych.

3. **`Date`** — oblicza datę poniedziałku tygodnia przez `pd.Timestamp.fromisocalendar`.

4. **Filtr i sortowanie** — wiersze z < 5 dni są usuwane (artefakty z początku/końca danych). Sortowanie chronologiczne jest **wymagane** dla kolejnych cech sekwencyjnych.

5. **`post_heat_wave`** — tydzień *bezpośrednio po* fali upałów: `shift(1) == 1 AND heat_wave == 0`. Uchwytuje efekt psychologiczny "wypalenia popytu" po ekstremalnych upałach.

6. **`first_warm_week`** — pętla po posortowanym szeregu:
   - Liczy `_cold_run`: ile tygodni z rzędu temperatura była < 15°C.
   - Jeśli temperatura ≥ 20°C **i** poprzedzały ją ≥ 4 zimne tygodnie → `first_warm_week = 1`, reset licznika.
   - Odzwierciedla skok popytu psychologicznego "pierwszego loda sezonu" — nieproporcjonalny do samej temperatury.

---

## 11. Stała `_WIELKANOC` i cechy kalendarza (linie 307–379)

### 11.1 `_WIELKANOC`

```python
_WIELKANOC = {
    2019: pd.Timestamp("2019-04-21"), ..., 2028: pd.Timestamp("2028-04-16"),
}
```

**Co robi:**
Słownik z dokładnymi datami Niedzieli Wielkanocnej dla lat 2019–2028. Wielkanoc ma ruchomą datę (zależy od cyklu księżycowego), więc nie można jej obliczyć prostą formułą. Zdefiniowanie jako stała na poziomie modułu (poza funkcją) pozwala używać jej **zarówno w `dodaj_cechy_kalendarza`, jak i w funkcji prognozy pogody** bez redundancji.

### 11.2 `dodaj_cechy_kalendarza(df)`

```python
def dodaj_cechy_kalendarza(df: pd.DataFrame) -> pd.DataFrame:
```

**Co robi — krok po kroku:**

1. **`ISO_Week` i `Month`** — dodaje jeśli jeszcze nie ma (mogło być już obliczone w agregacji).

2. **Czyszczenie** — usuwa istniejące kolumny kalendarza przed ponownym dodaniem (zapobiega duplikatom gdy funkcja jest wywoływana wielokrotnie).

3. **`is_holiday`, `days_to_holiday`, `days_since_holiday`** — funkcja `bliskosc()`:
   - Sprawdza czy w oknie ±3 dni od daty jest polskie święto (biblioteka `holidays.Poland()`).
   - Liczy ile dni do najbliższego przyszłego święta (max 14).
   - Liczy ile dni minęło od ostatniego święta (max 14).
   - Model uczy się, że sprzedaż rośnie *przed* i *w czasie* świąt, a spada po.

4. **`easter_week`** — 3-tygodniowe okno Wielkanocne: tydzień przed, tydzień Wielkanocny i tydzień po. Wielkanoc to najważniejszy okres dla sprzedaży lodów na wiosnę.

5. **`may_long_weekend`** — tygodnie 17–19 (majówka: 1–3 maja i wokół 2 maja). Długi weekend majowy to jeden z pierwszych ciepłych weekendów roku — skok sprzedaży impulsowej.

6. **`school_holiday`** — funkcja `_czy_wakacje()`:
   - Wakacje letnie: 20 czerwca – 31 sierpnia.
   - Przerwa wielkanocna: od Niedzieli Palmowej (7 dni przed Wielkanocą) do 2 tygodni po.
   - Ferie zimowe: tygodnie ISO 3–8 (pokrywa ferie we wszystkich regionach Polski).
   - Przerwa jesienna: tygodnie 42–43 (mid-październik).
   - Podczas wakacji/ferii sprzedaż familijnych lodów rośnie — dzieci są w domu.

---

## 12. Łączenie sprzedaży z pogodą (linie 386–406)

### 12.1 `dolacz_pogode(df_sprzedaz, df_pogoda_tyg)`

```python
def dolacz_pogode(df_sprzedaz, df_pogoda_tyg) -> pd.DataFrame:
```

**Co robi:**
Łączy tygodniową sprzedaż z tygodniową pogodą przez `left join` po `(Year, ISO_Week)`. "Left" oznacza że jeśli pogoda dla danego tygodnia jest niedostępna (np. brak danych API), tydzień zostanie zachowany z `NaN` w kolumnach pogodowych — model poradzi sobie z brakami.

### 12.2 `oblicz_srednie_pogodowe(df_pogoda_tyg)`

```python
def oblicz_srednie_pogodowe(df_pogoda_tyg) -> pd.DataFrame | None:
```

**Co robi:**
Oblicza **historyczne średnie pogodowe per numer tygodnia ISO** (np. "w tygodniu 25 historycznie temperatura max wynosi średnio 23.4°C"). Używane jako **fallback** dla tygodni prognozy poza zasięgiem 16-dniowego API (T+3 do T+8).

---

## 13. Lagi sprzedaży (linie 413–468)

### 13.1 `dodaj_lagi_sprzedazy(df_asort)`

```python
def dodaj_lagi_sprzedazy(df_asort: pd.DataFrame) -> pd.DataFrame:
```

**Co robi:**
Dodaje trzy zmienne opóźnione (lagi) do szeregu czasowego jednego asortymentu:

- **`lag_52`** — sprzedaż sprzed dokładnie 52 tygodni (rok temu). Najsilniejsza cecha sezonowości: model "wie" ile sprzedano w tym samym tygodniu rok temu. `y.shift(52)` przesuwa kolumnę o 52 wiersze w górę.
- **`lag_4`** — sprzedaż sprzed 4 tygodni (miesiąc temu). Krótkoterminowy trend.
- **`ma4`** — średnia krocząca z 4 tygodni poprzedzających aktualny tydzień (shift o 1 przed rolling, żeby nie używać aktualnego tygodnia). Wygładza krótkoterminowe wahania.
- Brakujące wartości na początku szeregu (gdy jeszcze nie ma 52 danych wstecz) uzupełniane **medianą** — neutralna wartość centralna.

### 13.2 `dodaj_lagi_do_prognozy(df_hist_asort, df_fc)`

```python
def dodaj_lagi_do_prognozy(df_hist_asort, df_fc) -> pd.DataFrame:
```

**Co robi:**
Uzupełnia wiersze prognozy (przyszłe tygodnie) o lagi sprzedaży z historii — bo w momencie prognozowania nie mamy jeszcze faktycznej sprzedaży:

- **`lag_52`** — dla każdego przyszłego tygodnia szuka w historii dokładnie rok wstecz (± tolerancja ±7 i ±14 dni, bo przez lata wzorzec tygodni ISO może się przesunąć o 1–2 tygodnie).
- **`lag_4` i `ma4`** — stałe z ostatnich 4 tygodni historii (aproksymacja: zakładamy że niedawna sprzedaż jest reprezentatywna).

---

## 14. Rozszerzanie prognozy pogody — `rozszerz_prognoza_pogody()` (linie 471–532)

```python
def rozszerz_prognoza_pogody(df_fc_tyg, df_srednie, data_start, n_tygodni=8) -> pd.DataFrame:
```

**Co robi:**
Buduje tabelę pogody na kolejne 8 tygodni z **trójwarstwowym fallbackiem**:

1. **Open-Meteo API** (warstwa 1) — szuka danego tygodnia w `df_fc_tyg` (dane z API). Oznaczone `"🌤️ Open-Meteo (API)"`.
2. **Historyczne średnie** (warstwa 2) — dla tygodni poza zasięgiem API (zwykle T+3..T+8): bierze historyczną średnią pogodową dla tego numeru tygodnia ISO. Oznaczone `"📊 Śr. historyczne (fallback)"`.
3. **Wartości domyślne** (warstwa 3) — jeśli brak jakichkolwiek historycznych danych: hardcodowane rozsądne wartości. Oznaczone `"⚙️ Wartości domyślne"`.

Na koniec wywołuje `dodaj_cechy_kalendarza` — żeby wiersze z fallbacku też miały kompletne cechy świąteczne.

Kolumna `zrodlo_pogody` jest kluczowa dla późniejszego rozróżnienia tygodni "deterministycznych" od "orientacyjnych" w prognozie.

---

## 15. Trening XGBoost z przedziałami ufności — `trenuj_z_ci()` (linie 539–605)

```python
def trenuj_z_ci(df_asort: pd.DataFrame, horyzont: int = 8) -> dict | None:
```

**Co robi — krok po kroku:**

1. **Przygotowanie danych** — sortuje chronologicznie, dodaje lagi sprzedaży, wybiera dostępne cechy z `CECHY_MODELU`, usuwa wiersze z NaN.

2. **Split treningowy/testowy** — ostatnie `horyzont` tygodni to zestaw testowy (hold-out), reszta to trening. Minimalna wymagana długość: `horyzont + 15`.

3. **Strojenie hiperparametrów** (jeśli `len(train) >= 30`):
   - `TimeSeriesSplit(n_splits=3)` — walidacja krzyżowa uwzględniająca porządek czasowy (fold 2 zawsze po foldu 1 w czasie).
   - Siatka: 6 kombinacji (`max_depth` ∈ {2, 3, 4} × `min_child_weight` ∈ {3, 8}).
   - Tylko model `p50` (mediana) do wyboru najlepszych parametrów — szybciej niż trenowanie 3 kwantyli × 6 kombinacji × 3 foldy.
   - Metryka selekcji: MAPE (procentowy błąd bezwzględny).
   - Wynik: `best_params` nadpisuje domyślne wartości z `PARAMETRY_XGB`.

4. **Trening finalnych modeli** — 3 modele kwantylowe:
   - `p10` (kwantyl 10%) — pesymistyczny (dolna granica).
   - `p50` (kwantyl 50%) — mediana, prognoza bazowa.
   - `p90` (kwantyl 90%) — optymistyczny (górna granica).
   - Wszystkie z `objective="reg:quantileerror"` — XGBoost minimalizuje bezwzględny błąd ważony przez kwantyl.

5. **Ewaluacja na zbiorze testowym**:
   - **MAPE** — Mean Absolute Percentage Error (w %): `mean(|y - p50| / y) * 100`.
   - **MAE** — Mean Absolute Error w jednostkach sprzedaży.
   - **R²** — współczynnik determinacji: `1 - SS_residual / SS_total`.
   - `mask = y > 0` — wyklucza tygodnie z zerową sprzedażą z MAPE (by uniknąć dzielenia przez zero).

6. **Zwracany słownik** — zawiera modele, daty, wartości rzeczywiste i prognozowane, metryki, ważność cech i `best_params` (do wyświetlenia w Tab 2).

---

## 16. Trening Prophet — `trenuj_prophet()` (linie 608–707)

```python
def trenuj_prophet(df_asort, horyzont=8, is_impulse=True) -> dict | None:
```

**Co robi:**

1. **Selekcja cech** — wybiera `CECHY_PROPHET_IMPULSOWE` lub `CECHY_PROPHET_FAMILIJNE` zależnie od `is_impulse`.

2. **Przygotowanie DataFrame** — Prophet wymaga kolumn `ds` (datetime) i `y` (wartość). Regressory muszą być dostępne w danych (filtr: `notna().sum() > horyzont + 10`), brakujące uzupełniane medianą.

3. **Konfiguracja modelu Prophet**:
   - `yearly_seasonality=True` — automatyczna sezonowość roczna (Fourier).
   - `weekly_seasonality=False` — wyłączona (dane tygodniowe, nie dzienne).
   - `interval_width=0.8` — 80% przedziały ufności (analogicznie do p10/p90 XGBoost).
   - `seasonality_mode="multiplicative"` — sezonowość mnoży trend (nie dodaje). Lepiej modeluje sytuacje gdy w szczycie sezonu wahania są proporcjonalnie większe.
   - `add_country_holidays("PL")` — wbudowane polskie święta.
   - `add_regressor(reg)` — każda cecha pogodowa/kalendarzowa jako zewnętrzny regresor.

4. **Ewaluacja** — identyczna formuła MAPE/MAE/R² jak w XGBoost.

5. **Feature importance dla Prophet** — Prophet nie ma `feature_importances_` jak drzewa, więc obliczane ręcznie:
   - Predykcja modelu na danych treningowych → dostęp do składowych (`comp`: trend, yearly, regressory).
   - W trybie multiplicatywnym: `yhat = trend × (1 + yearly + Σ regressors)`.
   - Efekt każdej składowej = `trend × składowa_regresor` (w jednostkach sprzedaży).
   - Ważność = odchylenie standardowe efektu / średni `|yhat|` (znormalizowane).

---

## 17. Prognozowanie Prophet — `prognozuj_prophet()` (linie 710–725)

```python
def prognozuj_prophet(res: dict, df_fc_tyg: pd.DataFrame):
```

**Co robi:**
Uruchamia wytrenowany model Prophet na przyszłych tygodniach z `df_fc_tyg`. Buduje DataFrame `future` z kolumną `ds` (daty) i wartościami regresorów z prognozy pogody. Zwraca `(p10, p50, p90)` po przycięciu do zera (`np.clip(..., 0, None)`).

---

## 18. Funkcje pomocnicze (linie 728–742)

### `fmt(n, dec=0)`

```python
def fmt(n, dec=0) -> str:
```

**Co robi:**
Formatuje liczby do czytelnego wyświetlenia:
- Separator tysięcy: spacja (np. `1 234 567`).
- Separator dziesiętny: kropka (np. `23.4`).
- Zwraca `"—"` dla `None` i `NaN`.

### `klasyfikuj_mape(m)`

```python
def klasyfikuj_mape(m: float) -> tuple[str, str]:
```

**Co robi:**
Mapuje wartość MAPE na etykietę tekstową i kolor tła:
- ≤ 15% → "Doskonały ✅", zielony.
- 15–20% → "Dobry 🔵", niebieski.
- 20–30% → "Akceptowalny ⚠️", żółty.
- > 30% → "Wymaga danych 🔴", czerwony.

---

## 19. Sidebar — panel konfiguracji (linie 749–958)

### 19.1 Źródło pliku i wczytanie

```python
zrodlo = st.radio("Źródło pliku", ["📂 Wgraj plik", "📁 Ścieżka lokalna"], horizontal=True)
```

**Co robi:**
Dwa tryby wczytania pliku:
- **Wgraj plik** — `st.file_uploader` do przesyłania przez przeglądarkę (standardowy przypadek użycia).
- **Ścieżka lokalna** — wpisanie ścieżki (przydatne przy uruchomieniu lokalnym, gdzie plik już jest na dysku).
- `st.stop()` — zatrzymuje dalsze wykonanie jeśli brak pliku (zapobiega błędom w dalszych sekcjach).

### 19.2 Wybór arkusza

```python
PREFEROWANY = "Stage3_Model_Input"
idx_ark = arkusze.index(PREFEROWANY) if PREFEROWANY in arkusze else 0
arkusz = st.selectbox("Arkusz", arkusze, index=idx_ark)
```

**Co robi:**
Pobiera listę arkuszy i domyślnie zaznacza `"Stage3_Model_Input"` — nazwę arkusza z danymi modelowymi w standardowym pliku roboczym projektu. Jeśli brak → pierwszy arkusz.

### 19.3 Mapowanie kolumn — `guess()`

```python
def guess(candidates):
    for c in candidates:
        for k in kolumny:
            if c.lower() in k.lower():
                return k
    return kolumny[0]
```

**Co robi:**
Funkcja auto-detekcji: dla każdego "kandydata" (np. `"date"`, `"data"`, `"tydzień"`) sprawdza czy któraś kolumna pliku zawiera tę frazę (case-insensitive). Pierwsze trafienie = domyślny wybór. Ogranicza ręczną konfigurację dla popularnych formatów plików.

### 19.4 Filtr kanału i metryki z auto-detekcją

```python
_SZT_SLOWA  = {"szt", "szt.", "sztuki", ...}
_PLN_SLOWA  = {"pln", "zł", "wartość", ...}
def _wykryj_grupe(v: str) -> str: ...
```

**Co robi:**
Auto-rozpoznaje czy dostępna metryka to "sztuki" czy "PLN" na podstawie słów kluczowych. Jeśli wykryje obie → wyświetla intuicyjne radio buttons `"📦 Sztuki"` / `"💰 Wartość [PLN]"` zamiast surowej listy wartości z pliku.

### 19.5 Agregacja tygodniowa i asortymenty

```python
df_tyg_sprzedaz = agreguj_sprzedaz_tygodniowo(df_std)
asortymenty = sorted(df_tyg_sprzedaz["Assortment"].unique())
```

**Co robi:**
Po standaryzacji i filtrowaniu — agreguje dane do tygodniowego wymiaru i buduje listę dostępnych asortymentów. Ta lista jest używana w całej aplikacji.

### 19.6 Zakres dat treningowych

```python
zakres_od = st.date_input("Od", ...)
zakres_do = st.date_input("Do", ...)
df_tyg_sprzedaz = df_tyg_sprzedaz[
    (df_tyg_sprzedaz["Date"] >= pd.Timestamp(zakres_od)) &
    (df_tyg_sprzedaz["Date"] <= pd.Timestamp(zakres_do))
]
```

**Co robi:**
Pozwala zawęzić dane treningowe do wybranego zakresu dat. Przydatne gdy plik zawiera dane z okresu atypowego (np. COVID 2020–2021) — można je wykluczyć bez modyfikacji źródłowego pliku.

### 19.7 Typ asortymentu (tylko Prophet)

```python
familijne_asort = st.multiselect("Asortymenty familijne", options=asortymenty, key="familijne_asort")
```

**Co robi:**
Pozwala użytkownikowi zaznaczyć które asortymenty są "familijne" (opakowania 900 ml+) — wówczas Prophet użyje dla nich `CECHY_PROPHET_FAMILIJNE` (temperatura średnia, suma opadów, wiatr) zamiast impulsowych (temperatura max, dni z deszczem). XGBoost zawsze używa `CECHY_MODELU` (pełny zestaw).

### 19.8 Horyzont, model i przycisk startu

```python
horyzont = st.slider("Horyzont testowy [tygodnie]", 4, 16, 8)
model_wybor = st.radio("🤖 Model predykcji", ["XGBoost", "Prophet", "Porównanie (oba)"])
run_btn = st.button("▶ Uruchom analizę", type="primary")
```

**Co robi:**
- `horyzont` — ile ostatnich tygodni historii "ukryć" przed modelem i użyć do oceny jakości prognozy.
- `model_wybor` — XGBoost (szybki), Prophet (szeregi czasowe z sezonowością), lub oba jednocześnie z porównaniem MAPE.
- `run_btn` — przycisk triggerujący trening modeli (do momentu kliknięcia Tab 2–4 pokazują komunikat "Kliknij Uruchom analizę").

---

## 20. Nagłówek — logo i tytuł (linie 963–986)

```python
_logo_path = next((Path(__file__).parent / f"koral_logo{ext}" for ext in (".png", ".jpg", ".jpeg")
     if (...).exists()), None)
```

**Co robi:**
Szuka pliku logo (`koral_logo.png/jpg/jpeg`) w folderze aplikacji. Jeśli istnieje — wyświetla logo obok tytułu w układzie 2-kolumnowym (1:4). Jeśli brak — wyświetla emoji 🍦 jako ikonę tytułu. Fallback jest też zintegrowany z `st.session_state["logo_bytes"]` na wypadek gdyby logo było wgrane przez sidebar.

---

## 21. Pobieranie pogody historycznej (linie 992–1006)

```python
with st.spinner("Pobieranie pogody historycznej..."):
    df_pogoda_hist_dzien, err_pogoda = pobierz_pogode_historyczna(data_min, data_max)
```

**Co robi:**
Pobiera dzienną pogodę historyczną dla całego zakresu dat z pliku sprzedażowego (od `data_min` do `data_max`). Wykonywane **przed tabami** — dane pogodowe są potrzebne do treningu i muszą być dostępne zanim użytkownik otworzy Tab 1 lub 2. Spinner informuje o trwającym pobieraniu. W razie błędu wyświetla ostrzeżenie, ale aplikacja działa dalej (bez cech pogodowych).

Po pobraniu: `agreguj_pogode_tygodniowo` → `dolacz_pogode` → `dodaj_cechy_kalendarza` — dane sprzedażowe są wzbogacone o wszystkie cechy.

---

## 22. Tab 1 — Dane historyczne (linie 1024–1182)

### Metryki nagłówkowe

```python
m1.metric("Wierszy (tygodniowo)", fmt(len(df_tyg_sprzedaz)))
m2.metric("Asortymentów", df_tyg_sprzedaz["Assortment"].nunique())
```

**Co robi:**
4 karty metryczne: liczba wierszy, liczba asortymentów, zakres dat, status cech pogodowych.

### Wykres sezonowości (wieloasortymentowy)

```python
asort_multi = st.multiselect("Asortymenty na wykresie", ..., max_selections=8)
```

**Co robi:**
Pozwala wybrać do 8 asortymentów jednocześnie. Każdy rysowany innym kolorem z palety `KOLORY_MULTI`. Wykres liniowy Plotly z `hovermode="x unified"` — po najechaniu myszy widać wszystkie asortymenty dla danej daty.

### Wykres korelacji sprzedaży z temperaturą

```python
kor = seria_corr["Temp_Max_C"].corr(seria_corr["Sales_Value"], method="spearman")
z = np.polyfit(x, y, 1)   # linia trendu liniowego
```

**Co robi:**
Scatter plot z linią trendu (regresja liniowa stopnia 1) i **korelacją Spearmana** (miara rankingowa — odporna na wartości odstające, lepsza dla danych sprzedażowych niż Pearson). Wyjaśniany za pomocą `st.expander`.

### Tabela statystyk asortymentów

```python
stats = df_tyg_sprzedaz.groupby("Assortment")["Sales_Value"].agg(
    Średnia="mean", Mediana="median", Max="max", N_tygodni="count"
)
```

**Co robi:**
Tabela porównawcza wszystkich asortymentów: średnia, mediana, maksimum i liczba tygodni. Sortowana malejąco po średniej.

---

## 23. Tab 2 — Model i ewaluacja (linie 1188–1485)

### Logika treningu

```python
if run_btn:
    for asort in asort_lista:
        if trenuj_xgb: wyniki_xgb[asort] = trenuj_z_ci(df_a, horyzont)
        if trenuj_prp: wyniki_prophet[asort] = trenuj_prophet(df_a, horyzont, is_impulse)
```

**Co robi:**
Po kliknięciu "▶ Uruchom analizę":
- Iteruje przez wszystkie asortymenty (lub tylko wybrany).
- Pomija asortymenty z < `horyzont + 15` tygodni (minimalna ilość danych).
- Wyświetla pasek postępu `st.progress`.
- Wyniki zapisuje w `st.session_state` — trwają przez całą sesję, Tab 3/4 korzystają z tych samych wyników.

### Tabela wyników z kolorowaniem MAPE

```python
def koloruj(v):
    if v <= 15: return "background-color:#d4edda"
    if v <= 20: return "background-color:#cce5ff"
    ...
st.dataframe(df_wyniki.style.map(koloruj, subset=["MAPE [%]"]))
```

**Co robi:**
Tabela z wynikami ewaluacji dla każdego asortymentu. Kolumna MAPE kolorowana zielono/niebiesko/żółto/czerwono. Jeśli dostępne `best_params` (hiperparametry dobrane przez CV) — pokazane jako dodatkowe kolumny `max_depth*` i `min_child_w*`.

### Porównanie XGBoost vs Prophet

```python
if _wx and _wp:
    # zestawia MAPE obu modeli, podkreśla lepszy
```

**Co robi:**
Widoczna tylko gdy wybrano "Porównanie (oba)". Koloruje wiersze: zielony = lepszy model, czerwony = gorszy. Podsumowuje ile razy wygrywa każdy model.

### Alert jakości prognozy

```python
if res["MAPE"] <= 15: st.success(...)
elif res["MAPE"] <= 20: st.info(...)
elif res["MAPE"] <= 30: st.warning(...)
else: st.error(...)
```

**Co robi:**
Kontekstowy komunikat dla wybranego asortymentu — wyjaśnia co wartość MAPE oznacza praktycznie (czy można automatyzować zamówienia, czy trzeba ręcznie weryfikować).

### Wykres prognoza vs rzeczywistość

```python
fig3.add_trace(go.Scatter(x=dates_tr, y=y_tr_show, ...))  # historia (szara)
fig3.add_trace(go.Scatter(..., fill="toself", ...))         # 80% CI (różowy obszar)
fig3.add_trace(go.Scatter(x=dates_te, y=res["y_test"], ...)) # rzeczywista (niebieska)
fig3.add_trace(go.Scatter(x=dates_te, y=res["pred_50"], ...)) # prognoza (czerwona przerywana)
```

**Co robi:**
"Egzamin modelu" — 4 serie na wykresie: ostatnie 20 tygodni historii, 80% przedział ufności (obszar wypełniony), rzeczywista sprzedaż w oknie testowym, prognoza p50. Dobry model: czerwona linia blisko niebieskiej, niebieska mieści się w różowym.

### Wykres ważności cech

```python
_fi_height = max(400, _n_cech * 28 + 60)
fig4 = go.Figure(go.Bar(x=df_fi["Waga"], y=df_fi["Cecha"], orientation="h", ...))
```

**Co robi:**
Poziomy wykres słupkowy z ważnością cech (gain XGBoost). Wysokość dynamiczna — ~28 px na każdą cechę zapewnia czytelność niezależnie od liczby cech.

---

## 24. Tab 3 — Prognoza T+1...T+8 (linie 1492–2179)

### Wybór daty startowej

```python
data_start_fc = st.date_input("📅 Data startowa prognozy", value=_dzisiaj.date(), ...)
if data_start_ts < _dzisiaj:
    st.caption("🔙 Tryb backtestingu — pogoda z archiwum Open-Meteo")
```

**Co robi:**
Użytkownik wybiera od kiedy liczyć T+1. Jeśli data w przeszłości → tryb backtestingu (pogoda z archiwum). Jeśli dzisiaj lub w przyszłości → prognoza (pogoda z forecast API). Zakres: 8 tygodni wstecz do 4 tygodni w przód.

### Pobieranie i łączenie pogody (archive + forecast)

```python
if data_start_ts <= _dzisiaj:
    df_fc_dzien_arch, _ = pobierz_pogode_historyczna(...)
    df_fc_dzien_fore, _ = pobierz_pogode_forecast(dni=16)
    df_fc_dzien = pd.concat([arch, fore]).drop_duplicates(subset=["time"])
```

**Co robi:**
Łączy archiwum + prognozę forecast w jeden ciągły szereg dzienny (deduplikacja po dacie — archiwum ma pierwszeństwo dla dat w przeszłości, forecast dla przyszłości). Następnie agregacja do tygodniowego poziomu i dodanie cech kalendarza.

### Rozszerzenie do 8 tygodni i cechy sekwencyjne

```python
df_fc_tyg = rozszerz_prognoza_pogody(df_fc_tyg_api, _srednie_pogodowe, data_start_ts, n_tygodni=8)

_hist_pogoda_tail = _pogoda_hist_saved.tail(12)[["Date", "Temp_Max_C", "heat_wave"]].copy()
_serie_pogodowa = pd.concat([_hist_pogoda_tail, _fc_kontekst]).sort_values("Date")
```

**Co robi:**
Po rozszerzeniu do 8 tygodni — przelicza `post_heat_wave` i `first_warm_week` na **połączonym** szeregu historii + prognozy. Ważne: bez kontekstu historycznych tygodni zimowych, `first_warm_week` zawsze byłoby 0 (bo nie wiedziałoby że był długi zimny ciąg). Wyniki mapowane z powrotem do `df_fc_tyg` po dacie.

### n_det — podział horyzontu API vs fallback

```python
n_det = int((df_fc_tyg["zrodlo_pogody"] == "🌤️ Open-Meteo (API)").sum())
n_det = max(1, min(n_det, len(dates_fc)))
```

**Co robi:**
Liczy ile tygodni ma pogodę z API (czyli dane pewne). Tygodnie T+1...T+n_det = "deterministyczne" (solidna linia na wykresie, wchodzą do rekomendacji). Tygodnie T+n_det+1...T+8 = "orientacyjne" (przerywana linia, tylko trend sezonowy).

### Obliczenie prognozy XGBoost

```python
_df_hist_asort = df_tyg_sprzedaz[df_tyg_sprzedaz["Assortment"] == asort_fc].sort_values("Date")
df_fc_tyg_asort = dodaj_lagi_do_prognozy(_df_hist_asort, df_fc_tyg)
X_fc = df_fc_tyg_asort[cechy].values
p50 = np.clip(res["modele"]["p50"].predict(X_fc), 0, None)
```

**Co robi:**
Dla wybranego asortymentu uzupełnia lagi w tabeli prognozy, a następnie predykcja batchowa — wszystkie 8 tygodni naraz (batch predict). `np.clip(..., 0, None)` zapobiega ujemnym prognozom.

### Wykres prognozy

```python
fig_fc.add_trace(...)  # historia (niebieska)
fig_fc.add_trace(...)  # 80% CI (różowy)
fig_fc.add_trace(...)  # T+1..n_det — solidna linia (deterministyczna)
fig_fc.add_trace(...)  # T+n_det+1..8 — przerywana (orientacyjna)
fig_fc.add_shape(...)  # pionowa linia podziału
```

**Co robi:**
Wizualnie dzieli prognozę na deterministyczną i orientacyjną część. Adnotacja tekstowa "◀ deterministyczny | orientacyjny ▶" przy linii podziału.

### Tabela prognozy

```python
df_out_dict = {
    "Tydzień": lbls,
    "Prognoza p50": [...],
    "Dolna p10": [...],
    "Górna p90": [...],
    "Pot. stockout [szt.]": [fmt(max(0, p90[i] - p50[i])) ...],
}
```

**Co robi:**
Tabela z tygodniami T+1..T+8, pogodą, trzema scenariuszami prognozy i potencjalnym brakiem towaru (stockout = p90 - p50). Jeśli wpisano cenę → dodatkowe kolumny wartości w PLN. Kolumna p50 i stockout podświetlone kolorami.

### Rekomendacja zamówienia (Option C)

```python
n_api = n_det
p50_rek = p50[:n_api]
p50_mean = float(p50_rek.mean())
spread_pct = (p90_mean - p50_mean) / p50_mean * 100
```

**Co robi:**
Rekomendacja dotyczy **tylko tygodni z prognozą API** (n_api tygodni). Oblicza:
- Średnie p10/p50/p90 dla tygodni API.
- `spread_pct` — rozpiętość przedziału ufności jako % p50: < 20% = niskie ryzyko, 20–40% = umiarkowane, > 40% = wysokie.
- Porównanie rok do roku (`diff_yoy`): szuka sprzedaży z analogicznych tygodni poprzedniego roku w historii.
- `sug_ilosc` — sugerowana ilość z buforem: p50 przy niskim ryzyku, p50 + 50% buforu przy umiarkowanym, p90 przy wysokim.

Wyniki w kolorowym boxie (`st.success/warning/error`).

### Trend sezonowy (zwinięty ekspander)

```python
if n_api < n_total:
    with st.expander(f"📅 Orientacyjny trend sezonowy — T+{n_api+1}–T+{n_total}", expanded=False):
```

**Co robi:**
Tygodnie poza zasięgiem API (orientacyjne) wyświetlane w **zwiniętym ekspanderze** — domyślnie niewidoczne, żeby nie mylić użytkownika precyzyjną rekomendacją z orientacyjnym trendem.

### Korekty manualne

```python
LOG_PATH = Path(__file__).parent / "korekty_manualne.xlsx"
if st.button("💾 Zapisz korektę"):
    nowy_wpis = pd.DataFrame([{...}])
    df_log = pd.concat([df_log, nowy_wpis])
    df_log.to_excel(LOG_PATH, ...)
```

**Co robi:**
Użytkownik może wpisać własną ilość zamówienia (korektę) z powodem i osobą odpowiedzialną. Log zapisywany do pliku Excel `korekty_manualne.xlsx` w folderze aplikacji. Historia korekt widoczna w ekspanderze z możliwością pobrania i wyczyścenia.

### Analiza czynników i uzasadnienie narracyjne

```python
top_cechy = sorted(fi.items(), key=lambda x: x[1], reverse=True)[:6]
for cecha, waga in top_cechy:
    fc_val = df_fc_tyg[cecha].mean()
    hist_val = df_hist_f[cecha].mean()
    diff_pct = (fc_val - hist_val) / abs(hist_val) * 100
    ikona = "✅ Korzystne" / "⚠️ Niekorzystne" / "➖ Neutralne"
```

**Co robi:**
Dla 6 najważniejszych cech porównuje prognozowaną wartość (np. temperatura w przyszłych tygodniach) z historyczną średnią. Ocena: cecha pogodowa korzystna jeśli temperature jest powyżej normy, deszcz poniżej normy. Wynik: tabela z ocenami + automatyczny tekst narracyjny (`uzasadnienie`) opisujący w prostym języku czemu prognoza jest taka a nie inna.

---

## 25. Tab 4 — Rekomendacje (linie 2185–2472)

### Podsumowanie jakości modeli

```python
m4.metric("Gotowych do wdrożenia", f"{len(doskonale)} / {len(df_w)}")
```

**Co robi:**
5 kart: liczba asortymentów, średni MAPE, najlepsza prognoza, gotowych do wdrożenia (MAPE ≤ 15%), wymagających poprawy (MAPE > 30%).

### Wykres kołowy i słupkowy MAPE

```python
fig_pie = go.Figure(go.Pie(...))
fig_bar = go.Figure(go.Bar(...))
fig_bar.add_vline(x=15, ...)   # linia 15%
fig_bar.add_vline(x=20, ...)   # linia 20%
fig_bar.add_vline(x=30, ...)   # linia 30%
```

**Co robi:**
Dwa wykresy obok siebie: kołowy pokazuje udział każdej kategorii jakości w portfolio, słupkowy — MAPE per asortyment z pionowymi liniami granic kategorii.

### Zbiorcza ważność cech

```python
fi_suma: dict = {}
for r in wyniki.values():
    for k, v in r["feature_importance"].items():
        fi_suma[k] = fi_suma.get(k, 0) + v
fi_avg = {k: v / len(wyniki) for k, v in fi_suma.items()}
```

**Co robi:**
Uśrednia ważność cech ze wszystkich wytrenowanych modeli (dla całego portfolio). Pokazuje które czynniki globalnie najsilniej wpływają na sprzedaż lodów. Wykres poziomy, wysokość dynamiczna, font 13px dla czytelności.

### Klasyfikacja ABC/XYZ

```python
# ABC — po skumulowanym wolumenie sprzedaży
sprzedaz_srednia["cum_pct"] = sprzedaz_srednia["Sales_Value"].cumsum() / total * 100
# XYZ — po współczynniku zmienności (std/mean)
cv = df_tyg_sprzedaz.groupby("Assortment")["Sales_Value"].agg(
    lambda x: x.std() / x.mean() * 100 if x.mean() > 0 else 0)
```

**Co robi:**
- **ABC** — sortuje asortymenty malejąco po średniej sprzedaży, oblicza skumulowany udział w całości:
  - A = pierwsze produkty generujące łącznie 70% sprzedaży.
  - B = kolejne do 90%.
  - C = pozostałe 10%.
- **XYZ** — współczynnik zmienności (CV = odchylenie standardowe / średnia):
  - X = CV < 50% (stabilna, łatwa do prognozowania).
  - Y = CV 50–100% (umiarkowanie zmienna, sezonowa).
  - Z = CV > 100% (bardzo nieregularna).
- Połączenie daje 9 klas (AX, AY, ..., CZ) z kolorowaniem i opisem dla każdej.

### Plan działań

```python
st.markdown(f"""
| Priorytet | Działanie | Dla kogo |
| 1 🟢 | Wdrożyć automatyczną prognozę | {len(doskonale)} asortymentów |
...
""")
```

**Co robi:**
Konkretne, spriorytetyzowane rekomendacje wdrożeniowe — automatycznie obliczone na podstawie wyników modeli.

---

## 26. Tab 5 — Actual vs Forecast (linie 2478–2712)

### Wczytanie pliku z aktualną sprzedażą

```python
avf_file = st.file_uploader("Wgraj plik Excel z rzeczywistą sprzedażą", type=["xlsx", "csv"])
```

**Co robi:**
Użytkownik wgrywa plik Excel lub CSV z kolumnami: asortyment, data tygodnia, rzeczywista sprzedaż. Aplikacja automatycznie próbuje dopasować kolumny (`_avf_guess`).

### Łączenie prognozy z aktualami

```python
df_prog = pd.DataFrame([{"Asortyment": a, "Data": d, "Prognoza_p50": p, ...}
                         for a, res in wyniki_avf.items() for d, p in zip(daty, preds)])
df_merge = df_act.merge(df_prog, on=["Asortyment", "ISO_Week", "Year"], how="inner")
```

**Co robi:**
Ekstrahuje prognozy `pred_50` ze zbioru testowego (ostatnie `horyzont` tygodni z fazy treningu) i łączy je z wgraną rzeczywistą sprzedażą po `(Asortyment, ISO_Week, Year)`. Inner join = tylko tygodnie gdzie jest i prognoza i aktual.

### Obliczanie błędów

```python
df_merge["Błąd [szt.]"] = df_merge["Actual"] - df_merge["Prognoza_p50"]
df_merge["MAPE_row"] = AbsBłąd / Actual * 100   # per wiersz
df_merge["Bias_dir"] = "📈 Niedoszacowanie" / "📉 Przeszacowanie" / "➖ Trafienie"
```

**Co robi:**
- Błąd = Actual - Prognoza (dodatni = model niedoszacował = brakowało towaru, ujemny = przeszacował = za dużo zapasów).
- MAPE per tydzień (row-level).
- Kierunek bias (systematyczne niedoszacowanie lub przeszacowanie).

### Wykres słupkowy prognoza vs rzeczywistość

```python
fig_avf.add_trace(go.Bar(y=df_avf_a["Prognoza_p50"], ...))  # niebieski
fig_avf.add_trace(go.Bar(y=df_avf_a["Actual"], ...))         # czerwony
fig_avf.add_trace(go.Scatter(y=df_avf_a["MAPE_row"], yaxis="y2", ...))  # żółta linia MAPE
```

**Co robi:**
Zgrupowane słupki (prognoza vs rzeczywistość) z drugą osią Y dla MAPE w procentach. Łatwa wizualna ocena gdzie model się mylił.

### Ranking dokładności per asortyment

```python
ranking = df_merge.groupby("Asortyment").agg(
    MAPE_sr=("MAPE_row", "mean"),
    MAE_sr=("AbsBłąd", "mean"),
    Bias_sr=("Błąd [szt.]", "mean"),
    N_tygodni=("Data_act", "nunique"),
)
```

**Co robi:**
Agreguje metryki per asortyment — średnie MAPE, MAE, bias i liczba tygodni. Sortowanie od najdokładniejszego. Kolorowanie tła (zielony ≤ 15%, żółty 15–25%, czerwony > 25%).

### Download zestawienia

```python
st.download_button("⬇ Pobierz pełne zestawienie actual vs forecast (.xlsx)", ...)
```

**Co robi:**
Eksportuje pełną tabelę porównawczą (wszystkie asortymenty, wszystkie tygodnie, kolumny: prognoza, actual, błąd, MAPE, kierunek) do pliku Excel do dalszej analizy lub archiwizacji.

---

## Podsumowanie architektury przepływu danych

```
1. Sidebar: wczytanie pliku Excel → standaryzacja kolumn → agregacja tygodniowa
                                 ↓
2. Open-Meteo Archive API → pogoda historyczna → agregacja tygodniowa
   + cechy kalendarza (święta, Wielkanoc, majówka, wakacje) + lagi sprzedaży
                                 ↓
3. Tab 2 (po kliknięciu Uruchom):
   XGBoost:
     - Strojenie hiperparametrów (TimeSeriesSplit 3-fold CV)
     - Trening 3 kwantyli (p10, p50, p90)
     - Ewaluacja (MAPE, MAE, R²) na zbiorze hold-out
   Prophet:
     - Sezonowość multiplikatywna + regresory pogodowe/kalendarza
     - Ewaluacja analogiczna
                                 ↓
4. Tab 3: Open-Meteo Forecast (16 dni) + historyczne średnie (fallback)
   → Rozszerzenie do 8 tygodni → Prognoza ML per asortyment
   → Rekomendacja zamówienia (tygodnie API) + trend sezonowy (tygodnie fallback)
                                 ↓
5. Tab 4: Agregacja feature importance + klasyfikacja ABC/XYZ + plan działań
                                 ↓
6. Tab 5: Porównanie prognoz z wgraną rzeczywistą sprzedażą → ranking błędów
```

---

*Dokument wygenerowany automatycznie — Prognoza Sprzedaży Lodów · Grupa IV AIFINC 2026 · ALK*
