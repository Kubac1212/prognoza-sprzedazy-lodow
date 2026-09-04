# 🍦 Prognoza Sprzedaży Lodów

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.57.0-red.svg)](https://streamlit.io)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![XGBoost](https://img.shields.io/badge/XGBoost-3.2.0-green.svg)](https://xgboost.readthedocs.io/)

Zaawansowana aplikacja do prognozowania tygodniowej sprzedaży lodów oparta na **modelach Machine Learning** (XGBoost + Prophet), **danych pogodowych** (Open-Meteo API) i inteligentnie wybranych **cechach kalendarza**. 

Aplikacja działa **w przeglądarce** dzięki Streamlit — bez konieczności instalowania czegokolwiek poza Pythonem.

---

## 📋 Spis treści

- [✨ Cechy](#-cechy)
- [🚀 Szybki start](#-szybki-start)
- [📋 Wymagania](#-wymagania)
- [🔧 Instalacja](#-instalacja)
- [📖 Użytkowanie](#-użytkowanie)
- [🏗️ Architektura](#-architektura)
- [📦 Zależności](#-zależności)
- [🌐 API](#-api)
- [📊 Struktura projektu](#-struktura-projektu)
- [🤝 Contributing](#-contributing)
- [📄 License](#-license)
- [✍️ Autorzy](#-autorzy)

---

## ✨ Cechy

✅ **Modele hybrydowe** — XGBoost dla dokładności + Prophet dla trendów sezonowych  
✅ **Przedziały ufności** — Kwantyle p10/p50/p90 dla każdej prognozy  
✅ **Rekomendacje zamówień** — Inteligentne sugestie z oceną ryzyka  
✅ **Analiza cech** — Feature importance oraz klasyfikacja ABC/XYZ asortymentów  
✅ **Pogoda w real-time** — Integracja z Open-Meteo API (brak potrzeby klucza API)  
✅ **Święta i kalendarz** — Polskie święta ustawowe, Wielkanoc, wakacje szkolne  
✅ **Korekty manualne** — Możliwość dostosowania prognoz z automatycznym zapisem  
✅ **Backtest** — Porównanie rzeczywistych sprzedaży z prognozami  

---

## 🚀 Szybki start

### Windows

```bash
# 1. Pobierz repozytorium
git clone https://github.com/twój-user/prognoza-sprzedazy-lodow.git
cd prognoza-sprzedazy-lodow

# 2. Uruchom skrypt
run.bat
```

**To wszystko!** Aplikacja zainstaluje zależności i otworzy się w przeglądarce.

### macOS / Linux

```bash
# 1. Pobierz repozytorium
git clone https://github.com/twój-user/prognoza-sprzedazy-lodow.git
cd prognoza-sprzedazy-lodow

# 2. Uruchom skrypt
bash run.sh
```

Aplikacja będzie dostępna pod adresem **http://localhost:8501**

---

## 📋 Wymagania

| Wymóg | Wersja |
|-------|--------|
| Python | 3.10+ |
| Połączenie internetowe | Wymagane (Open-Meteo API) |
| Miejsce na dysku | ~500 MB |
| RAM | Minimum 2 GB (zalecane 4 GB) |

---

## 🔧 Instalacja

### Instalacja manualna (zaawansowani użytkownicy)

```bash
# 1. Klonowanie repozytorium
git clone https://github.com/twój-user/prognoza-sprzedazy-lodow.git
cd prognoza-sprzedazy-lodow

# 2. Utworzenie środowiska wirtualnego
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Instalacja zależności
pip install -r requirements.txt

# 4. Uruchomienie aplikacji
streamlit run app_prognoza_lodow.py
```

### Rozwiązywanie problemów

**Problem**: „Python nie znaleziony"  
**Rozwiązanie**: Upewnij się, że Python jest w PATH → [https://docs.python.org/3/using/windows.html#finding-the-python-executable](https://docs.python.org/3/using/windows.html#finding-the-python-executable)

**Problem**: Błędy z biblioteka Prophet  
**Rozwiązanie**: `pip install --upgrade cmdstanpy`

**Problem**: Brak dostępu do internetu / Open-Meteo API  
**Rozwiązanie**: Sprawdź połączenie sieciowe; API jest bezpłatne i nie wymaga rejestracji

---

## 📖 Użytkowanie

### Przepływ pracy

1. **Tab 1 — Wgrano dane**
   - Wgraj plik Excel ze sprzedażą (kolumny: `data`, `asortyment`, `sprzedaż_szt`)
   - Aplikacja automatycznie agreguje do tygodni i łączy z pogodą

2. **Tab 2 — Trenuj modele**
   - Wybierz produkty impulsowe/familijne (do optymalnego ustawienia Prophet)
   - Kliknij „Trenuj XGBoost" oraz „Trenuj Prophet"
   - Przejrzyj metryki: MAPE, MAE, R² oraz feature importance

3. **Tab 3 — Prognoza**
   - Prognoza T+1…T+8 tygodni
   - Scenariusze: konserwatywny/średni/optymistyczny
   - Rekomendacja ilości zamówienia z oceną ryzyka

4. **Tab 4 — Rekomendacje**
   - Globalna ważność cech (średnia z wszystkich modeli)
   - Klasyfikacja ABC (wolumen) / XYZ (zmienność) asortymentów

5. **Tab 5 — Actual vs Forecast**
   - Wgraj rzeczywiste sprzedaże
   - Porównanie modelu z rzeczywistością (MAPE, MAE, bias)

---

## 🏗️ Architektura

### Przepływ danych

```
┌─────────────────────────────────────────────────────────────────┐
│ Użytkownik wgrywa plik Excel (data, asortyment, sprzedaż)      │
└────────────────────────┬────────────────────────────────────────┘
                         ↓
         ┌───────────────────────────────────┐
         │ Agregacja do tygodni              │
         │ + Pobierz pogodę historyczną      │
         └────────────┬────────────────────┬─┘
                      ↓                    ↓
       ┌──────────────────────┐  ┌─────────────────────┐
       │ Dodaj cechy kalendarza│  │ Dodaj cechy pogody  │
       │ (święta, wakacje)    │  │ (temp, opady, wiatr)│
       └──────────┬───────────┘  └────────┬────────────┘
                  └─────────────┬──────────┘
                                ↓
                  ┌──────────────────────────────┐
                  │ Treningowe modele ML:        │
                  │ • XGBoost (3 kwantyle)       │
                  │ • Prophet (sezonowość)       │
                  │ • TimeSeriesSplit CV         │
                  │ • Feature importance         │
                  └────────┬─────────────────────┘
                           ↓
            ┌──────────────────────────────────┐
            │ Prognoza T+1…T+8 (Tab 3)         │
            │ • Pobierz prognozę pogody        │
            │ • Utwórz scenariusze             │
            │ • Rekomendacja zamówienia        │
            └──────────────────────────────────┘
```

### Kluczowe komponenty

| Moduł | Odpowiedzialność |
|-------|-----------------|
| `pobierz_pogode_*` | Integracja z Open-Meteo Archive/Forecast API |
| `dodaj_cechy_kalendarza` | Polskie święta, ferie szkolne, Wielkanoc, majówka |
| `dodaj_lagi_sprzedazy` | Cykliczność roczna (lag_52), miesięczna (lag_4), MA4 |
| `trenuj_z_ci` | XGBoost z strojeniem hiperparametrów (TimeSeriesSplit) |
| `trenuj_prophet` | Prophet z sezonowością multiplikatywną i regresorami |
| `prognozuj_*` | Generowanie prognoz i przedziałów ufności |

---

## 📦 Zależności

```txt
streamlit==1.57.0           # UI i interaktywność
pandas==3.0.2              # Przetwarzanie danych
numpy==2.4.4               # Operacje numeryczne
xgboost==3.2.0             # Modele kwantylowe
prophet==1.3.0             # Szeregi czasowe + sezonowość
scikit-learn==1.8.0        # TimeSeriesSplit do CV
plotly==6.7.0              # Wykresy interaktywne
holidays==0.96             # Polskie święta
requests==2.33.1           # Zapytania do API
openpyxl==3.1.5            # Obsługa Excel
```

Pełna lista: patrz `requirements.txt`

---

## 🌐 API

Aplikacja wykorzystuje **darmowe, publiczne API**:

### Open-Meteo Archive API
- **Endpoint**: `https://archive-api.open-meteo.com/v1/archive`
- **Zastosowanie**: Pobieranie danych pogodowych historycznych (treningu)
- **Parametry**: data_start, data_end, latitude, longitude, zmienne pogodowe
- **Bez klucza API** ✅

### Open-Meteo Forecast API
- **Endpoint**: `https://api.open-meteo.com/v1/forecast`
- **Zastosowanie**: Prognoza pogody na 16 dni
- **Bez klucza API** ✅

### Dodatkowe opcje
- **OpenWeatherMap** (opcjonalnie) — pole w sidebarze do wklejenia klucza API
- **Polskie święta** — biblioteka `holidays` (offline)

---

## 📊 Struktura projektu

```
prognoza-sprzedazy-lodow/
│
├── app_prognoza_lodow.py        # Główna aplikacja (~2700 linii)
│                                # Sekcje:
│                                # - UI (Streamlit)
│                                # - Pobieranie pogody
│                                # - Inżynieria cech
│                                # - Treningu modeli
│                                # - Generowanie prognoz
│                                # - Rekomendacje
│
├── requirements.txt             # Zależności Python z wersjami
│
├── run.bat                      # Skrypt startowy dla Windows
│                                # - Sprawdza Python 3.10+
│                                # - Tworzy venv
│                                # - Instaluje zależności
│                                # - Uruchamia Streamlit
│
├── run.sh                       # Skrypt startowy dla macOS/Linux
│                                # Analogicznie do run.bat
│
├── README.md                    # Ten plik
│
├── LICENSE                      # MIT License
│
├── .gitignore                   # Standardowe ignorowanie (venv, __pycache__, *.pyc)
│
└── korekty_manualne.xlsx        # Tworzony automatycznie po pierwszej korekcie
                                # Zapis edycji użytkownika między sesjami
```

---

## 🎯 Przykład użycia

### Przygotowanie danych

Przygotuj plik Excel (`sprzedaz.xlsx`) z kolumnami:

```
| data       | asortyment      | sprzedaz_szt |
|------------|-----------------|--------------|
| 2024-01-01 | Lody waniliowe  | 150          |
| 2024-01-02 | Lody waniliowe  | 160          |
| 2024-01-01 | Lody czekoladowe| 80           |
| ...        | ...             | ...          |
```

### Uruchomienie

1. **Windows**: Kliknij dwukrotnie `run.bat`
2. **macOS/Linux**: `bash run.sh`
3. Wgraj plik Excel w Tab 1
4. Trenuj modele w Tab 2
5. Przejrzyj prognozy w Tab 3
6. Działaj na podstawie rekomendacji w Tab 4

---

## 🚀 Wydajność

### Metryki dokładności

Aplikacja raportuje dla każdego asortymentu:

- **MAPE** (Mean Absolute Percentage Error) — % błędu
- **MAE** (Mean Absolute Error) — średnia wartość błędu
- **R²** — wyjaśniona wariancja modelu

### Czasy operacji

| Operacja | Czas |
|----------|------|
| Wgranie danych (1000 wierszy) | ~1 s |
| Pobierz pogodę historyczną | ~5-10 s |
| Treningi XGBoost (10 asortymentów) | ~30-60 s |
| Treningi Prophet (10 asortymentów) | ~20-40 s |
| Generowanie prognozy T+8 | ~5-10 s |

---

## 🤝 Contributing

Wkład jest mile widziany! Aby zaproponować zmiany:

1. **Fork** repozytorium
2. Utwórz branch (`git checkout -b feature/AmazingFeature`)
3. Zacommituj zmiany (`git commit -m 'Add AmazingFeature'`)
4. Push do brancha (`git push origin feature/AmazingFeature`)
5. Otwórz **Pull Request**

### Kierunki rozwoju

- [ ] Integracja z Shopify / WooCommerce API
- [ ] Eksport prognoz do Google Sheets
- [ ] Modelowanie zmienności (model GARCH)
- [ ] Optymalizacja kosztu magazynowania
- [ ] Przewidywanie stockoutów (alert system)
- [ ] Wsparcie dla wielu lokalizacji geograficznych

---

## 📄 License

Projekt jest objęty licencją **MIT**. Patrz plik `LICENSE` po szczegóły.

---

## ✍️ Autorzy

- **Grupa IV** — AIFINC 2026 · ALK
- **Kuba** — główny deweloper

---

## 📞 Kontakt i wsparcie

Jeśli napotkasz problemy lub masz pytania:

1. **Issues** — otwórz [GitHub Issue](https://github.com/twój-user/prognoza-sprzedazy-lodow/issues)
2. **Dokumentacja** — przeczytaj sekcję [Użytkowanie](#-użytkowanie)
3. **Email** — twój-email@example.com

---

## 🙏 Podziękowania

- **Open-Meteo** — za bezpłatne API pogodowe
- **Streamlit** — za wspaniały framework
- **XGBoost & Prophet** — za zaawansowane modele ML
- Społeczności Python i open-source!

---

## 📈 Roadmap (v2.0)

- ✅ **v1.0** — Core XGBoost + Prophet
- 🔄 **v1.1** — Optymalizacja hyperparametrów (planowana Q4 2026)
- 🚀 **v2.0** — REST API + baza danych (PostgreSQL)
- 📱 **v2.1** — Mobilna aplikacja (React Native)

---

<div align="center">

**[⬆ Powrót na górę](#-prognoza-sprzedaży-lodów)**

Made with ❤️ by Grupa IV AIFINC 2026

</div>
