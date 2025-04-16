# 🏎️ F1 Japanese GP Qualifying Prediction (2025)

This project predicts Q3 qualifying lap times for the 2025 Japanese Grand Prix using historical data and machine learning. It leverages the FastF1 library to fetch real-time F1 data and builds a predictive model based on Q1 and Q2 lap times.

##  Overview

- **Goal**: Predict final Q3 lap times for drivers in the 2025 Japanese GP.
- **Approach**: Use a **Linear Regression** model trained on past qualifying data.
- **Tech**: FastF1, pandas, scikit-learn, matplotlib.

---

##  How It Works

1. **Data Collection**:
   - Fetches qualifying data from:
     - 2024 Japanese GP
     - 2025 Season Rounds 1–4
   - Uses FastF1’s API and session caching.

2. **Preprocessing**:
   - Cleans and merges Q1, Q2, and Q3 lap times.
   - Applies **median imputation** for missing Q1/Q2/Q3 values.

3. **Modeling**:
   - Trains a **Linear Regression** model with:
     - Features: Q1 and Q2 times
     - Target: Q3 times

4. **Prediction**:
   - Applies dynamic **driver and team multipliers** to simulate realistic changes in performance.
   - Predicts Q3 times for all drivers in the 2025 Japanese GP.

5. **Evaluation**:
   - Evaluates model performance using:
     - **Mean Absolute Error (MAE)**
     - **R² Score**
   - Outputs a **predicted qualifying grid**.

---

##  Requirements

- Python 3.8+
- Libraries:
  - `fastf1`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `seaborn` (optional for extra visuals)

Install them via:

```bash
pip install fastf1 pandas numpy scikit-learn matplotlib seaborn
