import os
import fastf1
import pandas as pd
import numpy as np

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score

import fastf1
print(fastf1.__version__)

import os

cache_dir = '/content/cache'
os.makedirs(cache_dir, exist_ok=True)
fastf1.Cache.enable_cache(cache_dir)

fastf1.Cache.enable_cache('cache')

def fetch_f1_data(year, round_number):
    """Fetch data using official F1 API via FastF1"""
    try:
        quali = fastf1.get_session(year, round_number, 'Q')
        quali.load()
        print(f"Fetched data for year {year}, round {round_number}")
        print("DataFrame columns available:", quali.results.columns.tolist())

        results = quali.results[['DriverNumber', 'FullName', 'TeamName', 'Q1', 'Q2', 'Q3']]

        results = results.rename(columns={'FullName': 'Driver'})

        for col in ['Q1', 'Q2', 'Q3']:
            results[col] = results[col].apply(lambda x: x.total_seconds() if pd.notnull(x) else None)

        results=results.rename(columns={'Q1':'Q1_sec','Q2':'Q2_sec','Q3':'Q3_sec'})

        print("\nQualifying Results Structure:")
        print(results.head())

        return results
    except Exception as e:
        print(f"Error fetching data: {e}")
        print("DataFrame columns available:", quali.results.columns.tolist())
        return None

dt = fetch_f1_data(2023, 5)

def fetch_recent_data():
    """Fetch data from recent races using FastF1"""
    all_data = []


    current_year = 2025
    for round_num in range(1, 5):  # First 4 races of 2025
        print(f"Fetching data for {current_year} round {round_num}...")
        df = fetch_f1_data(current_year, round_num)
        if df is not None:
            df['Year'] = current_year
            df['Round'] = round_num
            all_data.append(df)


    print("Fetching 2024 Japanese GP data...")
    japan_2024 = fetch_f1_data(2024, 4)
    if japan_2024 is not None:
        japan_2024['Year'] = 2024
        japan_2024['Round'] = 4
        all_data.append(japan_2024)

    return all_data

def compute_performance_factors(df, model):
    """Compute dynamic performance multipliers from real data"""

    # Average base Q1/Q2 to predict a 'base' Q3 using model
    avg_q1 = df['Q1_sec'].mean()
    avg_q2 = df['Q2_sec'].mean()
    base_input = pd.DataFrame([[avg_q1, avg_q2]], columns=['Q1_sec', 'Q2_sec'])
    base_time = model.predict(base_input)[0]

    # Filter only rows with valid Q3
    valid_df = df.dropna(subset=['Q3_sec'])

    # Team factors
    team_avg = valid_df.groupby('TeamName')['Q3_sec'].mean()
    team_factors = (team_avg / base_time).to_dict()

    # Driver factors
    driver_avg = valid_df.groupby('Driver')['Q3_sec'].mean()
    driver_factors = (driver_avg / base_time).to_dict()

    return team_factors, driver_factors

def predict_japanese_gp(model, latest_data):
    """Predict Q3 times for Japanese GP 2025 using computed multipliers."""

    # Get dynamic multipliers
    team_factors, driver_factors = compute_performance_factors(latest_data, model)

    driver_teams = {
        'Max Verstappen': 'Red Bull Racing',
        'Sergio Perez': 'Red Bull Racing',
        'Charles Leclerc': 'Ferrari',
        'Carlos Sainz': 'Ferrari',
        'Lewis Hamilton': 'Mercedes',
        'George Russell': 'Mercedes',
        'Lando Norris': 'McLaren',
        'Oscar Piastri': 'McLaren',
        'Fernando Alonso': 'Aston Martin',
        'Lance Stroll': 'Aston Martin',
        'Daniel Ricciardo': 'RB',
        'Yuki Tsunoda': 'RB',
        'Alexander Albon': 'Williams',
        'Logan Sargeant': 'Williams',
        'Valtteri Bottas': 'Kick Sauber',
        'Zhou Guanyu': 'Kick Sauber',
        'Kevin Magnussen': 'Haas F1 Team',
        'Nico Hulkenberg': 'Haas F1 Team',
        'Pierre Gasly': 'Alpine',
        'Esteban Ocon': 'Alpine'
    }

    results_df = pd.DataFrame(list(driver_teams.items()), columns=['Driver', 'Team'])

    # Base Q3 time prediction
    avg_q1 = latest_data['Q1_sec'].mean()
    avg_q2 = latest_data['Q2_sec'].mean()
    base_time = model.predict(pd.DataFrame([[avg_q1, avg_q2]], columns=['Q1_sec', 'Q2_sec']))[0]

    predicted_times = []
    for _, row in results_df.iterrows():
        driver = row['Driver']
        team = row['Team']
        driver_factor = driver_factors.get(driver, 1.0)
        team_factor = team_factors.get(team, 1.0)
        random_noise = np.random.uniform(-0.1, 0.1)
        predicted_q3 = base_time * team_factor * driver_factor + random_noise
        predicted_times.append(predicted_q3)

    results_df['Predicted_Q3'] = predicted_times
    results_df = results_df.sort_values('Predicted_Q3').reset_index(drop=True)

    print("\n🇯🇵 Japanese GP 2025 Qualifying Predictions:")
    print("=" * 100)
    print(f"{'Position':<10}{'Driver':<20}{'Team':<25}{'Predicted Q3':<15}")
    print("-" * 100)

    for idx, row in results_df.iterrows():
        print(f"{idx+1:<10}{row['Driver']:<20}{row['Team']:<25}{row['Predicted_Q3']:.3f}s")

print("Initializing enhanced F1 prediction model...")
all_data = fetch_recent_data()

if all_data:

        combined_df = pd.concat(all_data, ignore_index=True)


        valid_data = combined_df.dropna(subset=['Q1_sec', 'Q2_sec', 'Q3_sec'], how='all')

        imputer = SimpleImputer(strategy='median')


        X = valid_data[['Q1_sec', 'Q2_sec']]
        y = valid_data['Q3_sec']


        X_clean = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        y_clean = pd.Series(imputer.fit_transform(y.values.reshape(-1, 1)).ravel())

        model = LinearRegression()
        model.fit(X_clean, y_clean)

        predict_japanese_gp(model, valid_data)

        y_pred = model.predict(X_clean)
        mae = mean_absolute_error(y_clean, y_pred)
        r2 = r2_score(y_clean, y_pred)

        print("\nModel Performance Metrics:")
        print(f'Mean Absolute Error: {mae:.2f} seconds')
        print(f'R^2 Score: {r2:.2f}')
else:
        print("Failed to fetch F1 data")