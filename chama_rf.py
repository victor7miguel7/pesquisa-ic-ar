# inferencia_rf.py

import pandas as pd
import numpy as np
import joblib
import time
import psutil
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

print("==> Carregando dados CSV...")
df = pd.read_csv('df_concatenado.csv')
df = df.loc[:, df.isnull().sum() == 0]

print("==> Extraindo características temporais...")
df['Datetime_start'] = pd.to_datetime(df['Datetime_start'])
df['Year'] = df['Datetime_start'].dt.year
df['Month'] = df['Datetime_start'].dt.month
df['Day'] = df['Datetime_start'].dt.day
df['Hour'] = df['Datetime_start'].dt.hour
df['Weekday'] = df['Datetime_start'].dt.weekday
df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)

print("==> Aplicando regra das estações...")
def get_season(month):
    if month in [12, 1, 2]:
        return 'Verão'
    elif month in [3, 4, 5]:
        return 'Outono'
    elif month in [6, 7, 8]:
        return 'Inverno'
    else:
        return 'Primavera'

df['Season'] = df['Month'].apply(get_season)

print("==> Removendo colunas desnecessárias...")
df = df.drop([
    'Datetime_start', 'Datetime_end', 'Timezone', 'Temperature (Fahrenheit)', 'AQI CN',
    "PM2.5 (ug/m3)", 'PM10 (ug/m3)', 'PM1 (ug/m3)', 'slot.4.pm1', 'slot.4.pm10',
    'slot.4.pm25', 'Particle Count', 'slot.4.pc', 'Pressure (pascal)'
], axis=1)

print("==> Codificando variáveis categóricas...")
le = LabelEncoder()
df['Season'] = le.fit_transform(df['Season'])
df['Source'] = le.fit_transform(df['Source'])

print("==> Separando features e target...")
X = df.drop('AQI US', axis=1)
y = df['AQI US']

print("==> Separando treino e teste...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

print("==> Carregando modelo treinado (random_forest_model.joblib)...")
model = joblib.load('random_forest_model.joblib')

# Benchmark
def benchmark_inference(model, X_test, y_test, workload, duration_seconds=300):
    print(f"\n==> Iniciando benchmark para workload {workload}...")
    n_samples = int(len(X_test) * workload)
    X_benchmark = X_test[:n_samples]
    y_benchmark = y_test[:n_samples]

    start_time = time.time()
    inference_times = []
    predictions = []
    memory_usages = []

    while time.time() - start_time < duration_seconds:
        t0 = time.perf_counter()
        y_pred = model.predict(X_benchmark)
        t1 = time.perf_counter()

        inference_times.append(t1 - t0)
        predictions.append(y_pred)
        memory_usages.append(psutil.Process().memory_info().rss / 1024**2)  # Em MB

    print("==> Finalizando benchmark e calculando métricas...")

    y_pred_final = predictions[-1]
    mae = mean_absolute_error(y_benchmark, y_pred_final)
    rmse = root_mean_squared_error(y_benchmark, y_pred_final)
    r2 = r2_score(y_benchmark, y_pred_final)

    avg_time = np.mean(inference_times)
    avg_memory = np.mean(memory_usages)
    num_inferences = len(inference_times)

    print(f"\n=== Resultados para workload {workload} ===")
    print(f"MAE: {mae:.4f}")
    print(f"RMSE: {rmse:.4f}")
    print(f"R²: {r2:.4f}")
    print(f"Tempo médio de inferência: {avg_time*1000:.2f} ms")
    print(f"Uso médio de memória: {avg_memory:.2f} MB")
    print(f"Número de inferências em {duration_seconds} segundos: {num_inferences}")

# Executar benchmark
for workload in [1.0, 0.5, 0.1]:
    benchmark_inference(model, X_test, y_test, workload, duration_seconds=300)
