# treinar_rf.py

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
import joblib

# 1. Carregamento e preparação dos dados
df = pd.read_csv('df_concatenado.csv')
df = df.loc[:, df.isnull().sum() == 0]

df['Datetime_start'] = pd.to_datetime(df['Datetime_start'])
df['Year'] = df['Datetime_start'].dt.year
df['Month'] = df['Datetime_start'].dt.month
df['Day'] = df['Datetime_start'].dt.day
df['Hour'] = df['Datetime_start'].dt.hour
df['Weekday'] = df['Datetime_start'].dt.weekday
df['Is_Weekend'] = df['Weekday'].isin([5, 6]).astype(int)

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

# Remover colunas desnecessárias
df = df.drop([
    'Datetime_start', 'Datetime_end', 'Timezone', 'Temperature (Fahrenheit)', 'AQI CN',
    "PM2.5 (ug/m3)", 'PM10 (ug/m3)', 'PM1 (ug/m3)', 'slot.4.pm1', 'slot.4.pm10',
    'slot.4.pm25', 'Particle Count', 'slot.4.pc', 'Pressure (pascal)'
], axis=1)

# Codificação
le = LabelEncoder()
df['Season'] = le.fit_transform(df['Season'])
df['Source'] = le.fit_transform(df['Source'])

# Features e alvo
X = df.drop('AQI US', axis=1)
y = df['AQI US']

# Dividir dados
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'max_depth': [10, 20, 30, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['auto', 'sqrt', 'log2']
}

# Random Search
model = RandomForestRegressor(random_state=42)
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='r2',
    cv=10,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

random_search.fit(X_train, y_train)
print("Melhores hiperparâmetros:", random_search.best_params_)

# Treinar modelo final
best_model = RandomForestRegressor(**random_search.best_params_, random_state=42)
best_model.fit(X_train, y_train)

# Salvar modelo
joblib.dump(best_model, 'random_forest_model.joblib')
print("Modelo Random Forest salvo com sucesso.")
