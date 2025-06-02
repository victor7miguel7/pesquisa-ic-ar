# train_model.py

import pandas as pd
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder
from joblib import dump
import warnings

warnings.filterwarnings('ignore')

# Carregar e preparar os dados
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
    return 'Primavera'

df['Season'] = df['Month'].apply(get_season)

df = df.drop([
    'Datetime_start', 'Datetime_end', 'Timezone',
    'Temperature (Fahrenheit)', 'AQI CN', "PM2.5 (ug/m3)",
    'PM10 (ug/m3)', 'PM1 (ug/m3)', 'slot.4.pm1', 'slot.4.pm10',
    'slot.4.pm25', 'Particle Count', 'slot.4.pc', 'Pressure (pascal)'
], axis=1)

le = LabelEncoder()
df['Season'] = le.fit_transform(df['Season'])
df['Source'] = le.fit_transform(df['Source'])

X = df.drop('AQI US', axis=1)
y = df['AQI US']

X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)

# Otimização de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [-1, 5, 10, 15],
    'num_leaves': [20, 31, 40, 50],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

model = lgb.LGBMRegressor(objective='regression')

search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='r2',
    cv=10,
    verbose=1,
    n_jobs=-1,
    random_state=42
)

search.fit(X_train, y_train)

# Treinar o modelo final com os melhores hiperparâmetros
best_model = lgb.LGBMRegressor(**search.best_params_)
best_model.fit(X_train, y_train)

# Salvar modelo treinado
dump(best_model, 'best_lgbm_model.joblib')
print("Modelo treinado e salvo como 'best_lgbm_model.joblib'")
