# treinar_modelo.py

import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.preprocessing import LabelEncoder

# Carregar os dados
df = pd.read_csv('df_concatenado.csv')

# Filtrar colunas sem nulos
df = df.loc[:, df.isnull().sum() == 0]

# Extrair features temporais
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

# Remover colunas irrelevantes
df = df.drop(['Datetime_start', 'Datetime_end','Timezone','Temperature (Fahrenheit)', 'AQI CN',
              "PM2.5 (ug/m3)",'PM10 (ug/m3)', 'PM1 (ug/m3)','slot.4.pm1','slot.4.pm10',
              'slot.4.pm25','Particle Count','slot.4.pc','Pressure (pascal)'], axis=1)

# Encoding de categorias
le = LabelEncoder()
df['Season'] = le.fit_transform(df['Season'])
df['Source'] = le.fit_transform(df['Source'])

# Separar features e alvo
X = df.drop('AQI US', axis=1)
y = df['AQI US']

# Dividir em treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Modelo base
model = xgb.XGBRegressor(objective='reg:squarederror')

# Espaço de busca de hiperparâmetros
param_grid = {
    'n_estimators': [100, 200, 300, 400],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'max_depth': [3, 5, 7, 10],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Otimização com RandomizedSearchCV
random_search = RandomizedSearchCV(
    model,
    param_distributions=param_grid,
    n_iter=20,
    scoring='r2',
    cv=10,
    verbose=1,
    n_jobs=-1
)

# Treinamento
random_search.fit(X_train, y_train)
best_params = random_search.best_params_
print("Melhores hiperparâmetros:", best_params)

# Treinar com os melhores parâmetros
best_model = xgb.XGBRegressor(
    **best_params,
    objective='reg:squarederror'
)
best_model.fit(X_train, y_train)

# Salvar o modelo com joblib
joblib.dump(best_model, 'xgb_model_treinado.joblib')
print("Modelo salvo com sucesso.")
