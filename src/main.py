import pandas as pd
import numpy as np
import preprocessing as pp
from models import get_logistic_model, get_logistic_model_improved
from evaluate import evaluate_model
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import BorderlineSMOTE, SMOTE
from imblearn.combine import SMOTEENN

air_pollution_df_2025 = pp.load_preprocess_air_pollution('data/air-pollution-info-2025.csv', encoding='utf-8-sig')
air_pollution_df_2024 = pp.load_preprocess_air_pollution('data/air-pollution-info-2024.csv', encoding='cp949')
air_pollution_df_2023 = pp.load_preprocess_air_pollution('data/air-pollution-info-2023.csv', encoding='cp949')
weather_df = pp.load_process_weather('data/weather-info.csv')

air_pollution_df = pd.concat(
    [air_pollution_df_2023, air_pollution_df_2024, air_pollution_df_2025],
    ignore_index=True
)

df = pd.merge(air_pollution_df, weather_df, on='date', how='inner')
df = df[df['station_name'] == '강남구']
df['target_PM2.5'] = (df['PM2.5'].shift(-1) >= 36).astype(int)
df['target_PM10'] = (df['PM10'].shift(-1) >= 81).astype(int)

df_dropped = df.drop(columns=['date', 'station_name', 'PM2.5', 'PM10', 'target_PM2.5', 'target_PM10'])
# PM2.5 예측용
train_input_25, test_input_25, train_target_25, test_target_25 = train_test_split(
    df_dropped,
    df['target_PM2.5'],
    shuffle=False,
    test_size=0.2
)

# PM10 예측용
train_input_10, test_input_10, train_target_10, test_target_10 = train_test_split(
    df_dropped,
    df['target_PM10'],
    shuffle=False,
    test_size=0.2
)

pm25_pipeline = ImbPipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('smoteenn', SMOTEENN(smote=SMOTE(k_neighbors=3))),
    ('model', get_logistic_model())
])
pm25_pipeline.fit(train_input_25, train_target_25)

pm10_pipeline = ImbPipeline([
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler()),
    ('model', get_logistic_model_improved())
])
pm10_pipeline.fit(train_input_10, train_target_10)

print("--- 초미세먼지(PM2.5) 모델 ---")
evaluate_model(pm25_pipeline, train_input_25, train_target_25, test_input_25, test_target_25)
print("--- 미세먼지(PM10) 모델 ---")
evaluate_model(pm10_pipeline, train_input_10, train_target_10, test_input_10, test_target_10)