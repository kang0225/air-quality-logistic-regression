import pandas as pd
import numpy as np

def load_preprocess_air_pollution(filepath, encoding):
    data_frame = pd.read_csv(filepath, encoding=encoding)
    
    data_frame.rename(columns={
        '측정일시': 'date',
        '측정소명': 'station_name',
        '미세먼지농도(㎍/㎥)': 'PM10',
        '미세먼지농도 (㎍/㎥)': 'PM10',
        '초미세먼지농도(㎍/㎥)': 'PM2.5',
        '초미세먼지농도 (㎍/㎥)': 'PM2.5',
        '오존농도(ppm)': 'O3',
        '이산화질소농도(ppm)': 'NO2',
        '일산화탄소농도(ppm)': 'CO',
        '아황산가스농도(ppm)': 'SO2',
        '아황산가스농도 (ppm)': 'SO2'
    }, inplace=True)

    data_frame['date'] = pd.to_datetime(
        data_frame['date'].astype(str),
        format='%Y%m%d',
        errors='coerce')

    features = ['PM10', 'PM2.5', 'O3', 'NO2', 'CO', 'SO2']
    for col in features:
        data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')

    data_frame.sort_values('date', inplace=True)
    data_frame[features] = data_frame[features].ffill().bfill()

    train_set = data_frame[['date'] +['station_name'] + features]
    return train_set

def load_process_weather(filepath):
    data_frame = pd.read_csv(filepath)

    data_frame.rename(columns={
        '일시': 'date',
        '지점': 'station_id',
        '지점명': 'station_name',
        '평균기온(°C)': 'temp_c_mean',
        '최저기온(°C)': 'temp_c_min',
        '최고기온(°C)': 'temp_c_max',
        '일강수량(mm)': 'precip_mm',
        '평균 풍속(m/s)': 'wind_ms',
        '평균 상대습도(%)': 'rh',
        '평균 해면기압(hPa)': 'mslp'
    }, inplace=True)

    data_frame['date'] = pd.to_datetime(data_frame['date'], errors='coerce')
    data_frame.drop(columns=['station_id', 'station_name'], inplace=True,
                    errors='ignore')
    features = ['temp_c_mean', 'temp_c_min', 'temp_c_max', 'precip_mm', 'wind_ms', 'rh', 'mslp']
    
    for col in features:
        data_frame[col] = pd.to_numeric(data_frame[col], errors='coerce')

    data_frame.sort_values('date', inplace=True)
    data_frame[features] = data_frame[features].ffill().bfill()

    train_set = data_frame[['date'] + features]

    return train_set