import pandas as pd
from prophet import Prophet
import os

if not os.path.exists('2011-2021'):
    os.makedirs('2011-2021')

files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv',
    '25-29_years.csv', '30-34_years.csv', '35-39_years.csv',
    '40-44_years.csv', '45-49_years.csv', '5-9_years.csv',
    '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv',
    '80-84_years.csv', '85-89_years.csv', '90-94_years.csv',
    '95plus_years.csv', 'less_5_years.csv'
]

for file in files:
    df = pd.read_csv(f'rawData/{file}')
    df = df.rename(columns={'year': 'ds', 'value': 'y'})
    train = df[df['ds'] <= 2010]
    future = pd.DataFrame({'ds': range(2011, 2022)})
    model = Prophet()
    model.fit(train)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].rename(columns={'ds': 'year', 'yhat': 'value'})
    result.to_csv(f'2011-2021/result_{file}', index=False, float_format='%.8f')