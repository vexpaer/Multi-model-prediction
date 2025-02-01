import os
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

os.makedirs('2011-2021', exist_ok=True)

files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv', '25-29_years.csv',
    '30-34_years.csv', '35-39_years.csv', '40-44_years.csv', '45-49_years.csv',
    '5-9_years.csv', '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv', '80-84_years.csv',
    '85-89_years.csv', '90-94_years.csv', '95plus_years.csv', 'less_5_years.csv'
]

for file in files:
    data = pd.read_csv(f'rawData/{file}')
    train = data.iloc[:31]['value']
    model = ARIMA(train, order=(1,1,1))
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=11)
    result = pd.DataFrame({
        'year': data['year'].iloc[31:], 
        'actual': data['value'].iloc[31:], 
        'predicted': forecast 
    })
    result.to_csv(f'2011-2021/result_{file}', index=False)