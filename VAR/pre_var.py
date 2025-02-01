import os
import pandas as pd
from statsmodels.tsa.api import VAR
if not os.path.exists('2022-2050'):
    os.makedirs('2022-2050')

files = [
    '10-14_years.csv', '15-19_years.csv', '20-24_years.csv', '25-29_years.csv',
    '30-34_years.csv', '35-39_years.csv', '40-44_years.csv', '45-49_years.csv',
    '5-9_years.csv', '50-54_years.csv', '55-59_years.csv', '60-64_years.csv',
    '65-69_years.csv', '70-74_years.csv', '75-79_years.csv', '80-84_years.csv',
    '85-89_years.csv', '90-94_years.csv', '95plus_years.csv', 'less_5_years.csv'
]

for file in files:
    data = pd.read_csv(f'rawData/{file}')
    train_data = data[(data['year'] >= 1980) & (data['year'] <= 2021)]
    model = VAR(train_data[['year', 'value']])
    results = model.fit(maxlags=2, ic='aic')
    lag_order = results.k_ar
    forecast_input = train_data.values[-lag_order:]
    forecast = results.forecast(y=forecast_input, steps=29) 
    forecast_years = range(2022, 2051)
    result_df = pd.DataFrame({
        'year': forecast_years,
        'predicted': forecast[:, 1]
    })
    result_df.to_csv(f'2022-2050/result_{file}', index=False, float_format='%.8f')