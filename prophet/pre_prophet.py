import os
import pandas as pd
from prophet import Prophet

if not os.path.exists('2022-2050'):
    os.makedirs('2022-2050')
files = [f for f in os.listdir('rawData') if f.endswith('.csv')]

for file in files:
    df = pd.read_csv(f'rawData/{file}')
    df = df.rename(columns={'year': 'ds', 'value': 'y'})
    train = df[(df['ds'] >= 1980) & (df['ds'] <= 2021)]
    future = pd.DataFrame({'ds': range(2022, 2051)})
    model = Prophet()
    model.fit(train)
    forecast = model.predict(future)
    result = forecast[['ds', 'yhat']].rename(columns={'ds': 'year', 'yhat': 'value'})[['year', 'value']]
    result.to_csv(f'2022-2050/result_{file}', index=False, float_format='%.8f')

print("done process, result in'2022-2050' folder")