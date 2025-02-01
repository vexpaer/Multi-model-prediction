import os
import pandas as pd
from sklearn.svm import SVR
import numpy as np

if not os.path.exists('2022-2050'):
    os.makedirs('2022-2050')

files = [f for f in os.listdir('rawData') if f.endswith('.csv')]

for file in files:
    data = pd.read_csv(f'rawData/{file}')
    X_train = data['year'].values.reshape(-1, 1)
    y_train = data['value'].values
    svr = SVR(kernel='rbf', C=100, gamma=0.1, epsilon=.1)
    svr.fit(X_train, y_train)
    future_years = np.arange(2022, 2051).reshape(-1, 1)
    y_pred = svr.predict(future_years)
    result = pd.DataFrame({
        'year': future_years.flatten(),
        'predicted_value': y_pred
    })
    result.to_csv(f'2022-2050/result_{file}', index=False)

print("done process, result in '2022-2050'folder ")