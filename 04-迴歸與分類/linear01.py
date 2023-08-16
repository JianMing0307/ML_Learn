import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
temperatures = np.array([29, 28, 34, 31,25, 29, 32, 31,24, 33, 25, 31,26, 30])
drink_sales = np.array([7.7, 6.2, 9.3, 8.4,5.9, 6.4, 8.0, 7.5,5.8, 9.1, 5.1, 7.3,6.5, 8.4])
X = pd.DataFrame(temperatures, columns=["Temperature"])
target = pd.DataFrame(drink_sales, columns=["Drink_Sales"])
y = target["Drink_Sales"]
lm = LinearRegression( )
lm.fit(X, y)
print("�j�k�Y��:", lm.coef_)
print("�I�Z:", lm.intercept_ )
