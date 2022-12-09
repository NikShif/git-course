#мпорт необходимых библиотек
import numpy as np
from sklearn.linear_model import LinearRegression

#зададим начальные условия
f0 = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
f1 = np.array([1.1, 2.1, 3.1, 4.4, 5.2, 6.4, 7.1, 8.2, 9.4, 10.5])
f2 = np.array([1.4, 2.3, 3.4, 4.1, 5.5, 6.2, 7.3, 8.4, 9.2, 10.1])
f3 = np.array([1.2, 2.2, 3.4, 4.2, 5.3, 6.2, 7.3, 8.4, 9.2, 10.3])
y = np.array([[1.2], [2.2], [3.3], [4.3], [5.2], [6.3], [7.2], [8.3], [9.3], [10.2]])
w = np.array([np.nan, np.nan, np.nan, np.nan])
X = np.array([f0, f1, f2, f3]).T

#рассчитаем коэффициенты используя выведенную формулу
coef_matrix = np.dot(np.dot(np.linalg.inv(np.dot(X.T, X)), X.T), y)
print(f'Коэффициенты рассчитанные по формуле {coef_matrix.T[0]}')
#Коэффициенты рассчитанные по формуле [0.05994939 0.42839296 0.09249473 0.46642055]

#проверим расчет используя библиотеку sklearn
model = LinearRegression().fit(X, y)
coef_sklearn = model.coef_.T
coef_sklearn[0] = model.intercept_
print(f'Коэффициенты рассчитанные с использованием библиотеки sklearn {coef_sklearn.T[0]}')
#Коэффициенты полученные с рассчитанные библиотеки sklearn [0.05994939 0.42839296 0.09249473 0.4664205
