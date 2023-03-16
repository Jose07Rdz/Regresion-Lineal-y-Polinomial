import pandas as pd #pip install numpy
import numpy as np #pip install pandas
from sklearn.linear_model import LinearRegression #pip install scikit-learn
import matplotlib.pyplot as plt #pip install matplotlib
from sklearn.preprocessing import PolynomialFeatures #pip install scikit-learn
import operator
from sklearn.metrics import mean_squared_error #pip install scikit-learn

#!-----------PASO 3-----------
#IMPORTAMOS LOS ARCHIVOS
nombreX = "data1-X.csv"
nombreY = "data1-Y.csv"
dataCrudaX = pd.read_csv(nombreX)
dataCrudaY = pd.read_csv(nombreY)

#PREPARAMOS LOS DATOS PARA SU POSTERIOR USO
x = np.array(dataCrudaX)
y = np.array(dataCrudaY)


#!-----------PASO 4-----------
#CREAMOS EL MODELO DE REGRESION LINEAL
model1 = LinearRegression()

#ENTRENAMOS EL MODELO
model1.fit(x, y)


#!-----------PASO 5-----------
#CREAMOS LA FIGURA DE LA GRAFICA Y LOS EJES
fig, ax = plt.subplots()

#MOSTRMOS LOS PUNTOS (X,Y)
plt.scatter(x,y)

#PINTAMOS EL MODELO DE REGRESION LINEAL 
#?plt.plot(x, model1.predict(x), color="red", label="Regresion lineal") #TODO


#!-----------PASO 6-----------
#SE CREA UNA INSTANCIA DE TIPO "PolynomialFeatures" DE GRADO 4
polynomial_features= PolynomialFeatures(degree=4)

#SE TRANFORMAN LOS DATOS PARA HACER UNA REGRESION POLINOMIAL DE GRADO 4
x_poly = polynomial_features.fit_transform(x)

#SE CREA UN MODELO DE REGRESION LINEAL
model2 = LinearRegression()

#SE ENTRENA EL MODELO CON LOS DATOS TRANSFORMADOS
model2.fit(x_poly, y)

#SE PRECIDEN VALORES UTILIZANDO EL MODELO ENTRENADO
x_pred = np.linspace(4, 16)
x_pred_poly = polynomial_features.transform(x_pred.reshape(-1, 1))
y_pred = model2.predict(x_pred_poly)

#PINTAMOS EL MODELO DE REGRESION POLINOMIAL DE GRADO 4
#?plt.plot(x_pred, y_pred, color="black", label="Regresión polinomial de grado 4") #TODO


#!-----------PASO 7-----------
#PARA EL PASO 7 ES NECESARIO COMENTAR LAS LINEAS 34, 37, 69 Y 114
#FUNCION SALIDA QUE ORDENA EL OBJETO DADA UNA LLAVE
def salida(obj, key):
    sorted_obj = sorted(obj, key=key)
    return sorted_obj

#FUNCION QUE CALCULA EL ERROR DE RAIZ CUADRADA MEDIA (RMSE)
def degreeChoice (x,y,degree):
    polynomial_features= PolynomialFeatures(degree=degree)
    x_poly = polynomial_features.fit_transform(x)
    model = LinearRegression()
    model.fit(x_poly, y)
    y_poly_pred = model.predict(x_poly)
    rmse = np.sqrt(mean_squared_error(y,y_poly_pred))
    salida_zip = salida(zip(x,y_poly_pred), key=operator.itemgetter(0))
    x_p, y_poly_pred_P = zip(*salida_zip)
    return rmse, x_p, y_poly_pred_P

#SE CREAN 3 LISTAS VACIAS (DE LONGITUD 100) PARA LUEGO LLENARLAS CON EL RMSE
rmselist = np.zeros(100)
x_p_list = [None]*100
y_poly_pred_P_list=[None]*100

#ES UN CICLO QUE VA CALCULANDO EL RMSE EN UNA REGRESION DE EL GRADO 1 HASTA 100
for i in np.arange(1, 101):
    rmselist[i-1] ,x_p_list[i-1], y_poly_pred_P_list[i-1] = degreeChoice(x,y,i)

#SE GRAFICA EL RMSE DESDE EL GRADO 1 HASTA 100
#?plt.plot(np.arange(1, 101), rmselist, color='r', label="Error cuadratico medio")


#!-----------PASO 8-----------
# * REGRESION POLINOMIAL DE GRADO 2
#SE CREA UNA INSTANCIA DE TIPO "PolynomialFeatures" DE GRADO 2
polynomial_features= PolynomialFeatures(degree=2)

#SE TRANFORMAN LOS DATOS PARA HACER UNA REGRESION POLINOMIAL DE GRADO 2
x_poly = polynomial_features.fit_transform(x)

#SE CREA UN MODELO DE REGRESION LINEAL
model3 = LinearRegression()

#SE ENTRENA EL MODELO CON LOS DATOS TRANSFORMADOS
model3.fit(x_poly, y)

#SE PRECIDEN VALORES UTILIZANDO EL MODELO ENTRENADO
x_pred = np.linspace(4, 16)
x_pred_poly = polynomial_features.transform(x_pred.reshape(-1, 1))
y_pred = model3.predict(x_pred_poly)

#PINTAMOS EL MODELO DE REGRESION POLINOMIAL DE GRADO 2
#?plt.plot(x_pred, y_pred, color="brown", label="Regresión polinomial de grado 2") #TODO


# * REGRESION POLINOMIAL DE GRADO 16
#SE CREA UNA INSTANCIA DE TIPO "PolynomialFeatures" DE GRADO 16
polynomial_features= PolynomialFeatures(degree=16)

#SE TRANFORMAN LOS DATOS PARA HACER UNA REGRESION POLINOMIAL DE GRADO 16
x_poly = polynomial_features.fit_transform(x)

#SE CREA UN MODELO DE REGRESION LINEAL
model3 = LinearRegression()

#SE ENTRENA EL MODELO CON LOS DATOS TRANSFORMADOS
model3.fit(x_poly, y)

#SE PRECIDEN VALORES UTILIZANDO EL MODELO ENTRENADO
x_pred = np.linspace(4, 16)
x_pred_poly = polynomial_features.transform(x_pred.reshape(-1, 1))
y_pred = model3.predict(x_pred_poly)

#PINTAMOS EL MODELO DE REGRESION POLINOMIAL DE GRADO 16
#?plt.plot(x_pred, y_pred, color="yellow", label="Regresión polinomial de grado 16") #TODO


# * REGRESION POLINOMIAL DE GRADO 32
#SE CREA UNA INSTANCIA DE TIPO "PolynomialFeatures" DE GRADO 32
polynomial_features= PolynomialFeatures(degree=32)

#SE TRANFORMAN LOS DATOS PARA HACER UNA REGRESION POLINOMIAL DE GRADO 32
x_poly = polynomial_features.fit_transform(x)

#SE CREA UN MODELO DE REGRESION LINEAL
model3 = LinearRegression()

#SE ENTRENA EL MODELO CON LOS DATOS TRANSFORMADOS
model3.fit(x_poly, y)

#SE PRECIDEN VALORES UTILIZANDO EL MODELO ENTRENADO
x_pred = np.linspace(4, 16)
x_pred_poly = polynomial_features.transform(x_pred.reshape(-1, 1))
y_pred = model3.predict(x_pred_poly)

#PINTAMOS EL MODELO DE REGRESION POLINOMIAL DE GRADO 32
#?plt.plot(x_pred, y_pred, color="gray", label="Regresión polinomial de grado 32") #TODO


# * REGRESION POLINOMIAL DE GRADO 64
#SE CREA UNA INSTANCIA DE TIPO "PolynomialFeatures" DE GRADO 64
polynomial_features= PolynomialFeatures(degree=64)

#SE TRANFORMAN LOS DATOS PARA HACER UNA REGRESION POLINOMIAL DE GRADO 64
x_poly = polynomial_features.fit_transform(x)

#SE CREA UN MODELO DE REGRESION LINEAL
model3 = LinearRegression()

#SE ENTRENA EL MODELO CON LOS DATOS TRANSFORMADOS
model3.fit(x_poly, y)

#SE PRECIDEN VALORES UTILIZANDO EL MODELO ENTRENADO
x_pred = np.linspace(4, 16)
x_pred_poly = polynomial_features.transform(x_pred.reshape(-1, 1))
y_pred = model3.predict(x_pred_poly)

#PINTAMOS EL MODELO DE REGRESION POLINOMIAL DE GRADO 64
plt.plot(x_pred, y_pred, color="green", label="Regresión polinomial de grado 64") #TODO


#!-----------ESTO ES PARA MOSTRAR LA GRAFICA-----------
#* ESTA PARTE NO LA TOQUEN
#MOSTRAMOS EL LABEL DE CADA GRAFICA
ax.legend()

#AGREGAMOS TITULO Y NOMBRE A LOS EJES
plt.title("PRACTICA 1")
plt.xlabel("X")
plt.ylabel("Y")

#MOSTRAMOS LA GRAFICA
plt.show()