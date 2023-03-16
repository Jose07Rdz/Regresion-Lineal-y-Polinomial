# Regresión Lineal y Polinomial 

El siguiente código es fruto de una serie de prácticas, vistas en la materia de Análisis y Visualización de Datos, esta es la primera practica la cual consta de realizar Regresión lineal y Regresión polinomial.


# Antes de comenzar  

Es necesario tener las siguientes librerias instaladas `numpy, pandas, matplotlib y scikit-learn`, sino dispones de ellas se pueden instalar con el siguiente comando:

```
pip install numpy pandas matplotlib scikit-learn 
```


# Obtención de los datos 

Para la obtención de los datos, el profesor uso un servidor de Redis al cual nosotros nos conectamos mediante el código del archivo ***P1pubsub.py*** y dicho servidor nos responde con la información en un archivo tipo **JSON**, después el código extrae la información valida y esa es almacenada en unos archivos de tipo **CSV**, que son los archivos ***Data1-X.csv*** y ***Data1-Y.csv***.


# Grafica de los puntos 

Primero se leen los datos de los archivos y se transforman a tipo `<class 'numpy.ndarray'>` para su posterior uso:
```python
#IMPORTAMOS LOS ARCHIVOS 
nombreX = "data1-X.csv" 
nombreY = "data1-Y.csv" 
dataCrudaX = pd.read_csv(nombreX) 
dataCrudaY = pd.read_csv(nombreY) 

#PREPARAMOS LOS DATOS PARA SU POSTERIOR USO 
x = np.array(dataCrudaX) 
y = np.array(dataCrudaY) 
``` 


# Regresión lineal

Para la regresión lineal, se crea un modelo utilizando la clase `LinearRegression` del módulo `sklearn.linear_model` y dicho modelo es entrenado con los datos que previamente cargamos de los archivos.
```python
#CREAMOS EL MODELO DE REGRESION LINEAL
model1 = LinearRegression()

#ENTRENAMOS EL MODELO
model1.fit(x, y)
```

Después de entrenar el modelo, podemos crear la gráfica de dicho modelo:
```python
plt.plot(x, model1.predict(x), color="red", label="Regresion lineal")
```

Y la gráfica generada es la siguiente: 

![](https://i.imgur.com/tfk6ZBg.png)


# Regresión polinomial de grado 4

Para el modelo de regresión polinomial utilizamos la misma clase que en la regresión lineal, simplemente que en este caso los datos serán transformados, utilizamos la clase `PolynomialFeatures` del módulo `sklearn.preprocessing`. 
```python
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
```

Y finalmente generamos la nueva gráfica con los datos transformados: 
```python
plt.plot(x_pred, y_pred, color="black", label="Regresión polinomial de grado 4")
```

La gráfica generada es la siguiente:

![](https://i.imgur.com/E2ChXHV.png)


# ¿Como saber el grado adecuado para nuestra regresión polinomial? 

Para conocer el grado con menor error cuadrático medio utilizamos la función `mean_squared_error` del módulo `sklearn.metrics`.
Creamos una función que itera entre un rango de 0 a 100 y obtiene el error cuadrático medio de cada grado para nuestros datos.
```python
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
```

Finalmente graficamos los resultados obtenidos: 

![](https://i.imgur.com/SxyGoz1.png)

Como se puede notar en la gráfica, el punto donde el error cuadrático medio es menor es aproximadamente entre el rango 2 a 20 para nuestros datos puede que este cambie si se utilizan datos diferentes. ¿Podrás encontrar el grado con menor error cuadrático medio? 
