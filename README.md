
# Desafío 3

## Instrucciones para correr
Descargar los archivos MLP_Keras.ipynb y MLP_Matrices.ipynb. Se sugiere subirlos a Google Collaborative para una mejor visualización.
## Link al video
https://drive.google.com/file/d/17urDmW4MvUkNk63E44r_hbgD3T6twmsK/view?usp=sharing
## Descripción del problema
Para este desafío, se decidió abordar el problema de reconocer carácteres (letras) de una imagen con el propósito de digitalizar documentos escaneados. Un ejemplo de aplicación de este problema puede ser la digitalización de antiguas recetas médicas escritas a mano y asociarlas al perfil virtual del paciente, así evitar problemas al extraviar el documento físico.

La estrategia de solución es implementar dos perceptrones multicapa, uno utilizando la librería Keras y otro utilizando matrices. Se realizará al final una comparación entre estas dos redes para seleccionar la que obtuvo mejores resultados.

En esta ocasión, se tomará el dataset propuesto por https://github.com/OptativoPUCV/Handwritten-letter-dataset para entrenar la red y validar los resultados de las predicciones para generar métricas como la precisión del modelo.

## MLP usando Keras
A continuación se presentan snippets del código del perceptrón multicapa usando la librería Keras.

### Lectura del dataset

```
# Leer y guardar el dataset
!git clone https://github.com/OptativoPUCV/Handwritten-letter-dataset
!mkdir Datos-letras
!7z x "/content/Handwritten-letter-dataset/A_Z Handwritten Data.7z.001" -tsplit
!7z e "/content/A_Z Handwritten Data.7z"
data = pd.read_csv('A_Z Handwritten Data.csv').astype('float32')
```
```
X = data.drop('0',axis = 1)  # x serán los datos de ENTRADA
y = data['0']    # y representa el label o letra de SALIDA
```

### División del dataset en 70% 'train' y 30% 'test'
```
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size = 0.3)
```

### Normalización

Para la normalización del input, se divide cada valor por 255 para marcar las celdas completamente negras con 0s y las completamente blancas con 1s.
```
x_training = x_train.astype('float32')
x_training /= 255

x_testing = x_test.astype('float32')
x_testing /= 255
```
En la normalización del output encontramos las 26 salidas, una por cada letra del abecedario en inglés. La salida consiste en 0s y 1s siguiendo la lógica de las entradas.
```
num_classes = 26

y_training = to_categorical(y_train, num_classes)
y_testing = to_categorical(y_test, num_classes)
```
### Creación del modelo
Para el modelo se utiliza la función de activación relu debido a su mejor convergencia comparado con la función sigmoidal, además de ser menos costosa computacionalmente al no tener que calcular funciones como exponenciales. Finalmente, se usa Softmax debido a que se trata de un problema de reconocimiento de caracteres y los valores de salida son la probabilidad de que una muestra sea precisamente uno de los 26 caracteres.

```
from keras.engine import input_spec
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout

modelo = Sequential()
modelo.add(Dense(208, input_shape=(784,), activation="relu"))
modelo.add(Dense(104, activation="relu"))
modelo.add(Dense(52, activation="relu"))
modelo.add(Dense(26, activation="softmax"))


modelo.summary()
```
Para este modelo se usa el optimizador Adam (Adaptive Moment Estimation) debido a que el entorno de trabajo fue Google Collaborative y la cantidad de memoria disponible es limitada.

```
modelo.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
```
### Entrenamiento
Para este problema se entrenó la red con 100 iteraciones, tardando aproximadamente 80 minutos en terminar.
```
history = modelo.fit(x_training, y_training, validation_data=(x_testing, y_testing), batch_size=20, epochs=100, verbose=2)
```
### Resultados
Este MLP alcanzó una precisión del 97% al ser evaluado con los datos de testing, siendo bastante certero en reconocer los caracteres. En el código de este modelo se encuentran también las matrices de confusión por cada caracter. 
![alt text](https://cdn.discordapp.com/attachments/454085077568192515/975515055405817907/unknown.png)

## MLP usando matrices
Para este tipo de MLP se siguieron los mismos pasos para leer el dataset, dividirlo y normalizar los datos.

### Creación de las capas y de la red
Aquí se define la creación de la red. Cabe destacar que se utilizó la misma topología que la usada en Keras, siendo llamada en la sección del *Entrenamiento*.
```
class Capa():
  def __init__(self, n_conexiones: int, n_neuronas: int, activation):
    self.activation = activation
    self.W = np.random.rand(n_conexiones, n_neuronas) * 2 - 1
```
```
def crear_red(topologia: list, activation):
  red = []
  for l, capa in enumerate(topologia[:-1]):
    red.append( Capa(topologia[l], topologia[l+1], activation) )
  return red
```
### Forward y backpropagation
```
def forward(red, X):
  out = [(None, X)]
  for l, capa in enumerate(red):
    z = out[-1][1] @ red[l].W # Multiplicación de matrices
    a = red[l].activation(z)[0] # La función de activación retorna el valor activado y el derivado, necesitamos el activado para el forward
    out.append((z, a)) # Guardamos todas las combinaciones para poder usar la misma función en el backpropagation
  return out
```
```
def train(red, X, Y, coste, learning_rate=0.001):
  # forward 
  out = forward(red, X)

  # backward pass
  delta = []
  #for i in range(len(red)-1, -1, -1): # recorrer hacie atrás del largo a 0
  for i in reversed(range(0,len(red))):
    z = out[i+1][0]
    a = out[i+1][1]
    if i == len(red)-1:
        #delta última capa
        delta.insert(0, coste(a, Y)[1] * red[i].activation(a)[1] ) # delta 0 = derivada del coste (osea Ypred - Yesp) * derivada de activación de la capa
    else:
        # delta respecto al anterior
        delta.insert(0, delta[0] @ aux_W.T * red[i].activation(a)[1]) # delta n = delta(n+1) x W(n+1).T * derivada de activación de la capa 
    aux_W = red[i].W
    # Descenso del gradiente
    red[i].W = red[i].W - out[i][1].T @ delta[0] * learning_rate # nuevoW[i] = actualW[i] - salida[i].T x delta * learning_rate

  return out[-1][1]
```


### Función de activación
Para este modelo se usó la función sigmoide debido a que en el taller anterior mostró buenos resultados comparados con tanh y así poder comparar esta función con ReLu.
```
def activation(x):
  return ((1/(1+np.e**(-x))) , (x * (1-x)))
```
### Entrenamiento
Para el entrenamiento también se iteró 100 veces el algoritmo, tardando un total de 17 minutos aproximadamente, mucho menos que el modelo con Keras.
```
import time 
from IPython.display import clear_output
topologia = [784, 208, 104, 52, 26]
red = crear_red(topologia, activation)
loss = []

for i in range(100):
  pY = train(red, X, Y, coste, learning_rate=0.001)
  if i % 25 == 0:
    costo = coste(pY, Y)[0]
    print(f'Coste iteración {i}: {costo}')
    loss.append(costo)
    
plt.plot(range(len(loss)), loss)
plt.show()
```
![alt text](https://cdn.discordapp.com/attachments/454085077568192515/975526482074034196/unknown.png)
Cada color representa un caracter de los 26, ilustrando la disminución del costo de cada uno después de entrenar el modelo 100 veces.
### Resultados finales
Si bien el MLP usando matrices requiere de un menor tiempo para entrenarse (17 min vs 80 min), los resultados obtenidos de las predicciones son bastante inferiores. El caracter "D" fue el que obtuvo una mayor precisión cercana al 30% mientras que el resto de las letras fue entre 20% a 5%, muy por debajo del MLP con Keras.
En el código aparecen las matrices de confusión de cada uno. 

La principal desventaja de implementar un MLP con matrices es la dificultad de programar el modelo para que se adapte al problema, lo que consumió gran parte del tiempo para completar este desafío. Esto comparado con Keras que abstrae estas operaciones generales y que permite concentrarse en los hiperparámetros, siendo definitivamente la mejor opción de las dos.
