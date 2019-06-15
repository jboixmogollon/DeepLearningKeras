#to_categorical
#Función para transformar las etiquetas en un vector, indicando con 1 el numero al que pertenece

from keras.utils import to_categorical

from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(y_test[0])
print(y_train[0])
print(y_train.shape)
print(x_test.shape)

#Aplicamos transformación

y_train = to_categorical(y_train, num_classes=10)
y_test = to_categorical(y_test, num_classes=10)

print(y_test[0])
print(y_train[0])

print(y_train.shape)
print(y_test.shape)