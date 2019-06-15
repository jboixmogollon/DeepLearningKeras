from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.datasets import mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()
from keras.utils import to_categorical

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

x_train = x_train.reshape(60000, 784)
x_test = x_test.reshape(10000, 784)

print(x_train.shape)
print(x_test.shape)

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