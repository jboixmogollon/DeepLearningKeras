#Comprobar que el entorno funciona correctamente
from matplotlib import pyplot as plt
import numpy as np

N = 50
x = np.random.rand(N)
y = np.random.rand(N)
plt.scatter(x, y)
plt.show()