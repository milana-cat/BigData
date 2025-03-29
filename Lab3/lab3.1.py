import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn import neighbors, datasets
n_neighbors = 15
iris = datasets.load_iris()
x = iris.data[:, :2]
y = iris.target
h = 0.02 # шаг
# создадим список цветов
сmap_light = ListedColormap(['#FFAAAA', '#AAFFAA',
'#AAAAFF'])
сmаp_bold = ListedColormap(['#FF0000', '#00FF00',
'#0000FF'])
# построим модель
clf = neighbors.KNeighborsClassifier(n_neighbors)
clf.fit(x, y)
# изобразим границы классов и реальные их элементы
x_min,x_max = x[:, 0].min() - 1, x[:, 0].max() + 1
y_min,y_max = x[:, 1].min() - 1, x[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
z = z.reshape(xx.shape)
plt.figure()
plt.pcolormesh(xx, yy, z, cmap=сmap_light)
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=сmаp_bold, edgecolor='k',
s=20)
plt.xlim(xx.min(), xx.max())
plt.ylim(yy.min(), yy.max())
plt.show()