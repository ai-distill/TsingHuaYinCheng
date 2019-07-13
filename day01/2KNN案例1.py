# 引用数据
import sklearn.datasets as datasets
# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier


# 1. 获取数据当做样本
iris = datasets.load_iris()  # 蓝蝴蝶
print(iris)


x_train = iris.data[::2]
print(x_train)

y_train = iris.target[::2]

x_test = iris.data[1::2]
print(x_test)

y_test = iris.target[1::2]

# 创建KNN分类器
knn = KNeighborsClassifier()
# 训练数据
knn.fit(x_train, y_train)
# 数据预测
y_ = knn.predict(x_test)
print(y_)
print(y_test)
# 评分
print(knn.score(x_test, y_test))



# 绘图
import matplotlib.pyplot as plt
# 绘图引用的模块
from matplotlib.colors import ListedColormap

# 颜色列表
cmap = ListedColormap(["#FF0000", "#00FF00", "#0000FF"])
# 绘制散点图，根据颜色进行分类
plt.scatter(iris.data[:, 2], iris.data[:, 3], c=iris.target, cmap=cmap)
plt.plot(x_test, y_test)
plt.show()