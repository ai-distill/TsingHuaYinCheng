# 导入KNN分类器
from sklearn.neighbors import KNeighborsClassifier


# 身高，体重，鞋码
x_train = [[185, 80, 43], [170, 70, 41], [163, 45, 36], [165, 55, 39], [156, 41, 35]]
y_train = ["男", "男", "女", "男", "女"]


# 创建机器学习的KNN对象
knn = KNeighborsClassifier(n_neighbors=3)
# 训练数据，自适应模型
knn.fit(x_train, y_train)

# 随机数据用于测试
Test_data = [[185, 76, 45], [156, 43, 35]]

# 展示预测结果
print(knn.predict(Test_data))
print("Hell W")