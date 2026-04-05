import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

#生成3D月牙数据集
def make_moons_3d(n_samples=500, noise=0.1, random_state=42):
    np.random.seed(random_state)
    #C0 类
    t0 = np.linspace(0, np.pi, n_samples)
    x0 = 1.5 * np.cos(t0)
    y0 = np.sin(t0)
    z0 = 0.5 * np.sin(2 * t0)
    c0 = np.c_[x0, y0, z0] + noise * np.random.randn(n_samples, 3)
    
    #C1 类
    t1 = np.linspace(np.pi, 2 * np.pi, n_samples)
    x1 = 1.5 * np.cos(t1) + 1.5
    y1 = np.sin(t1)
    z1 = 0.5 * np.sin(2 * t1) + 0.5
    c1 = np.c_[x1, y1, z1] + noise * np.random.randn(n_samples, 3)
    
    X = np.r_[c0, c1]
    y = np.r_[np.zeros(n_samples), np.ones(n_samples)]
    return X, y

#生成训练集 & 测试集
X_train, y_train = make_moons_3d(n_samples=500, noise=0.1, random_state=42)
X_test, y_test = make_moons_3d(n_samples=250, noise=0.1, random_state=123)

#训练所有分类模型
#决策树
dt = DecisionTreeClassifier(max_depth=5, random_state=42)
dt.fit(X_train, y_train)
y_pred_dt = dt.predict(X_test)

#AdaBoost
ada = AdaBoostClassifier(
    estimator=DecisionTreeClassifier(max_depth=3),
    n_estimators=50, learning_rate=0.1, random_state=42
)
ada.fit(X_train, y_train)
y_pred_ada = ada.predict(X_test)

#SVM 线性核
svm_linear = SVC(kernel='linear', random_state=42)
svm_linear.fit(X_train, y_train)
y_pred_svm_l = svm_linear.predict(X_test)

#SVM RBF核
svm_rbf = SVC(kernel='rbf', gamma='scale', random_state=42)
svm_rbf.fit(X_train, y_train)
y_pred_svm_r = svm_rbf.predict(X_test)

#SVM 多项式核
svm_poly = SVC(kernel='poly', degree=3, gamma='scale', random_state=42)
svm_poly.fit(X_train, y_train)
y_pred_svm_p = svm_poly.predict(X_test)

#模型评估
def evaluate_model(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    print(f'========== {model_name} ==========')
    print(f'Accuracy: {acc:.4f}')
    print('Classification Report:\n', classification_report(y_true, y_pred))
    print('Confusion Matrix:\n', confusion_matrix(y_true, y_pred), '\n')

evaluate_model(y_test, y_pred_dt, 'Decision Tree')
evaluate_model(y_test, y_pred_ada, 'AdaBoost + Decision Tree')
evaluate_model(y_test, y_pred_svm_l, 'SVM (Linear Kernel)')
evaluate_model(y_test, y_pred_svm_r, 'SVM (RBF Kernel)')
evaluate_model(y_test, y_pred_svm_p, 'SVM (Poly Kernel, degree=3)')

#3D 分类结果可视化
def plot_3d_classification_result(X_test, y_true, y_pred, title):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    #正确预测样本
    correct = (y_true == y_pred)
    #错误预测样本
    wrong = (y_true != y_pred)
    
    #绘制正确：蓝色C0，红色C1
    ax.scatter(X_test[correct & (y_true==0), 0], X_test[correct & (y_true==0), 1], X_test[correct & (y_true==0), 2],
               c='blue', label='C0 分类正确', s=15, alpha=0.8)
    ax.scatter(X_test[correct & (y_true==1), 0], X_test[correct & (y_true==1), 1], X_test[correct & (y_true==1), 2],
               c='red', label='C1 分类正确', s=15, alpha=0.8)
    
    #绘制错误：黑色标记
    ax.scatter(X_test[wrong, 0], X_test[wrong, 1], X_test[wrong, 2],
               c='black', label='分类错误', s=60, marker='*', edgecolors='gold')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title, fontsize=14, pad=20)
    ax.legend()
    plt.tight_layout()
    plt.show()

#依次画出 5 个模型的可视化结果
plot_3d_classification_result(X_test, y_test, y_pred_dt, '决策树 3D 分类结果')
plot_3d_classification_result(X_test, y_test, y_pred_ada, 'AdaBoost 3D 分类结果')
plot_3d_classification_result(X_test, y_test, y_pred_svm_l, 'SVM 线性核 3D 分类结果')
plot_3d_classification_result(X_test, y_test, y_pred_svm_r, 'SVM RBF核 3D 分类结果')
plot_3d_classification_result(X_test, y_test, y_pred_svm_p, 'SVM 多项式核 3D 分类结果')