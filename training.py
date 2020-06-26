#导入pandas与numpy工具包
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

import string
import warnings
warnings.filterwarnings('ignore')

##导入处理好的特征值数据
X_train = pd.read_csv('data/X_train.csv')

# 从原始数据中提取出Survived列，作为y_train。这里就是为了弄一个y_train出来
temp = pd.read_csv('data/train.csv')
y_train = temp['Survived']

X_test = pd.read_csv('data/X_test.csv')
##建立模型并评估
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import Perceptron


from sklearn.preprocessing import LabelEncoder

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score

import warnings
warnings.filterwarnings('ignore')

###标准化编码
from sklearn.preprocessing import StandardScaler

X_train = StandardScaler().fit_transform(X_train)

X_test = StandardScaler().fit_transform(X_test)

#####定义拟合曲线显示函数
from sklearn.model_selection import learning_curve
#import matplotlib.pyplot as plt

# 定义函数 plot_learning_curve 绘制学习曲线。train_sizes 初始化为 array([ 0.1  ,  0.325,  0.55 ,  0.775,  1\.   ]),cv 初始化为 10，以后调用函数时不再输入这两个变量

def plot_learning_curve(estimator, title, X_train, y_train, cv=10,
                        train_sizes=np.linspace(.1, 1.0, 5)):
    plt.figure()
    plt.title(title) # 设置图的 title
    plt.xlabel('Training examples') # 横坐标
    plt.ylabel('Score') # 纵坐标
    train_sizes, train_scores, test_scores = learning_curve(estimator, X_train, y_train, cv=cv,
                                                            train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1) # 计算平均值
    train_scores_std = np.std(train_scores, axis=1) # 计算标准差
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid() # 设置背景的网格

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std,
                     alpha=0.1, color='g') # 设置颜色
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std,
                     alpha=0.1, color='r')
    plt.plot(train_sizes, train_scores_mean, 'o-', color='g',
             label='traning score') # 绘制训练精度曲线
    plt.plot(train_sizes, test_scores_mean, 'o-', color='r',
             label='testing score') # 绘制测试精度曲线
    plt.legend(loc='best')
    plt.show()
    return plt

leaderboard_model = RandomForestClassifier(criterion='gini',
                                           n_estimators=1750,
                                           max_depth=7,
                                           min_samples_split=6,
                                           min_samples_leaf=6,
                                           max_features='auto',
                                           oob_score=True,
                                           random_state=42,
                                           n_jobs=-1,
                                           verbose=1)
leaderboard_model.fit(X_train,y_train)
g = plot_learning_curve(leaderboard_model, 'RFC', X_train,y_train) #调用定义的 plot_learning_curve 绘制学习曲线
##保存结果
pred1 = leaderboard_model.predict(X_test)
pred1 = pd.DataFrame(pred1)
pred1 = pred1.astype(int)
pred1['Survived'] = pred1

test = pd.read_csv('data/test.csv')

submission = pd.DataFrame({'PassengerId':test.loc[:,'PassengerId'],
                               'Survived':pred1.loc[:,'Survived']})
submission.to_csv('data/lead-2020625.csv',index=False,sep=',')