#导入pandas与numpy工具包
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式

import string
import warnings
warnings.filterwarnings('ignore')

train = pd.read_csv('data/train_data.csv')
test = pd.read_csv('data/test_data.csv')
#####训练集处理
print(train.columns)
# 将训练数据分成标记和特征两部分
# 提取出训练集数据标记
y_train = train['Survived']
# 删除明确不需要的列
X_train = train.drop(['PassengerId', 'Survived','Name','Age','Ticket','Fare', 'Cabin','Family','Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA'],axis=1)
X_train.info()
#####测试集处理
# 把PassengerId提取出来，后面用
Id = test['PassengerId']
# 删除明确不需要的列
X_test = test.drop(['PassengerId', 'Survived','Name','Age','Ticket','Fare', 'Cabin','Family','Ticket_Survival_Rate', 'Family_Survival_Rate', 'Ticket_Survival_Rate_NA', 'Family_Survival_Rate_NA'],axis=1)
X_test.info()

#####特征工程
###训练集特征二值化编码
##训练集特定的数值列做二值化编码（分列）
# 对数值型的列做二值化分列处理，非数值的get_dummies会自动分列处理
# 对数值列Pclass做二值化分列处理
'''
这里对7列数值型特征值做了二值化处理，使用是pandas的get_dummies，这是一种one-hot编码方式
注意这里的Parch，训练集和测试集的数值不完全相同，所以给训练集中补了一个数据。否则做完编码，训练集会少一列'''
X_train = X_train.join(pd.get_dummies(X_train.Pclass, prefix= 'Pclass'))
# 对数值列SibSp做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.SibSp, prefix= 'SibSp'))
# 对数值列Parch做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Parch, prefix= 'Parch'))
# 因为测试集里面Parch多了一个数，训练集里面没有，如果不做补充，训练集和测试集维度会不一样
X_train['Parch_9'] = 0
# 对数值列FamilySize做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.FamilySize, prefix= 'FamilySize'))
# 对数值列Ticket_Frequency做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Ticket_Frequency, prefix= 'Ticket_Frequency'))
# 对数值列Survival_Rate做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Survival_Rate, prefix= 'Survival_Rate'))
# 对数值列Survival_Rate_NA做二值化分列处理
X_train = X_train.join(pd.get_dummies(X_train.Survival_Rate_NA, prefix= 'Survival_Rate_NA'))
# 删除7个数值列  ,多余列
X_train = X_train.drop(['Pclass','SibSp','Parch','FamilySize','Ticket_Frequency','Survival_Rate','Survival_Rate_NA'],axis=1)
###特征统一做One-Hot编码
X_train = pd.get_dummies(X_train)
encoded = list(X_train.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))
X_train.info()

###测试集特征二值化编码
##测试集特定的数值列做二值化编码（分列）
# 对数值列Pclass做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Pclass, prefix= 'Pclass'))

# 对数值列SibSp做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.SibSp, prefix= 'SibSp'))

# 对数值列Parch做二值化分列处理
X_test= X_test.join(pd.get_dummies(X_test.Parch, prefix= 'Parch'))

# 对数值列FamilySize做二值化分列处理
X_test =X_test.join(pd.get_dummies(X_test.FamilySize, prefix= 'FamilySize'))

# 对数值列Ticket_Frequency做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Ticket_Frequency, prefix= 'Ticket_Frequency'))

# 对数值列Survival_Rate做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Survival_Rate, prefix= 'Survival_Rate'))

# 对数值列Survival_Rate_NA做二值化分列处理
X_test = X_test.join(pd.get_dummies(X_test.Survival_Rate_NA, prefix= 'Survival_Rate_NA'))
#再次删除测试集中不需要的列
X_test = X_test.drop(['Pclass','SibSp','Parch','FamilySize','Ticket_Frequency','Survival_Rate','Survival_Rate_NA'],axis=1)
#特征统一做One-Hot编码
X_test = pd.get_dummies(X_test)
encoded = list(X_test.columns)
print ("{} total features after one-hot encoding.".format(len(encoded)))

X_test.info()
#######特征筛选
'''
特征筛选在这里的目的是通过几个机器学习模型，筛选出对结果影响最大的特征
然后将最重要的特征合并起来为后面机器学习和预测使用
'''
from time import time
from sklearn import ensemble
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.ensemble import ExtraTreesClassifier

from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron

from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

from xgboost.sklearn import XGBClassifier

from sklearn.preprocessing import LabelEncoder

from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GridSearchCV, cross_val_score, StratifiedKFold, learning_curve
from sklearn.model_selection import cross_val_score
from sklearn.metrics import precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score,roc_auc_score

import warnings
warnings.filterwarnings('ignore')

#定义特征筛选函数
def get_top_n_features(X_train, y_train, top_n_features):
    # 随机森林
    rf_est = RandomForestClassifier(random_state=42)
    rf_param_grid = {'n_estimators': [500], 'min_samples_split': [2, 3], 'max_depth': [20]}
    rf_grid = model_selection.GridSearchCV(rf_est, rf_param_grid, n_jobs=25, cv=10, verbose=1)   #这里使用了网格搜索
    rf_grid.fit(X_train,y_train)
    #将feature按Importance排序
    feature_imp_sorted_rf = pd.DataFrame({'feature': list(X_train), 'importance': rf_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_rf = feature_imp_sorted_rf.head(top_n_features)['feature']
    print('Sample 25 Features from RF Classifier')

    print(str(features_top_n_rf[:25]))


    # AdaBoost
    ada_est = ensemble.AdaBoostClassifier(random_state=42)
    ada_param_grid = {'n_estimators': [500], 'learning_rate': [0.5, 0.6]}
    ada_grid = model_selection.GridSearchCV(ada_est, ada_param_grid, n_jobs=25, cv=10, verbose=1)
    ada_grid.fit(X_train, y_train)
    #排序
    feature_imp_sorted_ada = pd.DataFrame({'feature': list(X_train),'importance': ada_grid.best_estimator_.feature_importances_}).sort_values( 'importance', ascending=False)
    features_top_n_ada = feature_imp_sorted_ada.head(top_n_features)['feature']
    print('Sample 25 Features from Ada Classifier')
    print(str(features_top_n_ada[:25]))

    # ExtraTree
    et_est = ensemble.ExtraTreesClassifier(random_state=42)
    et_param_grid = {'n_estimators': [500], 'min_samples_split': [3, 4], 'max_depth': [15]}
    et_grid = model_selection.GridSearchCV(et_est, et_param_grid, n_jobs=25, cv=10, verbose=1)
    et_grid.fit(X_train, y_train)
    #排序
    feature_imp_sorted_et = pd.DataFrame({'feature': list(X_train), 'importance': et_grid.best_estimator_.feature_importances_}).sort_values('importance', ascending=False)
    features_top_n_et = feature_imp_sorted_et.head(top_n_features)['feature']
    print('Sample 25 Features from ET Classifier:')
    print(str(features_top_n_et[:25]))

    # 将三个模型挑选出来的前features_top_n_et合并,并去除重复项
    features_top_n = pd.concat([features_top_n_rf, features_top_n_ada, features_top_n_et], ignore_index=True).drop_duplicates()
    return features_top_n


feature_to_pick = 46
feature_top_n = get_top_n_features(X_train,y_train,feature_to_pick)
X_train = X_train[feature_top_n]
X_test = X_test[feature_top_n]

X_train.to_csv('data/X_train.csv',index=False,sep=',')

X_test.to_csv('data/X_test.csv',index=False,sep=',')

y_train.to_csv('data/y_train.csv',index=False,sep=',')