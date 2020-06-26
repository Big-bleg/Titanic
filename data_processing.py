import pandas as pd
import numpy as np
#画图工具
import matplotlib.pyplot as plt
import seaborn as sns

#下面两句是解决绘图显示中文问题
plt.rcParams['font.sans-serif'] = ['SimHei'] # 替换sans-serif字体
plt.rcParams['axes.unicode_minus'] = False   # 解决坐标轴负数的负号显示问题

import string#该库中存放了常用字母和符号
import warnings#忽略警告
warnings.filterwarnings('ignore')

####数据预处理
#读取泰坦尼克号乘客数据集
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')
#数据混合，数据混合的目的主要是在于可以一次性的处理训练集和测试集数据，而且处理的方式方法可以保持一致
#对测试集中添加Survived列，并将该列填充为0
test['Survived'] = 0
#将测试集放到训练集之后，形成混合数据集
combined = train.append(test, sort=False)
#print(train.shape, test.shape, combined.shape)
#print(combined.head())#查看数据信息
#print(combined.info())#打印header的信息

#数据重命名，方便后面显示
train.name ='Training Set'
test.name = 'Test Set'
combined.name = 'ALL Set'
dfs = [train, test]

####缺失值查看和分析
'''
缺失值填充问题有点复杂，它会涉及之后分析和处理次序问题
个人理解，如果先填充缺失值，后面的分析可能会受到缺失值填充时引入噪声影响，所以这里没有先填充缺失值，而是根据情况确定哪些先处理和后处理
注意1：数据的填充有时要考虑数据相关性，决定选择的具体方法
注意2：特别是如果选定了使用一些预测方法来填充缺失值的情况下，有些参数就不能先填充
Embarked缺失很少，所以可以先行考虑处理
'''
#定义缺失值显示函数
def display_missing(df):
    for col in df.columns.tolist():#df.columns.tolist输出列名，并转化为list
        print('{} column missing values: {}'.format(col, df[col].isnull().sum()))
    print('\n')

for df in dfs:
    print('{}'.format(df.name))
    display_missing(df)
'''
训练集
1）Age缺失177个
2）Cabin缺失687个，缺失太多，不太适合先填充，除非后面
3）Embarked缺失2个
测试集
1）Age缺失86个
2）Cabin缺失327个
3）Fare缺失1个
'''
###处理Fare团体票，为什么要先处理Fare团体票，因为后面的一些分析还是要在将票价因素包含在内的，而考虑的前提应该是尽可能接近真实票价花费
#建立一个临时列，存放团体怕票的计数值
combined['Group_Ticket'] = combined['Fare'].groupby(by=combined['Ticket']).transform('count')
#print(combined['Group_Ticket'])
# 票价对应除以团体票计数值，得到每张票的真实价格。如果非团体票，那么就是除以1，价格不变
combined['Fare'] = combined['Fare'] / combined['Group_Ticket']
#print(combined['Fare'])
# 删除临时列
combined.drop(['Group_Ticket'], axis=1, inplace=True)

####数据相关性分析
###生存乘客分布
survived = train['Survived'].value_counts()[1]#.value_counts可以用于统计dataframe中不同数字或字符串出现的次数
not_survived = train['Survived'].value_counts()[0]
survived_per = survived / train.shape[0] * 100
not_survived_per = not_survived / train.shape[0] * 100
print('{} of {} passengers survived and it is the {:.2f}% of the training set.'.format(survived, train.shape[0], survived_per))
print('{} of {} passengers didnt survive and it is the {:.2f}% of the training set.'.format(not_survived, train.shape[0], not_survived_per))
plt.figure(figsize=(6,4))
sns.countplot(train['Survived'])#统计每个类别的数量

plt.xlabel('Survival', size=15, labelpad=15)
plt.ylabel('Passager Count', size=15, labelpad=15)
plt.xticks((0, 1), ['Not Survived({0:.2f}%)'.format(not_survived_per), 'Survived({0:.2f}%)'.format(survived_per)])
plt.tick_params(axis='x', labelsize=13)
plt.tick_params(axis='y', labelsize=13)

plt.title('Training Set Survival Distribution', size=15, y=1.05)
plt.show()
###相关性显示
train_corr = train.drop(['PassengerId'], axis=1).corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
train_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
#print(train_corr)
train_corr.drop(train_corr.iloc[1::2].index, inplace=True)
#print(train_corr)
train_corr_nd = train_corr.drop(train_corr[train_corr['Correlation Coefficient'] == 1.0].index)
#print(train_corr_nd)
# 训练集数据特征间的高相关性
corr = train_corr_nd['Correlation Coefficient'] > 0.1
print('训练集相关性：\n',train_corr_nd[corr])

test_corr = test.corr().abs().unstack().sort_values(kind="quicksort", ascending=False).reset_index()
test_corr.rename(columns={"level_0": "Feature 1", "level_1": "Feature 2", 0: 'Correlation Coefficient'}, inplace=True)
test_corr.drop(test_corr.iloc[1::2].index, inplace=True)
test_corr_nd = test_corr.drop(test_corr[test_corr['Correlation Coefficient'] == 1.0].index)

# 测试集数据特征间的高相关性
corr = test_corr_nd['Correlation Coefficient'] > 0.1
print('测试集相关性：\n',test_corr_nd[corr])
###相关性显示图示
fig, axs = plt.subplots(nrows=2, figsize=(16, 16))

sns.heatmap(train.drop(['PassengerId'], axis=1).corr(), ax=axs[0], annot=True, square=True, cmap='coolwarm',
            annot_kws={'size': 14})
sns.heatmap(test.drop(['PassengerId'], axis=1).corr(), ax=axs[1], annot=True, square=True, cmap='coolwarm',
            annot_kws={'size': 14})

for i in range(2):
    axs[i].tick_params(axis='x', labelsize=14)
    axs[i].tick_params(axis='y', labelsize=14)

axs[0].set_title('Training Set Correlations', size=15)
axs[1].set_title('Test Set Correlations', size=15)

plt.show()

#### 基于各种特征的生存分布
### 删除空值
# 临时把测试集和训练集从混合集中分开，建两个分析用的临时数据集
train_data = combined[:891]
test_data = combined[891:]

# 删除空值
train_data.dropna(axis=0, how='any', inplace=True)

train_temp = train_data

# 删除空值
test_data.dropna(axis=0, how='any', inplace=True)

test_temp = test_data

##基于年龄和票价的生存特征分布
cont_features = ['Age', 'Fare']
surv = train_temp['Survived'] == 1

fig, axs = plt.subplots(ncols=2, nrows=2, figsize=(10, 10))
plt.subplots_adjust(right=1.5)

for i, feature in enumerate(cont_features):
    # 生存的特征分布
    sns.distplot(train_temp[~surv][feature], label='Not Survived', hist=True, color='#e74c3c', ax=axs[0][i])
    sns.distplot(train_temp[surv][feature], label='Survived', hist=True, color='#2ecc71', ax=axs[0][i])

    # 各个特征在数据集中的分布
    sns.distplot(train_temp[feature], label='Training Set', hist=False, color='#e74c3c', ax=axs[1][i])
    sns.distplot(test_temp[feature], label='Test Set', hist=False, color='#2ecc71', ax=axs[1][i])

    axs[0][i].set_xlabel('')
    axs[1][i].set_xlabel('')

    for j in range(2):
        axs[i][j].tick_params(axis='x', labelsize=10)
        axs[i][j].tick_params(axis='y', labelsize=10)

    axs[0][i].legend(loc='upper right', prop={'size': 10})
    axs[1][i].legend(loc='upper right', prop={'size': 10})
    axs[0][i].set_title('关于 {} 特征的生存分布'.format(feature), size=10, y=1.05)

axs[1][0].set_title('关于 {} 特征分布'.format('Age'), size=10, y=1.05)
axs[1][1].set_title('关于 {} 特征分布'.format('Fare'), size=10, y=1.05)

plt.show()

##基于其他几个特征的生存分布
cat_features = ['Embarked', 'Parch', 'Pclass', 'Sex', 'SibSp']

fig, axs = plt.subplots(ncols=2, nrows=3, figsize=(10, 10))
plt.subplots_adjust(right=1.5, top=1.25)

for i, feature in enumerate(cat_features, 1):
    plt.subplot(2, 3, i)
    sns.countplot(x=feature, hue='Survived', data=train_temp)

    plt.xlabel('{}'.format(feature), size=10, labelpad=15)
    plt.ylabel('乘客数量', size=10, labelpad=15)
    plt.tick_params(axis='x', labelsize=10)
    plt.tick_params(axis='y', labelsize=10)

    plt.legend(['Not Survived', 'Survived'], loc='upper center', prop={'size': 12})
    plt.title('在 {} 特征下的生存数量统计'.format(feature), size=8, y=1.05)

plt.show()

####特征数据处理
###Embarked处理—众数填充缺失值
# 取Embarked的众数（也就是数值最多的）
print(combined['Embarked'].mode())
# 取Embarked的众数第一行
print(combined['Embarked'].mode().iloc[0])

#总共缺失2个，采用众数填充
if combined['Embarked'].isnull().sum() != 0:
    combined['Embarked'].fillna(combined['Embarked'].mode().iloc[0], inplace=True)

combined.info()
###Name处理
##从名称中提取称呼
import re
# 在下面的代码中，我们通过正则提取了Title特征，正则表达式为(\w+\.)，它会在Name特征里匹配第一个以“.”号为结束的单词。同时，指定expand=False的参数会返回一个DataFrame。
# 西方姓名中间会加入称呼，比如小男童会在名字中间加入Master，女性根据年龄段及婚姻状况不同也会使用Miss 或 Mrs 等
# 这算是基于业务的理解做的衍生特征，原作者应该是考虑可以用作区分人的特征因此在此尝试清洗数据后加入

combined['Title'] = combined['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print(combined['Title'])
print(pd.crosstab(combined['Title'], combined['Sex']))#交叉取值
print(combined['Title'].value_counts())
###将名称分类
combined['Title'].replace(['Capt', 'Col', 'Major', 'Dr', 'Rev'], 'Officer',inplace = True)
combined['Title'].replace(['Jonkheer', 'Don', 'Sir', 'Countess', 'Dona', 'Lady'], 'Royalty',inplace =True )
combined['Title'].replace(['Mlle', 'Miss'], 'Miss',inplace =True )
combined['Title'].replace('Ms', 'Miss',inplace =True )
combined['Title'].replace(['Mme', 'Mrs'], 'Mrs',inplace =True )
combined['Title'].replace(['Mr'], 'Mr',inplace =True )
combined['Title'].replace(['Master'], 'Master',inplace =True )
print(combined['Title'].value_counts())

##下面验证一下包含Master的都是小童
# 使用一个临时数据表
temp = combined[combined['Title'].str.contains('Master')]
print(temp['Age'].value_counts())
# 查看年龄分段后的生存率
print(combined[['Title', 'Survived']].groupby(['Title'], as_index=False).mean())

###Fare处理
##填充缺失值
#检查缺失值个数
print(combined['Fare'].isnull().sum())
print(combined[combined['Fare'].isnull()])

# 按一二三等舱各自的均价来对应填充NaN
if combined['Fare'].isnull().sum() != 0:
    combined['Fare'] = combined[['Fare']].fillna(combined.groupby('Pclass').transform('mean'))

# 查看填充后的数据
print(combined.iloc[1043])
##使用qcut切分Fare
combined['Fare_Category'] = pd.qcut(combined['Fare'],13)
print(combined['Fare_Category'])

###SibSp和处理,这两组数据都能显著影响到Survived，但是影响方式不完全相同，所以将这两项合并成FamilySize组的同时保留这两项。Parch
# 合并SibSp和Parch，得到家庭成员总数
combined['FamilySize'] = combined['Parch'] + combined['SibSp'] + 1
print(combined['FamilySize'])
print(combined[['FamilySize', 'Survived']].groupby(['FamilySize'], as_index=False).mean().sort_values(by='Survived', ascending=False))
# 把家庭成员总数做分段处理
combined['FamilySizeCategory']=combined['FamilySize'].map(lambda x:'Single' if x<2 else 'small' if x<4 else 'middle' if x<8 else 'large')
print(combined['FamilySizeCategory'])

###Age处理

#求出Age为非空，同时Name中包含的Master的乘客年龄均值
ZZ = combined[combined['Age'].notnull() & combined['Title'].str.contains('Master')]['Age'].mean()
print(ZZ)
combined.loc[65,'Age'] = ZZ
combined.loc[159,'Age'] = ZZ
combined.loc[176,'Age'] = ZZ
combined.loc[709,'Age'] = ZZ
combined.loc[244,'Age'] = ZZ
combined.loc[339,'Age'] = ZZ
combined.loc[344,'Age'] = ZZ
combined.loc[417,'Age'] = ZZ

combined.to_csv('data/com.csv')
###使用混合预测模型预测Age
missing_age_df = pd.DataFrame(combined[['Pclass', 'Name', 'Sex', 'Age', 'FamilySize', 'FamilySizeCategory','Fare','Embarked', 'Title']])
print(missing_age_df)
missing_age_df = pd.get_dummies(missing_age_df,columns=[ 'Name', 'Sex','Embarked','FamilySizeCategory','Title'])
print(missing_age_df)
# 注意，这里没有对数值型数据做标准化处理
missing_age_train = missing_age_df[missing_age_df['Age'].notnull()]
print(missing_age_train)
missing_age_test = missing_age_df[missing_age_df['Age'].isnull()]
print(missing_age_test)

from sklearn import ensemble
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from sklearn import ensemble
from sklearn import model_selection
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.preprocessing import StandardScaler


def fill_missing_age(missing_age_train, missing_age_test):
    missing_age_X_train = missing_age_train.drop(['Age'], axis=1)
    missing_age_Y_train = missing_age_train['Age']
    missing_age_X_test = missing_age_test.drop(['Age'], axis=1)

    # 这里对训练数据做了标准化处理，原作者没有做，我做的也不一定就对
    missing_age_X_train = StandardScaler().fit_transform(missing_age_X_train)
    missing_age_X_test = StandardScaler().fit_transform(missing_age_X_test)

    # GBM模型预测
    gbm_reg = GradientBoostingRegressor(random_state=42)
    gbm_reg_param_grid = {'n_estimators': [2000], 'max_depth': [4], 'learning_rate': [0.01], 'max_features': [3]}
    gbm_reg_grid = model_selection.GridSearchCV(gbm_reg, gbm_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                                scoring='neg_mean_squared_error')
    gbm_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best GB Params:' + str(gbm_reg_grid.best_params_))
    print('Age feature Best GB Score:' + str(gbm_reg_grid.best_score_))
    print('GB Train Error for "Age" Feature Regressor:' + str(
        gbm_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_GB'] = gbm_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_GB'][:4])

    # 随机森林模型预测
    rf_reg = RandomForestRegressor()
    rf_reg_param_grid = {'n_estimators': [200], 'max_depth': [5], 'random_state': [0]}
    rf_reg_grid = model_selection.GridSearchCV(rf_reg, rf_reg_param_grid, cv=10, n_jobs=25, verbose=1,
                                               scoring='neg_mean_squared_error')
    rf_reg_grid.fit(missing_age_X_train, missing_age_Y_train)
    print('Age feature Best RF Params:' + str(rf_reg_grid.best_params_))
    print('Age feature Best RF Score:' + str(rf_reg_grid.best_score_))
    print(
        'RF Train Error for "Age" Feature Regressor' + str(rf_reg_grid.score(missing_age_X_train, missing_age_Y_train)))
    missing_age_test.loc[:, 'Age_RF'] = rf_reg_grid.predict(missing_age_X_test)
    print(missing_age_test['Age_RF'][:4])

    # 模型预测结果合并
    print('shape1', missing_age_test['Age'].shape, missing_age_test[['Age_GB', 'Age_RF']].mode(axis=1).shape)
    # missing_age_test['Age'] = missing_age_test[['Age_GB', 'Age_LR']].mode(axis=1)

    missing_age_test.loc[:, 'Age'] = np.mean([missing_age_test['Age_GB'], missing_age_test['Age_RF']])
    print(missing_age_test['Age'][:4])

    # 做了标准化以后，数据会变成np.array格式，这里再做一次转换
    missing_age_test = pd.DataFrame(missing_age_test)
    missing_age_test.drop(['Age_GB', 'Age_RF'], axis=1, inplace=True)

    return missing_age_test
combined.loc[(combined.Age.isnull()), 'Age'] = fill_missing_age(missing_age_train, missing_age_test)
##Age分段处理
combined['Age_group'] = combined['Age'].map(lambda x: 'child' if x<12 else 'youth' if x<18 else 'adult' if x<30 else 'middle' if x<50 else 'old' if x<70 else 'too old' if x>=70 else 'null')

by_age = combined.groupby('Age_group')['Survived'].mean()

print(by_age)

##Cabin处理
# 创建Deck列，根据Cabin列的第一个字母（M表示missing）
# Creating Deck column from the first letter of the Cabin column (M stands for Missing)
combined['Deck'] = combined['Cabin'].apply(lambda s: s[0] if pd.notnull(s) else 'M')

combined_decks = combined.groupby(['Deck', 'Pclass']).count().drop(columns=['Survived', 'Sex', 'Age', 'SibSp', 'Parch',
                                                                            'Fare', 'Embarked', 'Cabin', 'PassengerId',
                                                                            'Ticket']).rename(
    columns={'Name': 'Count'}).transpose()


def get_pclass_dist(df):
    # Creating a dictionary for every passenger class count in every deck
    deck_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}, 'T': {}}
    decks = df.columns.levels[0]

    for deck in decks:
        for pclass in range(1, 4):
            try:
                count = df[deck][pclass][0]
                deck_counts[deck][pclass] = count
            except KeyError:
                deck_counts[deck][pclass] = 0

    df_decks = pd.DataFrame(deck_counts)
    deck_percentages = {}

    # Creating a dictionary for every passenger class percentage in every deck
    for col in df_decks.columns:
        deck_percentages[col] = [(count / df_decks[col].sum()) * 100 for count in df_decks[col]]

    return deck_counts, deck_percentages


def display_pclass_dist(percentages):
    df_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M', 'T')
    bar_count = np.arange(len(deck_names))
    bar_width = 0.85

    pclass1 = df_percentages[0]
    pclass2 = df_percentages[1]
    pclass3 = df_percentages[2]

    plt.figure(figsize=(10, 5))
    plt.bar(bar_count, pclass1, color='#b5ffb9', edgecolor='white', width=bar_width, label='Passenger Class 1')
    plt.bar(bar_count, pclass2, bottom=pclass1, color='#f9bc86', edgecolor='white', width=bar_width,
            label='Passenger Class 2')
    plt.bar(bar_count, pclass3, bottom=pclass1 + pclass2, color='#a3acff', edgecolor='white', width=bar_width,
            label='Passenger Class 3')

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Passenger Class Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Passenger Class Distribution in Decks', size=18, y=1.05)

    plt.show()


all_deck_count, all_deck_per = get_pclass_dist(combined_decks)
display_pclass_dist(all_deck_per)

# 把T甲板的乘客改到A甲板
# Passenger in the T deck is changed to A
idx = combined[combined['Deck'] == 'T'].index
combined.loc[idx, 'Deck'] = 'A'

combined_decks_survived = combined.groupby(['Deck', 'Survived']).count().drop(
    columns=['Sex', 'Age', 'SibSp', 'Parch', 'Fare',
             'Embarked', 'Pclass', 'Cabin', 'PassengerId', 'Ticket']).rename(columns={'Name': 'Count'}).transpose()


def get_survived_dist(df):
    # Creating a dictionary for every survival count in every deck
    surv_counts = {'A': {}, 'B': {}, 'C': {}, 'D': {}, 'E': {}, 'F': {}, 'G': {}, 'M': {}}
    decks = df.columns.levels[0]

    for deck in decks:
        for survive in range(0, 2):
            surv_counts[deck][survive] = df[deck][survive][0]

    df_surv = pd.DataFrame(surv_counts)
    surv_percentages = {}

    for col in df_surv.columns:
        surv_percentages[col] = [(count / df_surv[col].sum()) * 100 for count in df_surv[col]]

    return surv_counts, surv_percentages


def display_surv_dist(percentages):
    df_survived_percentages = pd.DataFrame(percentages).transpose()
    deck_names = ('A', 'B', 'C', 'D', 'E', 'F', 'G', 'M')
    bar_count = np.arange(len(deck_names))
    bar_width = 0.85

    not_survived = df_survived_percentages[0]
    survived = df_survived_percentages[1]

    plt.figure(figsize=(10, 5))
    plt.bar(bar_count, not_survived, color='#b5ffb9', edgecolor='white', width=bar_width, label="Not Survived")
    plt.bar(bar_count, survived, bottom=not_survived, color='#f9bc86', edgecolor='white', width=bar_width,
            label="Survived")

    plt.xlabel('Deck', size=15, labelpad=20)
    plt.ylabel('Survival Percentage', size=15, labelpad=20)
    plt.xticks(bar_count, deck_names)
    plt.tick_params(axis='x', labelsize=15)
    plt.tick_params(axis='y', labelsize=15)

    plt.legend(loc='upper left', bbox_to_anchor=(1, 1), prop={'size': 15})
    plt.title('Survival Percentage in Decks', size=18, y=1.05)

    plt.show()


all_surv_count, all_surv_per = get_survived_dist(combined_decks_survived)
display_surv_dist(all_surv_per)

combined['Deck'] = combined['Deck'].replace(['A', 'B', 'C'], 'ABC')
combined['Deck'] = combined['Deck'].replace(['D', 'E'], 'DE')
combined['Deck'] = combined['Deck'].replace(['F', 'G'], 'FG')

combined['Deck'].value_counts()

###Family和Tiket处理
combined['Ticket_Frequency'] = combined.groupby('Ticket')['Ticket'].transform('count')

fig, axs = plt.subplots(figsize=(12, 9))
sns.countplot(x='Ticket_Frequency', hue='Survived', data=combined)

plt.xlabel('Ticket Frequency', size=15, labelpad=20)
plt.ylabel('Passenger Count', size=15, labelpad=20)
plt.tick_params(axis='x', labelsize=15)
plt.tick_params(axis='y', labelsize=15)

plt.legend(['Not Survived', 'Survived'], loc='upper right', prop={'size': 15})
plt.title('Count of Survival in {} Feature'.format('Ticket Frequency'), size=15, y=1.05)

plt.show()


def extract_surname(data):
    families = []

    for i in range(len(data)):
        name = data.iloc[i]

        if '(' in name:
            name_no_bracket = name.split('(')[0]
        else:
            name_no_bracket = name

        family = name_no_bracket.split(',')[0]
        title = name_no_bracket.split(',')[1].strip().split(' ')[0]

        for c in string.punctuation:
            family = family.replace(c, '').strip()

        families.append(family)

    return families


combined['Family'] = extract_surname(combined['Name'])

train = combined.loc[:890]
test = combined[891:]
dfs = [train, test]
##下面这段是为了创建一个同事存在于训练集合测试集的家庭和Ticket列表
non_unique_families = [x for x in train['Family'].unique() if x in test['Family'].unique()]
non_unique_tickets = [x for x in train['Ticket'].unique() if x in test['Ticket'].unique()]

df_family_survival_rate = train.groupby('Family')['Survived', 'Family','FamilySize'].median()
df_ticket_survival_rate = train.groupby('Ticket')['Survived', 'Ticket','Ticket_Frequency'].median()

family_rates = {}
ticket_rates = {}

for i in range(len(df_family_survival_rate)):
    # Checking a family exists in both training and test set, and has members more than 1
    if df_family_survival_rate.index[i] in non_unique_families and df_family_survival_rate.iloc[i, 1] > 1:
        family_rates[df_family_survival_rate.index[i]] = df_family_survival_rate.iloc[i, 0]

for i in range(len(df_ticket_survival_rate)):
    # Checking a ticket exists in both training and test set, and has members more than 1
    if df_ticket_survival_rate.index[i] in non_unique_tickets and df_ticket_survival_rate.iloc[i, 1] > 1:
        ticket_rates[df_ticket_survival_rate.index[i]] = df_ticket_survival_rate.iloc[i, 0]

    mean_survival_rate = np.mean(train['Survived'])

    train_family_survival_rate = []
    train_family_survival_rate_NA = []
    test_family_survival_rate = []
    test_family_survival_rate_NA = []

    for i in range(len(train)):
        if train['Family'][i] in family_rates:
            train_family_survival_rate.append(family_rates[train['Family'][i]])
            train_family_survival_rate_NA.append(1)
        else:
            train_family_survival_rate.append(mean_survival_rate)
            train_family_survival_rate_NA.append(0)

    for i in range(len(test)):
        if test['Family'].iloc[i] in family_rates:
            test_family_survival_rate.append(family_rates[test['Family'].iloc[i]])
            test_family_survival_rate_NA.append(1)
        else:
            test_family_survival_rate.append(mean_survival_rate)
            test_family_survival_rate_NA.append(0)

    train['Family_Survival_Rate'] = train_family_survival_rate
    train['Family_Survival_Rate_NA'] = train_family_survival_rate_NA
    test['Family_Survival_Rate'] = test_family_survival_rate
    test['Family_Survival_Rate_NA'] = test_family_survival_rate_NA

    train_ticket_survival_rate = []
    train_ticket_survival_rate_NA = []
    test_ticket_survival_rate = []
    test_ticket_survival_rate_NA = []

    for i in range(len(train)):
        if train['Ticket'][i] in ticket_rates:
            train_ticket_survival_rate.append(ticket_rates[train['Ticket'][i]])
            train_ticket_survival_rate_NA.append(1)
        else:
            train_ticket_survival_rate.append(mean_survival_rate)
            train_ticket_survival_rate_NA.append(0)

    for i in range(len(test)):
        if test['Ticket'].iloc[i] in ticket_rates:
            test_ticket_survival_rate.append(ticket_rates[test['Ticket'].iloc[i]])
            test_ticket_survival_rate_NA.append(1)
        else:
            test_ticket_survival_rate.append(mean_survival_rate)
            test_ticket_survival_rate_NA.append(0)

    train['Ticket_Survival_Rate'] = train_ticket_survival_rate
    train['Ticket_Survival_Rate_NA'] = train_ticket_survival_rate_NA
    test['Ticket_Survival_Rate'] = test_ticket_survival_rate
    test['Ticket_Survival_Rate_NA'] = test_ticket_survival_rate_NA

#把基于家庭计算的生存率和基于Tiket计算的生存率做个平均
for df in [train, test]:
    df['Survival_Rate'] = (df['Ticket_Survival_Rate'] + df['Family_Survival_Rate']) / 2
    df['Survival_Rate_NA'] = (df['Ticket_Survival_Rate_NA'] + df['Family_Survival_Rate_NA']) / 2

#保存数据
train_data = train
test_data =test
train_data.to_csv('data/train_data.csv',index= False)
test_data.to_csv('data/test_data.csv',index= False)