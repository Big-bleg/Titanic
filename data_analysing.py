#导入pandas与numpy工具包
import pandas as pd
import numpy as np
#导入绘图工具包
import matplotlib.pyplot as plt
import seaborn as sns #要注意的是一旦导入了seaborn，matplotlib的默认作图风格就会被覆盖成seaborn的格式



'''数据说明
PassengerId：乘客Id
Survived：0代表NO，未能生还；1代表YES，生还；
Pclass：舱位等级，1,2,3，1最高，3最低；也就是俗称的一等舱、二等舱、三等舱；
Name：乘客姓名
Sex：乘客性别
Age：年龄
SibSp：由两部分组成，Sibling（兄弟姐妹，堂妹都合适），Spouse代表（丈夫或妻子）
Parch：父母和孩子组成，若只跟保姆写0
Ticket：票的数字
Fare：乘客票价
Cabin：船舱数字
Embarked：登船仓：C=Cherbourg，Q=Queenstown，S=Southampton¶
'''
#读取数据集
# 此处使用pandas，读取数据的同时转换为pandas独有的dataframe格式（二维数据表格）
train = pd.read_csv('data/train.csv')
test = pd.read_csv('data/test.csv')

#print(train.head())#显示数据内容
#print(train.describe())#对数据简单描述

# 使用pandas的info()，查看数据的统计特性
#print(train.info())#发现数据存在缺失值

plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
#以下为数据分析部分
##############################################################################################
###survived-生还情况
Survived_train=train['Survived'].value_counts()
print("查看训练集中的生还和死亡人数")
print(Survived_train)
Survived_train.plot.pie(autopct='%1.2f%%')
plt.title("用饼图显示饼图显示生还/死亡乘客对比")
plt.show()

###Pclass-舱位等级
Pclass_train=train['Pclass'].value_counts()
print("查看训练集舱位人数分布情况")
print(Pclass_train)
Pclass_train.plot.pie(autopct='%1.2f%%')
plt.title("用饼图显示训练集舱位人数分布情况")
plt.show()
sns.countplot(x='Pclass', hue='Survived',data=train)
plt.title("舱位等级与生存对比")
plt.show()
train[['Pclass','Survived']].groupby(['Pclass']).mean().plot.bar(color=[['r', 'g', 'b']])
plt.title("比较舱位等级存活率")
plt.show()#不同舱位等级下，一等舱的生存率明显很高；二等舱也比三等舱高很多。说明舱位等级的这个特征很重要。
train[['Sex','Pclass','Survived']].groupby(['Pclass','Sex']).mean().plot.bar()
plt.title("比较舱位等级中生存男女的比例")
plt.show()#不同舱位等级下，女性的生存率明显是高的。说明Sex这个特征很重要
print("查看不同舱位等级中男女生存数值")
print(train.groupby(['Sex','Pclass','Survived'])['Survived'].count())
#说明： 船仓等级越高，生存率越高

###Sex-乘客性别分布
print("查看训练集人员性别分布数值")
print(train['Sex'].value_counts())
train['Sex'].value_counts().plot.pie(autopct='%1.2f%%')
plt.title("饼图显示训练集人员性别分布")
plt.show()
print("查看性别与生存数值")
train.groupby(['Sex','Survived'])['Survived'].count()
sns.countplot(x="Sex", hue="Survived", data=train)
plt.title("柱状图显示人员性别与生存")
plt.show()
train[['Sex','Survived']].groupby(['Sex']).mean().plot.bar(color= [['r','g']])
plt.title("比较性别与生存平均比率")
plt.show()

###Age-年龄
print('查看乘客年龄分布')
print(train['Age'].value_counts())

# 查看总体的年龄分布
fig=plt.figure(figsize=(12,5))
plt.subplot(121)
train['Age'].hist(bins=70)
plt.xlabel('Age')
plt.ylabel('Num')

plt.subplot(122)
train.boxplot(column='Age', showfliers=False)
fig.suptitle("查看总体的年龄分布")
plt.show()

#乘客年龄分布与生存对比
g = sns.FacetGrid(train, col='Survived')
g.map(plt.hist, 'Age', bins=50)
plt.title('柱形图比对年龄-生存分布')
plt.show()
#小提琴图，比较年龄和生存关系
sns.violinplot(x='Survived',y='Age',data=train)
plt.title('小提琴图，比较年龄和生存关系')
plt.show()
# 小提琴图还是不大容易看出区别,换一种图形来对比一下
facet = sns.FacetGrid(train, hue="Survived",aspect=4)
facet.map(sns.kdeplot,'Age',shade= True)
facet.set(xlim=(0, train['Age'].max()))
facet.add_legend()
plt.show()#说明：1.低龄人群的生存比例高；2.30左右的人生存和死亡比例差不多；3.20-25左右的人群死亡率更高。
#乘客年龄分段对比生还情况
#年龄特征分段
train['Age']=train['Age'].map(lambda x: 'child' if x<12 else 'youth' if x<30 else 'adult' if x<60 else 'old' if x<70 else 'too old' if x>=70 else 'null')

# 柱状图比较分段后的年龄-生存关系
sns.countplot(x="Age", hue="Survived", data=train)
plt.title('柱状图比较分段后的年龄-生存关系')
plt.show()
#不同年龄下平均生存率
# 重新导入数据
train = pd.read_csv('data/train.csv')
# 因为年龄有缺失值，暂时用200填充缺失值
train["Age"].fillna('200',inplace = True)

# 按年龄划分的平均生存率
fig, axis1 = plt.subplots(1,1,figsize=(18,4))
#if train["Age"].isnull() == True:
train["Age_int"] = train["Age"].astype(int)
average_age = train[["Age_int", "Survived"]].groupby(['Age_int'],as_index=False).mean()
sns.barplot(x='Age_int', y='Survived', data=average_age)
plt.show()
# 舱位-年龄-性别之间的生存关系
#再次导入原训练集
train = pd.read_csv('data/train.csv')

# 小提琴图显示舱位-年龄-生存关系
f,ax=plt.subplots(1,2,figsize=(18,8))
sns.violinplot("Pclass","Age", hue="Survived", data=train,split=True,ax=ax[0])
ax[0].set_title('Pclass and Age vs Survived')
ax[0].set_yticks(range(0,110,10))
sns.violinplot("Sex","Age", hue="Survived", data=train,split=True,ax=ax[1])

# 小提琴显示年龄-性别-生存关系
ax[1].set_title('Sex and Age vs Survived')
ax[1].set_yticks(range(0,110,10))
plt.show()#说明：一等舱，生存者的年龄总体偏大。但相对一等舱的死亡者来说，偏低。二等舱、三等舱，生存者和死亡者的年龄都差不多。

##SibSp-船上的兄弟姐妹/配偶
# 查看训练集SibSp数据分布
print('查看训练集SibSp数据分布')
print(train['SibSp'].value_counts())
# 查看数据分布柱状图
train['SibSp'].value_counts().plot.bar()
plt.title('查看SibSp数据分布柱状图')
plt.show()
# 按照比例查看
print('按照比例查看SibSp数据分布')
print(train['SibSp'].value_counts(sorted))
train['SibSp'].value_counts(sorted).plot.bar()
plt.title('查看SibSp数据分布柱状图')
plt.show()
#  SibSp与生存对比
sns.countplot(x="SibSp", hue="Survived", data=train)
plt.title('查看不同SibSp人群下死亡/生存对比')
plt.show()
print(' 查看不同SibSp的生存比率')
print(train[['SibSp','Survived']].groupby(['SibSp']).mean())#说明SibSp为1、2的，生存比例比较高
train[['SibSp','Survived']].groupby(['SibSp']).mean().plot.bar().set_title('SibSp and Survived')
plt.title('查看不同SibSp的生存平均率')
plt.show()

###Parch-父母以及小孩的数量
print('查看训练集Parch数据分布')
print(train['Parch'].value_counts())
train['Parch'].value_counts().plot.bar()
plt.title('查看训练集数据分布柱状图')
plt.show()
sns.countplot(x="Parch", hue="Survived", data=train)
plt.title('查看Parch与生存对比')
plt.show()
train[['Parch','Survived']].groupby(['Parch']).mean().plot.bar().set_title('Parch and Survived')
plt.title('查看不同Parch下的生存比率')
plt.show()

###Ticket-船票编号
print('查看票号分布')
print(train['Ticket'].value_counts())

###Fare-船票花费
print('查看Fare分布')
print(train['Fare'].value_counts())
train['Fare'].value_counts().plot.bar()
plt.title('Fare数据分布柱状图')
plt.show()

feg=plt.figure(figsize=(12,5))
plt.subplot(121)
train['Fare'].hist(bins=70)
plt.xlabel('Fare')
plt.ylabel('Num')

plt.subplot(122)
train.boxplot(column='Fare', showfliers=False)
feg.suptitle('查看总体的船票花费分布')
plt.show()

train.groupby(['Fare','Pclass'])['Pclass'].count().plot.bar()
plt.title('查看船票与舱位对比')
plt.show()


# 小提琴图对比票价和生存关系
sns.violinplot(x='Survived',y='Fare',data=train)
plt.show()
# 上图显示的很不均匀分布
#用numpy库里的对数函数对Fare的数值进行对数转换
train['Fare']=train['Fare'].map(lambda x:np.log(x+1))
#作小提琴图：
sns.violinplot(x='Survived',y='Fare',data=train)
plt.show()#说明穷人还是死亡率高些。当然穷人的舱位一般是三等舱，舱位靠下，而且沉船的时候，肯定越靠下越不利。

###对团体票处理
#再次导入原训练集
train = pd.read_csv('data/train.csv')

# 提取团体票的计数值，形成一个新列
train['Group_Ticket'] = train['Fare'].groupby(by=train['Ticket']).transform('count')

# 团体票除以票价计数值，求出每张票价格
train['Fare'] = train['Fare'] / train['Group_Ticket']

# 删除临时列
train.drop(['Group_Ticket'], axis=1, inplace=True)

# 查看数据分布柱状图
train['Fare'].value_counts().plot.bar()
plt.show()
# 查看总体的船票花费分布
plt.figure(figsize=(12,5))
plt.subplot(121)
train['Fare'].hist(bins=100)
plt.xlabel('Fare')
plt.ylabel('Num')

plt.subplot(122)
train.boxplot(column='Fare', showfliers=False)
plt.show()

###Embarked- 乘客上船的港口
#不同符号代表的港口含义
#S = Southampton,南安普顿 （第1站）
#C = Cherbourg, 瑟堡 （第2站）
#Q = Queenstown, 皇后城 （第3站）
# 查看数据分布
print('查看Embarked数据分布')
print(train['Embarked'].value_counts())
# 查看数据分布柱状图
train['Embarked'].value_counts().plot.bar()
plt.title('查看Embarked树状分布')
plt.show()
# 查看数据分布饼图
train['Embarked'].value_counts().plot.pie(autopct='%1.2f%%')
plt.title('查看Embarked饼状分布')
plt.show()
##Embarked与生存对比
# Emabarked人群与生存对比
sns.countplot(x="Embarked", hue="Survived", data=train)
plt.title('Emabarked人群与生存对比')
plt.show()
##登船港口-舱位对比
# 不同登船港口-舱位对比
sns.countplot(x="Embarked", hue="Pclass", data=train)
plt.title('不同登船港口-舱位对比')
plt.show()

### Cabin—船舱的编号
print('查看Cabin数据分布')
print(train['Cabin'].value_counts())
# 船舱编号-舱位对比
sns.countplot(x="Cabin", hue="Pclass", data=train)
plt.title('船舱编号-舱位对比')
plt.show()
##Cabin-生存对比
#有编号的为yes,没有的为no
train['Cabin']=train['Cabin'].map(lambda x:'yes' if type(x)==str else 'no')
#作图
sns.countplot(x="Cabin", hue="Survived", data=train)
plt.title('Cabin-生存对比')
plt.show()