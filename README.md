# 人力资源“综合效益”评估——多元线性回归
## 一.项目背景介绍
* 选择（[人力资源数据集](https://aistudio.baidu.com/aistudio/datasetdetail/106631/0) ）数据集，做数据可视化分析，有效用于调节整体人力资源的多样性和激发个体优势潜能。
### 项目背景及意义：

#### 项目背景：

* 在新经济中，胜利将来源于组织能力，包括速度、响应性、敏捷性、学习能力和员工素质。

* 人力资源是指能够推动国民经济和社会发展的、具有智力劳动和体力劳动能力的人们的总和。

* 人事管理的升级，是指在经济学与人本思想指导下，通过招聘、甄选、培训、报酬等管理形式对组织内外相关人力资源进行有效运用，满足组织当前及未来发展的需要，保证组织目标实现与成员发展的最大化的一系列活动的总称。

* 切合“实现三大战略性转变：由再现型素质向开拓创新型素质转化；由内向型素质向外向型、国际通用型素质转化；由单一型素质向复合型素质转化。”战略构想。

#### 项目意义：

* “最优化”人员资源配置，提升综合效益和员工荣誉获得感。

* 提升组织能力、推动战略目标实现。

## 二、数据介绍
* 名称：人力资源数据集
* 格式：csv
* 来源：Carla Patalano博士和朋友着手创建自己的与HR相关的数据集
* 核心内容：包含姓名，出生日期，年龄，性别，婚姻状况，雇用日期，离职原因，部门，是否处于活跃状态或离职，职位名称，薪资水平，经理姓名，和性能得分。

## 三、模型介绍
* 多元线性回归：经典的线性回归模型主要用来预测一些存在着线性关系的数据集，实质是多个权重x映射到y。
* 其思路是，假设人力资源数据集中的工资、活跃度等因素和评分效益之间的关系可以被属性间的线性组合描述。在模型训练阶段，让假设的预测结果和真实值之间的误差越来越小。在模型预测阶段，预测器会读取训练好的模型，对从未遇见过的人力资源属性进行效益初步评估。

## 四、模型训练
import paddle
import numpy as np
import os
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import warnings
warnings.filterwarnings("ignore")
# 从文件导入数据
datafile = './HRE.data'
HRE_data = np.fromfile(datafile, sep=' ')
feature_names = ['SPC','ESa','sa','ESu']
feature_num = len(feature_names)
# 将原始数据进行Reshape，变成[N, 4]这样的形状
HRE_data = HRE_data.reshape([HRE_data.shape[0] // feature_num, feature_num])

# 画图看特征间的关系,主要是变量两两之间的关系（线性或非线性，有无明显较为相关关系）
features_np = np.array([x[:3] for x in HRE_data], np.float32)
labels_np = np.array([x[-1] for x in HRE_data], np.float32)
# data_np = np.c_[features_np, labels_np]
df = pd.DataFrame(HRE_data, columns=feature_names)
matplotlib.use('TkAgg')
%matplotlib inline
sns.pairplot(df.dropna(), y_vars=feature_names[-1], x_vars=feature_names[::-1], diag_kind='kde')
plt.show()

# 相关性分析
fig, ax = plt.subplots(figsize=(5, 1)) 
corr_data = df.corr().iloc[-1]
corr_data = np.asarray(corr_data).reshape(1, 4)
ax = sns.heatmap(corr_data, cbar=True, annot=True)
plt.show()

features_max = HRE_data.max(axis=0)
features_min = HRE_data.min(axis=0)
features_avg = HRE_data.sum(axis=0) / HRE_data.shape[0]

BATCH_SIZE = 20
def feature_norm(input):
    f_size = input.shape
    output_features = np.zeros(f_size, np.float32)
    for batch_id in range(f_size[0]):
        for index in range(3):
            output_features[batch_id][index] = (input[batch_id][index] - features_avg[index]) / (features_max[index] - features_min[index])
    return output_features 

# 只对属性进行归一化
HRE_features = feature_norm(HRE_data[:, :3])
# print(feature_trian.shape)
HRE_data = np.c_[HRE_features, HRE_data[:, -1]].astype(np.float32)
# print(training_data[0])

# 归一化后的train_data, 看下各属性的情况
features_np = np.array([x[:3] for x in HRE_data],np.float32)
labels_np = np.array([x[-1] for x in HRE_data],np.float32)
data_np = np.c_[features_np, labels_np]
df = pd.DataFrame(data_np, columns=feature_names)
sns.boxplot(data=df.iloc[:, 0:3])

# 将训练数据集和测试数据集按照8:2的比例分开
ratio = 0.8
offset = int(HRE_data.shape[0] * ratio)
train_data = HRE_data[:offset]
test_data = HRE_data[offset:]

class Regressor(paddle.nn.Layer):
    def __init__(self):
        super(Regressor, self).__init__()
        self.fc = paddle.nn.Linear(3, 1,)

    def forward(self, inputs):
        pred = self.fc(inputs)
        return pred

train_nums = []
train_costs = []

def draw_train_process(iters, train_costs):
    plt.title("training cost", fontsize=23)
    plt.xlabel("iter", fontsize=13)
    plt.ylabel("cost", fontsize=13)
    plt.plot(iters, train_costs, color='red', label='training cost')
    plt.show()
