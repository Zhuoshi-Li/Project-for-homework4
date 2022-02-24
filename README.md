# 人力资源“综合效益”评估——多元线性回归
## 一、项目背景介绍
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
```
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
```

![Obvious](https://github.com/Zhuoshi-Li/Project-for-homework4/blob/main/Obvious.png)
![Analysis](https://github.com/Zhuoshi-Li/Project-for-homework4/blob/main/Analysis.png)
![Normalization](https://github.com/Zhuoshi-Li/Project-for-homework4/blob/main/Normalization.png)

>模型参数
>>lr_schedule：learning_rate=0.001

>>optimize： optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())

>>epoch： EPOCH_NUM = 500

>>batch_size：INFER_BATCH_SIZE = 63

>>Loss function： paddle.nn.MSELoss()

```
import paddle.nn.functional as F 
y_preds = []
labels_list = []

def train(model):
    print('start training ... ')
    # 开启模型训练模式
    model.train()
    EPOCH_NUM = 500
    train_num = 0
    optimizer = paddle.optimizer.SGD(learning_rate=0.001, parameters=model.parameters())
    for epoch_id in range(EPOCH_NUM):
        # 在每轮迭代开始之前，将训练数据的顺序随机的打乱
        np.random.shuffle(train_data)
        # 将训练数据进行拆分，每个batch包含20条数据
        mini_batches = [train_data[k: k+BATCH_SIZE] for k in range(0, len(train_data), BATCH_SIZE)]
        for batch_id, data in enumerate(mini_batches):
            features_np = np.array(data[:, :3], np.float64)
            labels_np = np.array(data[:, -1:], np.float64)
            features = paddle.to_tensor(features_np)
            labels = paddle.to_tensor(labels_np)
            # 前向计算
            y_pred = model(features)
            cost = F.mse_loss(y_pred, label=labels)
            train_cost = cost.numpy()[0]
            # 反向传播
            cost.backward()
            # 最小化loss，更新参数
            optimizer.step()
            # 清除梯度
            optimizer.clear_grad()
            
            if batch_id%30 == 0 and epoch_id%50 == 0:
                print("Pass:%d,Cost:%0.5f"%(epoch_id, train_cost))

            train_num = train_num + BATCH_SIZE
            train_nums.append(train_num)
            train_costs.append(train_cost)
        
model = Regressor()
train(model)

matplotlib.use('TkAgg')
%matplotlib inline
draw_train_process(train_nums, train_costs)
```
* start training ... 
* Pass:0,Cost:20.46077
* Pass:50,Cost:1.35356
* Pass:100,Cost:0.71079
* Pass:150,Cost:0.31794
* Pass:200,Cost:0.54955
* Pass:250,Cost:0.60700
* Pass:300,Cost:0.41050
* Pass:350,Cost:0.98337
* Pass:400,Cost:0.62748
* Pass:450,Cost:0.25535

![Accuracy](https://github.com/Zhuoshi-Li/Project-for-homework4/blob/main/Accuracy.png)
## 五、模型评估
```
# 获取预测数据
INFER_BATCH_SIZE = 63

infer_features_np = np.array([data[:3] for data in test_data]).astype("float64")
infer_labels_np = np.array([data[-1] for data in test_data]).astype("float64")

infer_features = paddle.to_tensor(infer_features_np)
infer_labels = paddle.to_tensor(infer_labels_np)
fetch_list = model(infer_features)

sum_cost = 0
for i in range(INFER_BATCH_SIZE):
    infer_result = fetch_list[i][0]
    ground_truth = infer_labels[i]
    if i % 10 == 0:
        print("No.%d: infer result is %.2f,ground truth is %.2f" % (i, infer_result, ground_truth))
    cost = paddle.pow(infer_result - ground_truth, 2)
    sum_cost += cost
mean_loss = sum_cost / INFER_BATCH_SIZE
print("Mean loss is:", mean_loss.numpy())

def plot_pred_ground(pred, ground):
    plt.figure()   
    plt.title("Predication v.s. Ground truth", fontsize=23)
    plt.xlabel("ground truth price(unit:$1000)", fontsize=4)
    plt.ylabel("predict price", fontsize=4)
    plt.scatter(ground, pred, alpha=0.5)  #  scatter:散点图,alpha:"透明度"
    plt.plot(ground, ground, c='red')
    plt.show()

plot_pred_ground(fetch_list, infer_labels_np)
```
* No.0: infer result is 4.15,ground truth is 4.30
* No.10: infer result is 4.00,ground truth is 2.34
* No.20: infer result is 4.29,ground truth is 4.24
* No.30: infer result is 4.15,ground truth is 3.02
* No.40: infer result is 4.15,ground truth is 3.66
* No.50: infer result is 4.29,ground truth is 3.45
* No.60: infer result is 3.86,ground truth is 3.00
* Mean loss is: [1.18700965]

![Prediction](https://github.com/Zhuoshi-Li/Project-for-homework4/blob/main/Prediction.png)
## 六、总结与升华
>本项目采用多元线性回归的方法分析工资以及评价机制对员工行动力和效益的影响，亮点是提取出关键的数据集，不足之处有数据体现的仍不充分，需要更加多维和健全的指标进行人力资源的综合评估，以后需要扩大精选的数据集，进行更加全面客观的分析。
## 七、个人总结
>我是东北大学悉尼智能科技学院的大一新生，主修计算机科学与技术专业，对Al有着浓厚的兴趣，在本次飞桨领航团AI达人创造营收获颇丰，感受到讲师的极其扎实的专业水平和育人之心。我的兴趣方向是目标检测，希望在此学习基础上，能找机会利用优质免费的飞桨平台开发一个目标检测的项目，谢谢！

## 提交链接
### aistudio链接：

### github链接：
[github链接](https://github.com/Zhuoshi-Li/Project-for-homework4/edit/main/README.md)
### gitee链接：
