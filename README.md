# tansformer-
利用transformer对告警序列进行预测

### 一、原始数据
该数据来自于中国移动举办的基站退服告警预测。（网址：https://js.aiiaorg.cn/match/matchitem/85）
数据形式如下：
告警开始时间	基站名称	告警名称
2019/12/23 15:31	ACZDoAAEEAAAI1wAC8	BBU风扇堵转告警
2019/12/23 16:23	ACZDoAAEEAAAI1wAC8	BBU风扇堵转告警
2019/12/23 16:23	ACZDoAAEEAAAI1wAC8	BBU风扇堵转告警
2019/12/23 17:18	ACZDoAAEEAAAI1wAC8	BBU风扇堵转告警
2019/12/25 15:12	ACZDoAAEEAAAI1wAC8	X2接口故障告警
2019/12/31 7:00	ACZDoAAEEAAAI1wAC8	BBU风扇堵转告警
2019/12/31 10:18	ACZDoAAEEAAAI1wAC8	X2接口故障告警
2020/1/1 9:09	ACZDoAAEEAAAI1wAC8	小区接收通道干扰噪声功率不平衡告警
2020/1/1 16:56	ACZDoAAEEAAAI1wAC8	小区接收通道干扰噪声功率不平衡告警
2020/1/1 19:09	ACZDoAAEEAAAI1wAC8	小区接收通道干扰噪声功率不平衡告警
2020/1/3 9:12	ACZDoAAEEAAAI1wAC8	X2接口故障告警
2020/1/5 10:29	ACZDoAAEEAAAI1wAC8	网元连接中断

### 二、目的
预测未来1天是否会出现重要告警（重要告警有两类，分别是网元连接中断和小区服务中断。）
数据存在严重的样本不平衡现象（label1 : label0 = 0.07:1）。

### 三、transformer
将某个基站的前7天的告警名称作为文本序列，经过embedding（word2vector）后输入到transformer模型中。
因为数据存在样本不平衡现象，故在构建batch样本时，特意挑选50%的正样本，挑选50%的负样本，这样可以在
一定程度上缓解过拟合。当对loss函数根据样本比例去调整，模型效果并没有变好。不知道什么原因。

### 四、模型效果
模型验证集负的log交叉熵只能下降到0.44（前1000个iter即可），当训练集的loss继续下降时，验证机却下降不了。
提交线上的成绩最高0.5（f1 score）。

### 五、模型改进
如果以后有改进的方向，再加上来。
