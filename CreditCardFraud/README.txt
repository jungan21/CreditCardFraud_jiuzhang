Jiuzhang LintCode: https://www.lintcode.com/ai/creditcard_fraud/overview
Data: https://www.lintcode.com/ai/creditcard_fraud/data

#题目描述
信用卡公司需要能够识别虚假的信用卡交易，以避免用户被骗。本题就是利用欧洲某段时间的信用卡交易记录，来判断某条交易是否属于信用卡欺诈。

#数据描述
数据集为2013年9月里某两天欧元区的信用卡交易记录。在这两天中共有284807笔交易，其中的492笔是欺诈。把欺诈交易的类(class) 认为是1，非欺诈交易的类认为是0. 那么这个二元问题中，class为1的样本概率只有0.172%。可见在这个二元问题中，两个类所占的比重相差特别大。也就是说我们的数据集是特别不平衡的。因此同学们在本题中要思考如何处理这种失衡的数据。失衡数据很可能导致train出来的模型是无效模型，具体会在下面“小提示”部分详细说明。

数据集里面有从PCA转换得到的28个features。在下方我放了一点关于PCA转换的阅读材料，感兴趣的同学可以阅读了解一下。**但是不了解不影响答题，不建议过多把时间花在上面。**之所以进行了PCA转换是因为不能把原始的消费者信用卡记录暴露给公众，这样触犯了用户的隐私。所以做PCA转换处理。可以简单理解为加密。

Features V1, V2, ..., V28都是经过PCA转换后获得的features。没有被PCA转换的，保留了原始数据的是"Time"和"Amount"。"Time"记录了每笔交易和第一笔交易之间的时间间隔，以秒为单位。"Amount"是交易数额。"Class"就是这笔交易的最终分类。如果为1说明这是一笔欺诈交易，为0说明这不是一笔欺诈交易。

PCA Transformation
1. https://en.wikipedia.org/wiki/Principal_component_analysis
2. https://www.cs.princeton.edu/picasso/mats/PCA-Tutorial-Intuition_jp.pdf

#小提示
本题数据是非常不平衡的，只有0.172%的交易为欺诈交易。所以如果模型什么也不做，直接把所有的交易预测为非欺诈，也会有超过90%的准确率。因此准确率在本题中不是衡量模型好坏的有效指标。而且对于银行来说，检验到欺诈交易远比检验到非欺诈交易重要。因此本题用来衡量模型好坏的指标是F1值。F1值有既考虑precision，又考虑recall的优点。

可以尝试使用Resampling，包括oversampling, undersampling来处理unbalance数据。参考资料:

http://contrib.scikit-learn.org/imbalanced-learn/stable/over_sampling.html

https://www.marcoaltini.com/blog/dealing-with-imbalanced-data-undersampling-oversampling-and-proper-cross-validation

Classification问题，可以从Naive Bayes, Logistic Regression, Decision Tree入手

尝试用Cross Validation提高模型效果

# 先修知识
F1 score: https://en.wikipedia.org/wiki/F1_score
TN / True Negative: case was negative and predicted negative
TP / True Positive: case was positive and predicted positive
FN / False Negative: case was positive but predicted negative
FP / False Positive: case was negative but predicted positive

目标
成功找到欺诈交易，尽量不遗漏。

评价
一般classification问题可能考虑准确率就够了。但是本题由于超过90%的分类都为非欺诈，即使直接“预测”成非欺诈也会有90%以上的正确率。因此precision在本题并不是最重要的指标。应当注意到，银行更关注的是欺诈交易有没有被都筛选出来，也是recall这个指标所表示的含义。因此本题为了兼顾precision和recall，选择F1 score作为模型好坏的衡量指标，公式为

测评配置环境
python

pip install -U scikit-learn
pip install -U numpy
pip install -U pandas
从"数据"那一栏获得代码测试文件。命令行里跑

python3 scorer.py --predicted_file <predicted_file_path> --true_file <true_file_path>
可通过这个script运用Cross Validation验证你的模型在training data上的performance
