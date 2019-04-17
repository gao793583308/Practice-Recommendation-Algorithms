# Practice-Recommendation-Algorithms
最近读了很多推荐系统相关的论文，想自己动手实现一下这些算法，于是在网上找了抖音提供的一个数据集，计划实现一下LR,FM,FFM，DeepFm等常见的推荐系统算法。
## 数据集
数据集取自2019年字节跳动短视频内容理解和推荐竞赛
### [数据集下载地址及赛题说明](https://biendata.com/competition/icmechallenge2019/data/)
我们这次使用的时track2的数据集，包括了7万用户的1000多万条交互信息以及面部特征，视频内容特征，标题特征和BGM特征，这些特征都是嵌入向量的形式。我们的目标是根据这些数据来预测一个用户是否会浏览完一个短视频或给这个短视频点赞。 
![image](https://github.com/gao793583308/Practice-Recommendation-Algorithms/blob/master/pic/data.jpeg)
## 已经实现的算法
#### 随机取了1/100的数据作为验证集，bath大小为4096。2000个batch后终止，结果为验证集上的AUC值。
#### [FM(Factorization Machine)](https://github.com/gao793583308/Practice-Recommendation-Algorithms/tree/master/code/FM)  like:0.824 
