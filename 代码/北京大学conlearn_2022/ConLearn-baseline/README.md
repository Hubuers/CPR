复现2022北大论文（CCF-B）
ConLearn: Contextual-knowledge-aware Concept Prerequisite Relation Learning with Graph Neural Network

论文流程：
1.获取概念描述
2.使用预训练bert获取概念描述的嵌入向量（原文使用教育资源微调后的bert）
~~3.（可省略，我感觉没啥意义）运行[降维.py](src%2F%BD%B5%CE%AC.py)（原文把bert的768维通过简单线性变换降到300维）~~
3.运行[DataLoader.py](src%2FDataLoader.py)划分数据集
4.运行[GGNNmodel.py](src%2FGGNNmodel.py)使用GGNN获得更新后的概念向量
5.运行[概念融合.py](src%2F%B8%C5%C4%EE%C8%DA%BA%CF.py)使用多头自注意力更新概念向量
6.运行[SiameseNetwork.py](src%2FSiameseNetwork.py)使用孪生网络预测先决条件关系