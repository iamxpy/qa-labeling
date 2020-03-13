### Bert Finetune

为了使BERT适应目标任务，我们需要考虑几个因素：1）第一个因素是长文本的预处理，因为BERT的最大序列长度为512。2）第二个因素是图层选择。 官方的基于BERT的模型由嵌入层，12层编码器和池化层组成。 我们需要为文本分类任务选择最有效的层。3）第三个因素是过拟合问题。 需要具有适当学习率的更好的优化器





层学习率递减



 Further Pre-training

- Within-task pre-training
- In-domain  pre-training
- Cross-domain pre-training



 Multi-Task Fine-Tuning

- All the tasks share the BERT layers and the em-bedding layer.  The only layer that does not shareis the final classification layer



dropout为0.1，base learning rate is 2e-5,  and the warm-up pro-portion is 0.1, Decay factor 0.95



先看看head和tail哪个比较好

如果长度过长：head+tail。

empirically  select  the  first  128 and the last 382 tokens





Custom head for BERT, XLNet and GPT2 and Bucket Sequencing Collator

Auxiliary tasks for models

Custom mimic loss

SWA and checkpoint ensemble

Rank average ensemble of 2x XLNet, 2x BERT and GPT2 medium





Predict from last 4 layers for CLS 

token for whole training datasetReplace model head with 2 layer DNN with BN



第四名：

Negative down sampling during training (remove samples with zeros toxicity labels)



第8名：

Multi-Sample Dropout（没有效果。）