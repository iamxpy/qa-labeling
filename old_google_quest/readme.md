这是组队之前自己写的一些代码，值得以后参考的地方有：

fuse_bert_v1.py: 实现了multi-head self-attention

fuse_bert_v2.py: 为了减少参数量只使用了attention, 实现代码来自fuse_bert_v1.py

mixed_precision.py: 尝试了tf2的半精度训练

multi_bert_swa.py: 使用自定义Callback尝试了SWA，但是由于tf2取出和设置参数都复制一份导致太慢，而且效果不好。自定义Callback的方法值得以后学习