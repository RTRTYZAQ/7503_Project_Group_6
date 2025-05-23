# 7503_Project_Group_6

## VIT

### extra_attention

- 在相应的attention.py中，参考__init__实现forward方法（不需要调整__init__）
- 实现时最好考虑下时间复杂度，如果只修改mask，则无法体现优化效果（相比原版attention会更慢）
- 可以参考下我乱写的random_attention.py

### main.ipynb

- 只需设置好set up标题下的参数即可训练
- cifar10和cifar100会自动下载
- Tiny ImageNet需要手动下载，链接：http://cs231n.stanford.edu/tiny-imagenet-200.zip， 需在data文件夹下创建Tiny-ImageNet文件夹
- 统一使用ImageNet21K上训练的VIT-B_16的预训练权重，下载链接：https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz， 放置在pretrian_weights目录下


## BERT

### Code Structure

- bert.py is the main file that contains the BERT model structure.
- train.py is the file that contains the training loop and evaluation functions.
- main.py is the file that contains the main function to run the training and evaluation.
- bert_moe.py defines the Mixture of Experts (MoE) attention structure, which is imported in to bert.py for attention structure choosing.
- preprocessing_datasets.py is the file that contains the preprocessing functions for the datasets.

### Requirements
- torch, transformers, datasets
- costs 7GB of GPU memory for training(setting max_tokens=128 in preprocessing_datasets.py)

### Running
To run the training and evaluation, use the following command:

```python preprocessing_datasets.py```
```python main.py``` 

### Add new attention

- 可以直接在BERT文件夹下添加bert-yourmethod.py，里面的定义方式可以参照bert_moe.py或者bert_random.py
- 保证输入输出和原始bert的attention部分一致即可
- 记得在bert.py中import新的attention方法，并在判断部分加入语句用于选择