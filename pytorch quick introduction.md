###### tags: `Machine Learning` `python`


Pytorch
===
[If CUDA doesn't work, check whether driver is available](https://www.nvidia.com/Download/index.aspx?lang=en-us)
[Multi-Input Deep Neural Networks with PyTorch-Lightning - Combine Image and Tabular Data](https://rosenfelder.ai/multi-input-neural-network-pytorch/)


- ```requires_grad=True```: track computational history and support gradient computation
- ```torch.no_grad()```: stop tracking computation (only to forward computation)
![](https://i.imgur.com/zsCEwKn.png)
```python=
z = torch.matmul(x, w)+b
print(z.requires_grad)

with torch.no_grad():
    z = torch.matmul(x, w)+b
print(z.requires_grad)

[Output]:
    True
    False
```
- ```.backward()```: called, then ```autograd```
    - computes the gradients from each ```grad_fn```
    - accumulates them in the respective tensor's ```.grad``` attribute
    - using the chain rule, propagates all the way to the leaf tensors



### ```DataLoader``` & ```Dataset```
#### Dataset

```python=
import os
import pandas as pd
from torchvision.io import read_image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```
- [Load Pandas Dataframe using Dataset and DataLoader in PyTorch](https://androidkt.com/load-pandas-dataframe-using-dataset-and-dataloader-in-pytorch/)
```python=
class CustomDataset(Dataset):
    def __init__(self, df, transform=None, target_transform=None):
        self.image_df = df['image']
        self.image_labels = df['label'] 
        self.transform = transform
        self.target_transform = target_transform
    
    def __len__(self):
        return df.shape[0]
    
    def __getitem__(self, idx):
        image = self.image_df.loc[idx]
        label = self.image_labels.loc[idx]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label
```


#### DataLoader
- Sampler:
    - SequentialSampler
    - RandomSampler
    - WeightedSampler
    - SubsetRandomSampler

```python=
class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False, sampler=None,
                 batch_sampler=None, num_workers=0, collate_fn=default_collate,
                 pin_memory=False, drop_last=False, timeout=0,
                 worker_init_fn=None)
```

- if you define ```batch_sampler```, then ```batch_size, shuffle, sampler, drop_last``` should be default setting
- if you define ```sampler```, then ```shuffle``` should be set as False
- if ```sampler, batch_sampler``` is None, then
    - if ```shuffle=True```, then ```sampler=RandomSampler(dataset)```
    - if ```shuffle=False```, then ```sampler=SequentialSampler(dataset)```


### The difference between ```.to(device)``` and ```.cuda()```
- ```.to(device)```: 指定CPU or GPU
- ```.cuda()```: 只能指定GPU

### Learning rate scheduling
[Different Optimizer](https://www.youtube.com/watch?v=HYUXEeh3kwY&feature=youtu.be)
[LR scheduling overview](https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook)
[Chain Scheduler](https://github.com/sooftware/pytorch-lr-scheduler): 當今天需要使用warm-up + decay機制時，可能需要set up 不同的scheduler
[```Attention is all you need```: use warm-up and decay](https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer)
> 上面寫法有錯請參考 [warm-up+decay](https://stackoverflow.com/questions/65343377/adam-optimizer-with-warmup-on-pytorch): also need to define a function to update the state_dict

**[參考該作法Customized learning rate scheduler](https://github.com/sooftware/pytorch-lr-scheduler)**
1. learning rate decay
2. warm up: increase then decrease (also be used in Residual Network, Transformer) -> 在一開始使用Adagrad, RMSprop, Adam 要計算$\sigma$，但一開始樣本數少，較不精準，所以可能需要使用warm up的可能性
    - Residual Network: 0.01 warm up, then go back to 0.1
    - Transformer: $d_{model}^{-0.5} min(step\_num^{-0.5}, step\_num \ warmup\_steps^{-1.5})$

- RMSprop: 希望在相同方向上面，也可以隨著隨gradient大小變化
- Adam = RMSprop + momentum
- AdamW = Adam + weight
- RAdam
- [Pytorch official: Learning rate scheduler](https://pytorch.org/docs/stable/optim.html)
```python=
scheduler = ReduceLROnPlateau(monitor='val_loss', factor=0.5,
                              patience=5, mode='min', verbose=1,
                              min_lr=1e-4)
```

```python=
scheduler = ...
for epoch in range(100):
    train(...)
    validate(...)
    scheduler.step()
```

:::info
### Discussion between different optimizer and the way to use learning scheduler
- [**為什麼Adam常常打不過SGD？**](https://medium.com/ai-blog-tw/deep-learning-%E7%82%BA%E4%BB%80%E9%BA%BCadam%E5%B8%B8%E5%B8%B8%E6%89%93%E4%B8%8D%E9%81%8Esgd-%E7%99%A5%E7%B5%90%E9%BB%9E%E8%88%87%E6%94%B9%E5%96%84%E6%96%B9%E6%A1%88-fd514176f805)
![](https://i.imgur.com/JryjAES.png)
- 很多篇paper討論```Adam```在testing時error較大。```Adam```快速收斂、調參容易，但收斂問題與泛化能力一直不如SGDM的結果。原因可能為 **Weight decay** 或**訓練後期不穩定的optimization step**。
    - 訓練後期大部分gradient都會很小，可能上千萬個batch中只會有幾個batch會有比較大的gradient，但依照Adam的算法，這些大的gradient影響力卻不如那些大量的小gradient
![](https://i.imgur.com/O7fL4De.png)
- 在使用```Adam```時，使用 Learning rate scheduling (簡單的warm-up加上decay)
    - warm-up: 不要一開始就使用高的learning rate，應從低慢慢升到Base learning rate (減少一開始因樣本數不足產生的訓練誤差)
    - decay: 隨optimization步數增加，降低learning rate
![](https://i.imgur.com/1hXODz5.png)
- 小結: 當一開始訓練一個模型時，還是推薦可以先使用Adam、AdamW配上learning rate warm-up & decay，較簡單好實作，不用調太多參數。
:::

### Save and Load model

[Save/ Load model](https://pytorch.org/docs/master/notes/serialization.html)

### Skorch
[Basic Usage](https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Transfer_Learning.ipynb#scrollTo=aLPo7JE7-BvM)


### TENSORBOARD/ TENSORBOARDX
[TORCH.UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)
[recommend to use TENSORBOARDX]()


### Parallel Distributed 
[tutorial](https://pytorch.org/tutorials/beginner/dist_overview.html)



## Multi-input output




```python=
# model
class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.features1 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features2 = nn.Sequential(
            nn.Conv2d(1, 3, 3, 1, 1),
            nn.MaxPool2d(2),
            nn.ReLU(),
        )
        self.features3 = nn.Sequential(
            nn.Linear(10, 5),
            nn.ReLU(),
        )

        self.classifier = nn.Linear(128*128*3 + 32*32*3 + 5, 4)
        
    def forward(self, x1, x2, x3):
        x1 = self.features1(x1)
        x2 = self.features2(x2)
        x3 = self.features3(x3)

        x1 = x1.view(x1.size(0), -1)
        x2 = x2.view(x2.size(0), -1)
        x3 = x3.view(x3.size(0), -1)
        
        x = torch.cat((x1, x2, x3), dim=1)
        x = self.classifier(x)
        return x

model = MyModel()
batch_size = 1
x1 = torch.randn(batch_size, 1, 256, 256)
x2 = torch.randn(batch_size, 1, 64, 64)
x3 = torch.randn(batch_size, 10)

output = model(x1, x2, x3)
```