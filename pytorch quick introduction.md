###### tags: `Machine Learning` `python`


Pytorch
===
[If CUDA doesn't work, check whether driver is available](https://www.nvidia.com/Download/index.aspx?lang=en-us)

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
- For grayscale image, convert it into RGB images ```Image.open('PATH').convert('RGB')```
- Another way:
    - ```torchvision.io.ImageReadMode.GRAY``` for converting to grayscale
    - ```torchvision.io.ImageReadMode.RGB``` for converting to RGB with transparency
```python
image = read_image(img_path, mode=ImageReadMode.RGB)
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
[LR scheduling](https://www.kaggle.com/code/isbhargav/guide-to-pytorch-learning-rate-scheduling/notebook)
1. learning rate decay
2. warm up: increase then decrease (also be used in Residual Network, Transformer) -> 在一開始使用Adagrad, RMSprop, Adam 要計算$\sigma$，但一開始樣本數少，較不精準，所以可能需要使用warm up的可能性
    - Residual Network: 0.01 warm up, then go back to 0.1
    - Transformer: $d_{model}^{-0.5} min(step\_num^{-0.5}, step\_num \ warmup\_steps^{-1.5})$

- RMSprop: 希望在相同方向上面，也可以隨著隨gradient大小變化
- Adam = RMSprop + momentum
- RAdam
- [Learning rate scheduler](https://pytorch.org/docs/stable/optim.html)
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

### Save and Load model

[Save/ Load model](https://pytorch.org/docs/master/notes/serialization.html)

### Skorch
[Basic Usage](https://colab.research.google.com/github/skorch-dev/skorch/blob/master/notebooks/Transfer_Learning.ipynb#scrollTo=aLPo7JE7-BvM)


### TENSORBOARD/ TENSORBOARDX
[TORCH.UTILS.TENSORBOARD](https://pytorch.org/docs/stable/tensorboard.html)
[recommend to use TENSORBOARDX]()


### Parallel Distributed 
[](https://pytorch.org/tutorials/beginner/dist_overview.html)
