[More information about convolution (standford lecture)](https://cs231n.github.io/convolutional-networks/)

[Conv versus ConvTranspose (the operation procedure)](https://blog.csdn.net/weixin_39228381/article/details/112970097)

## Comparison between two structure
- ```INPUT -> [CONV -> RELU -> POOL]*2 -> FC -> RELU -> FC```: a single convlution layer between every pooling layer
- ```INPUT -> [CONV -> RELU -> CONV -> RELU -> POOL]*3 -> [FC -> RELU]*2 -> FC```: two convolution layers stacked before every pooling layer. Generally, a good idea for larger and deeper networks, because multiple stacked convolution layers can develop more complex features of the input volume before the destrcutive pooling operation


### Comparison (kernel size)
- For small kernel size with multiple layer, the non-linear approximation is more expressive and smaller size of storage size. (e.g. compare 1 layer 5x5 conv versus 2 layers 3x3 conv
- However, in practice, ew might need more memory to hold all the intermediate CONV layer result if we plan to do backpropagation


### Layer Sizing Pattern
- The common rules of thumb of sizing the architectures
    - The **input layer** should be **divisible by 2 many times**  (e.g. CIFAR-10), 64, 96 (e.g. STL-10), or 224 (e.g. common ImageNet ConvNets), 384, and 512.
    - The **conv layer** should be using small filters (e.g. **3x3 or 5x5 at most**), with **```stride=1```** and padding to remain the spatial dimensions of the input
        - **kernel_size=3, padding=1**
        - **kernel_size=5, padding=2**
        - if using biggere kernel_size, it is only common to see on the very first convolution layer that is looking at the image. Compromise based on memory constraints. 
    - The **pooling layer**: commonly use **```kernel_size=2, stride=2```**


### Different Residual Block
- [More exploration about residual net](http://torch.ch/blog/2016/02/04/resnets.html)
![](https://i.imgur.com/unjyMuv.png)
- **putting batch normalization after the addition significantly hurt test error on CIFAR**
- After construct the model, try to alternate different optimizer
- [ResNet architecture](https://arxiv.org/pdf/1512.03385.pdf)
![image](https://user-images.githubusercontent.com/54303314/166627494-14f62129-d37d-4fdd-a1f0-ee0232bcc243.png)


### High level Consideration
- [Going deeper with convolutions](https://arxiv.org/pdf/1409.4842.pdf)
- increase the depth, the number of levels, and model's width (the number of units at each level) to increase the model quality
- Bigger size means a larger number of parameters, more prone to overfitting and more computation resource.
- Inception architecture was especially useful in the context of localization and object detection
- ```Inception modules``` are stacked on top of each other, features of higher abstaction are captured by higher layers, their spatial concentration is expected to decrease suggesting that the ratio of 3x3 and 5x5 convolutions should increase as we move to higher layers
- It seems beneficial to start **using Inception modules only at higher layers while keeping the lower layers in traditional convolutional fashion**



### FLOPs
- A measure of **computation cost** 
    - **1 FLOP = 1 addition + 1 multiplication e.g. wx+b**
- For traditional convolution: # of FLOPs = $O(C_{in} \times k^2 \times C_{out} \times H_{out} \times W_{out}$
    - Model size:   $C_{in} \times k^2 \times C_{out}$
- For depthwise convolution: 
    - Model size:   $C_{in} \times k^2  + C_{in} \times 1 \times 1 \times C_{out}$
    - Step 1: # of filter = # of channel 每個filter對應其中一個filter ( $C_{in} \times k^2$ )
    - Step 2: Use $C_{in} \times 1 \times 1$ cross-channel filter to weighted sum ($C_{out}$) filters ( $C_{in} \times 1 \times 1 \times C_{out}$ ) 

```python=
class DepthwiseConv(nn.Module):
    def __init__(self, num_in, num_out):
        super(DepthwiseConv, self).__init__()
        self.depthwise = nn.Conv2d(num_in, num_in, kernel_size=3, padding=1, groups=num_in)
        self.pointwise = nn.Conv2d(num_in, num_out, kernel_size=1)
       
    def forward(self, x):
        x = self.depthwise(x)
        out = self.pointwise(x)
        return out
```

