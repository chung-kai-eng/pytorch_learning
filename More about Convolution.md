[More information about convolution (standford lecture)](https://cs231n.github.io/convolutional-networks/)

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