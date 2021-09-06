# CIFAR-10 (Training from scratch)
# Model 
Wide ResNet (depth = 28, wide = 10)
```` 
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1           [-1, 16, 32, 32]             448
       BatchNorm2d-2           [-1, 16, 32, 32]              32
            Conv2d-3          [-1, 160, 32, 32]          23,200
           Dropout-4          [-1, 160, 32, 32]               0
       BatchNorm2d-5          [-1, 160, 32, 32]             320
            Conv2d-6          [-1, 160, 32, 32]         230,560
            Conv2d-7          [-1, 160, 32, 32]           2,720
        wide_basic-8          [-1, 160, 32, 32]               0
       BatchNorm2d-9          [-1, 160, 32, 32]             320
           Conv2d-10          [-1, 160, 32, 32]         230,560
          Dropout-11          [-1, 160, 32, 32]               0
      BatchNorm2d-12          [-1, 160, 32, 32]             320
           Conv2d-13          [-1, 160, 32, 32]         230,560
       wide_basic-14          [-1, 160, 32, 32]               0
      BatchNorm2d-15          [-1, 160, 32, 32]             320
           Conv2d-16          [-1, 160, 32, 32]         230,560
          Dropout-17          [-1, 160, 32, 32]               0
      BatchNorm2d-18          [-1, 160, 32, 32]             320
           Conv2d-19          [-1, 160, 32, 32]         230,560
       wide_basic-20          [-1, 160, 32, 32]               0
      BatchNorm2d-21          [-1, 160, 32, 32]             320
           Conv2d-22          [-1, 160, 32, 32]         230,560
          Dropout-23          [-1, 160, 32, 32]               0
      BatchNorm2d-24          [-1, 160, 32, 32]             320
           Conv2d-25          [-1, 160, 32, 32]         230,560
       wide_basic-26          [-1, 160, 32, 32]               0
      BatchNorm2d-27          [-1, 160, 32, 32]             320
           Conv2d-28          [-1, 320, 32, 32]         461,120
          Dropout-29          [-1, 320, 32, 32]               0
      BatchNorm2d-30          [-1, 320, 32, 32]             640
           Conv2d-31          [-1, 320, 16, 16]         921,920
           Conv2d-32          [-1, 320, 16, 16]          51,520
       wide_basic-33          [-1, 320, 16, 16]               0
      BatchNorm2d-34          [-1, 320, 16, 16]             640
           Conv2d-35          [-1, 320, 16, 16]         921,920
          Dropout-36          [-1, 320, 16, 16]               0
      BatchNorm2d-37          [-1, 320, 16, 16]             640
           Conv2d-38          [-1, 320, 16, 16]         921,920
       wide_basic-39          [-1, 320, 16, 16]               0
      BatchNorm2d-40          [-1, 320, 16, 16]             640
           Conv2d-41          [-1, 320, 16, 16]         921,920
          Dropout-42          [-1, 320, 16, 16]               0
      BatchNorm2d-43          [-1, 320, 16, 16]             640
           Conv2d-44          [-1, 320, 16, 16]         921,920
       wide_basic-45          [-1, 320, 16, 16]               0
      BatchNorm2d-46          [-1, 320, 16, 16]             640
           Conv2d-47          [-1, 320, 16, 16]         921,920
          Dropout-48          [-1, 320, 16, 16]               0
      BatchNorm2d-49          [-1, 320, 16, 16]             640
           Conv2d-50          [-1, 320, 16, 16]         921,920
       wide_basic-51          [-1, 320, 16, 16]               0
      BatchNorm2d-52          [-1, 320, 16, 16]             640
           Conv2d-53          [-1, 640, 16, 16]       1,843,840
          Dropout-54          [-1, 640, 16, 16]               0
      BatchNorm2d-55          [-1, 640, 16, 16]           1,280
           Conv2d-56            [-1, 640, 8, 8]       3,687,040
           Conv2d-57            [-1, 640, 8, 8]         205,440
       wide_basic-58            [-1, 640, 8, 8]               0
      BatchNorm2d-59            [-1, 640, 8, 8]           1,280
           Conv2d-60            [-1, 640, 8, 8]       3,687,040
          Dropout-61            [-1, 640, 8, 8]               0
      BatchNorm2d-62            [-1, 640, 8, 8]           1,280
           Conv2d-63            [-1, 640, 8, 8]       3,687,040
       wide_basic-64            [-1, 640, 8, 8]               0
      BatchNorm2d-65            [-1, 640, 8, 8]           1,280
           Conv2d-66            [-1, 640, 8, 8]       3,687,040
          Dropout-67            [-1, 640, 8, 8]               0
      BatchNorm2d-68            [-1, 640, 8, 8]           1,280
           Conv2d-69            [-1, 640, 8, 8]       3,687,040
       wide_basic-70            [-1, 640, 8, 8]               0
      BatchNorm2d-71            [-1, 640, 8, 8]           1,280
           Conv2d-72            [-1, 640, 8, 8]       3,687,040
          Dropout-73            [-1, 640, 8, 8]               0
      BatchNorm2d-74            [-1, 640, 8, 8]           1,280
           Conv2d-75            [-1, 640, 8, 8]       3,687,040
       wide_basic-76            [-1, 640, 8, 8]               0
      BatchNorm2d-77            [-1, 640, 8, 8]           1,280
           Linear-78                   [-1, 10]           6,410
      Wide_ResNet-79                   [-1, 10]               0
================================================================
Total params: 36,489,290
Trainable params: 36,489,290
Non-trainable params: 0
----------------------------------------------------------------
Input size (MB): 0.01
Forward/backward pass size (MB): 63.38
Params size (MB): 139.20
Estimated Total Size (MB): 202.58
----------------------------------------------------------------
````

# Training
- Resize, RandomCrop, Rotation, Flipping  
- Label Smoothing (alpha = 0.2)  
- Mix Up Augmentation (alpha = 1.0)  
- Cut Mix Augmentation (alpha = 1.0)  
- Cut Out Augmentation (rate >= 0.80)  
- Optimizer: Stochatic Gradient Descent (initial lr = 0.1)  
- Scheduler : Cyclic learning rate and Cosine learning rate (Tmax = 100)  
- Progressing Resize (32 -> 36 -> 40)  
- Test Time Augmentation
# Data
CIFAR-10 datasets 
# Result
````
Epoch [230/230] Iter[100/100]  
Loss on testing data: 0.3858  
Accuracy on testing data: 96.6800%  
````
**Confusion Matrix**  
![image](https://user-images.githubusercontent.com/74145100/132237407-64455ffe-87e2-4644-999d-503c59998969.png)

Checkpoint weights: https://drive.google.com/file/d/1UQkhsb-OAWa7_K7tbSgyH6wDa4E67pho/view?usp=sharing

