# Diode

The dataset used is a portion of the dataset from the DIODE paper (...) the dataset provided consists of 2k RGB images of indoor scenes, and as many files for the reactive mask and depth_map, for a total of 10GB. Indoor scenes are much easier than outdoor scenes as the range of values is smaller.


# Operation

In RGBD images, the red colour indicates that that pixel has a high value, while the blue colour indicates that that pixel has a low value.

## Clipping

We performed a clipping of the image values

![1687259695053](image/README/1687259695053.png)



## Analysis 

and decribe the min/max/mean value distribution [Analysis](FinalCode/_Analysis.ipynb)

![1687259756855](image/README/1687259756855.png)

## Augmentation

The following operations ([Augmentation](FinalCode/_Augmentation.ipynb)) were performed in combination, resulting in a 30% increase in dataset size:

- Horizontal flip
- Inversion of bands
- Saturation
- Gaussian filtering 

![1687259821392](image/README/1687259821392.png)


# Parameter

We define all parameters in [Configuration.yaml](hyp/Config.yaml)


# Models

- Encoder decoder with DenseNet121 like encoder

  ![1687259988510](image/README/1687259988510.png)
- Encoder decoder with DenseNet121 like encoder with skipp connection

  ![1687260044321](image/README/1687260044321.png)
- Plain Models

  ![1687260125734](image/README/1687260125734.png)
- Plain Models with VGG16

  ![1687260224292](image/README/1687260224292.png)
- Plain Models with VGG16 with leakyRelu

  ![1687260204310](image/README/1687260204310.png)


# Conclusion

- Complexity of the task in relation to poor data availability
- Insufficient hardware resources for training more massive models
- Difficulties in reconstructing the depth map
- Improvements through transfer learning in both Encoder-Decoder and plain architecture contexts
