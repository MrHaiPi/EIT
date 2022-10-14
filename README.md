# Deep Transformers Thirst for Comprehensive-Frequency Data

# Cited by
```
@ARTICLE{2022arXiv220307116X,
       author = {{Xia}, Rui and {Xue}, Chao and {Deng}, Boyu and {Wang}, Fang and {Wang}, Jingchao}
        title = "{Deep Transformers Thirst for Comprehensive-Frequency Data}",
      journal = {arXiv e-prints},
     keywords = {Computer Science - Computer Vision and Pattern Recognition},
         year = 2022,
        month = mar,
          eid = {arXiv:2203.07116},
        pages = {arXiv:2203.07116},
archivePrefix = {arXiv},
       eprint = {2203.07116},
 primaryClass = {cs.CV},
       adsurl = {https://ui.adsabs.harvard.edu/abs/2022arXiv220307116X},
      adsnote = {Provided by the SAO/NASA Astrophysics Data System}
}
```


# Model Zoo

We provide baseline EIT models pretrained on ImageNet 2012.

| name | acc@1 | acc@5 | #FLOPs | #params | url |     
| --- | --- | --- | --- | --- | --- | 
| EIT16/4/3-Mini | 70.0 | 89.6 | 1.73G | 3.5M | [model](https://github.com/MrHaiPi/EIT/model/eit-16-4-3-mini/best_checkpoint.pth) |  
| EIT16/4/3-Tiny | 78.1 | 94.0 | 3.84G | 8.9M | [model](https://github.com/MrHaiPi/EIT/model/eit-16-4-3-tiny/best_checkpoint.pth) |  
| EIT16/4/3-Base | 80.6 | 95.3 | 6.52G | 16.0M | [model](https://github.com/MrHaiPi/EIT/model/eit-16-4-3-base/best_checkpoint.pth) |  
| EIT16/4/3-Large | 81.8 | 95.6 | 10.0G | 25.3M | [model](https://github.com/MrHaiPi/EIT/model/eit-16-4-3-large/best_checkpoint.pth) |  


# Data preparation

Download and extract ImageNet train and val images from http://image-net.org/.
The directory structure is the standard layout for the torchvision [`datasets.ImageFolder`](https://pytorch.org/docs/stable/torchvision/datasets.html#imagefolder), and the training and validation data is expected to be in the `train/` folder and `val` folder respectively:

```
/path/to/imagenet/
  train/
    class1/
      img1.jpeg
    class2/
      img2.jpeg
  val/
    class1/
      img3.jpeg
    class2/
      img4.jpeg
```
