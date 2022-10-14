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

| name | acc@1 | acc@5 | #params | #FLOPs | url |     
| --- | --- | --- | --- | --- | --- | 
| EIT16/4/3-Mini | 70.0 | 89.6 | 3.5M | [model](https://github.com/MrHaiPi/EIT/model/eit-16-4-3-mini/best_checkpoint.pth) |     


| name | acc@1 | acc@5 | #params | url |
| --- | --- | --- | --- | --- |
| DeiT-tiny | 72.2 | 91.1 | 5M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth) |
| DeiT-small | 79.9 | 95.0 | 22M| [model](https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth) |
| DeiT-base | 81.8 | 95.6 | 86M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth) |
| DeiT-tiny distilled | 74.5 | 91.9 | 6M | [model](https://dl.fbaipublicfiles.com/deit/deit_tiny_distilled_patch16_224-b40b3cf7.pth) |
| DeiT-small distilled | 81.2 | 95.4 | 22M| [model](https://dl.fbaipublicfiles.com/deit/deit_small_distilled_patch16_224-649709d9.pth) |
| DeiT-base distilled | 83.4 | 96.5 | 87M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_224-df68dfff.pth) |
| DeiT-base 384 | 82.9 | 96.2 | 87M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_patch16_384-8de9b5d1.pth) |
| DeiT-base distilled 384 (1000 epochs) | 85.2 | 97.2 | 88M | [model](https://dl.fbaipublicfiles.com/deit/deit_base_distilled_patch16_384-d0272ac0.pth) |
|CaiT-S24 distilled 384| 85.1 | 97.3 | 47M | [model](README_cait.md)|
|CaiT-M48 distilled 448| 86.5 | 97.7 | 356M | [model](README_cait.md)|
