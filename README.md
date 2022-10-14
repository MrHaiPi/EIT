# Deep Transformers Thirst for Comprehensive-Frequency Data

# Introduction
Current researches indicate that the introduction of inductive bias (IB) can improve the performance of Vision Transformer (ViT). However, they introduce a pyramid structure at the same time to counteract the incremental FLOPs and parameters caused by introducing IB. A structure like this destroys the unification of computer vision and natural language processing (NLP) and is also unsuitable for pixel-level tasks. We study an NLP model called LSRA, which introduces IB under a pyramid-free structure with fewer FLOPs and parameters. We analyze why it outperforms ViT, discovering that introducing IB increases the share of high-frequency data (HFD) in each layer, giving 'attention' more comprehensive information. As a result, the head notices more diverse information, showing increased diversity of the head-attention distances (Head Diversity). However, the Head Diversity indicates the way LSRA introduced IB is inefficient. To further improve the HFD share, increase the Head Diversity, and explore the potential of transformers, we propose EIT. EIT Efficiently introduces IB to ViT with a novel decreasing convolutional structure and a pyramid-free structure. In four small-scale datasets, EIT has an accuracy improvement of 13% on average with fewer parameters and FLOPs than ViT. In the ImageNet-1K, EIT achieves 70%, 78%, 81% and 82% Top-1 accuracy with 3.5M, 8.9M, 16M and 25M parameters, respectively, which are competitive with the representative state-of-the-art (SOTA) methods. In particular, EIT achieves SOTA performance over other models which have a pyramid-free structure. Finally, ablation studies show that EIT does not require position embedding, which offers the possibility of simplified adaptation to more visual tasks without the need to redesign embedding.

# Citation
If you find this work or code is helpful in your research, please cite:
```
@article{xia2022eit,
  title={Deep Transformers Thirst for Comprehensive-Frequency Data},
  author = {{Xia}, Rui and {Xue}, Chao and {Deng}, Boyu and {Wang}, Fang and {Wang}, Jingchao},
  journal={arXiv preprint arXiv:2203.07116},
  year={2022}
}
```

# Model Zoo

We provide baseline EIT models pretrained on ImageNet 2012.

| name | acc@1 | acc@5 | #FLOPs | #params | url |     
| --- | --- | --- | --- | --- | --- | 
| EIT16/4/3-Mini | 70.0 | 89.6 | 1.73G | 3.5M | [model](https://github.com/MrHaiPi/EIT/tree/main/model/eit-16-4-3-mini) |  
| EIT16/4/3-Tiny | 78.1 | 94.0 | 3.84G | 8.9M | [model](https://github.com/MrHaiPi/EIT/tree/main/model/eit-16-4-3-tiny) |  
| EIT16/4/3-Base | 80.6 | 95.3 | 6.52G | 16.0M | [model](https://github.com/MrHaiPi/EIT/tree/main/model/eit-16-4-3-base) |  
| EIT16/4/3-Large | 81.8 | 95.6 | 10.0G | 25.3M | [model](https://github.com/MrHaiPi/EIT/tree/main/model/eit-16-4-3-large) |  


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

# Training
To train EIT on ImageNet on two nodes with 4 gpus for 300 epochs run:
```
# node0, ip="192.168.1.32"
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=0 --master_addr="192.168.1.32" --master_port=12355 train.py
# node1, ip="192.168.1.31"
python -m torch.distributed.launch --nproc_per_node=2 --nnodes=2 --node_rank=1 --master_addr="192.168.1.32" --master_port=12355 train.py
```
