# Efficient Feedback Network

This repository is the official implementation of our paper titled 'Efficient Multi-stage Feedback Attention for Diverse Lesion Segmentation in Cancer Imaging,' has been published on the Computerized Medical Imaging and Graphics journal.

## Requirements
Install the libraries below to use the model.
1. PyTorch and Torchvision => https://pytorch.org/ (the latest and GPU version are recommended)
2. Pytorch Image Model => https://pypi.org/project/timm/
3. Transformers => https://huggingface.co/docs/transformers/installation


## Training
Training on specific dataset.
```python
python --dataset_name='Specify you dataset name' main.py --model=1
```

## Testing
To run the trained model, you can run the following code.
```python
python --dataset_name='KVASIR-SEG' main.py --model=1 --no-train
```


## Acknowledgement
We would like to acknowledge the contributions of the following authors to this work:
1. Dewa Made Sri Arsa
2. Talha Ilyas
3. Seok-Hwan Park
4. Leon Chua
5. Hyongsuk Kim

This work is the intellectual property of the Core Research Institute of Intelligent Robots, Jeonbuk National University, Korea.

This work was supported in part by National Research Foundation of Korea (NRF) grant funded by the Korea government (NRF-2019R1A6A1A09031717) and the National Research Foundation of Korea (NRF) grant, funded by the Korean government (MSIT) (RS-2024-00347768).

## Citations
Our published paper can be found in <a href="https://www.sciencedirect.com/science/article/pii/S0895611124000946#d1e7034">HERE</a>.

Please cite our paper if you find our code help your work.
```latex
@article{ARSA2024102417,
title = {Efficient multi-stage feedback attention for diverse lesion in cancer image segmentation},
journal = {Computerized Medical Imaging and Graphics},
volume = {116},
pages = {102417},
year = {2024},
issn = {0895-6111},
doi = {https://doi.org/10.1016/j.compmedimag.2024.102417},
url = {https://www.sciencedirect.com/science/article/pii/S0895611124000946},
author = {Dewa Made Sri Arsa and Talha Ilyas and Seok-Hwan Park and Leon Chua and Hyongsuk Kim},
}
```