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


