# Efficient Feedback Network

This repo is the official implementation of Efficient Feedback Networks (under review on CMIG)

## Requirements
Install the libraries below to use the model.
1. PyTorch and Torchvision => https://pytorch.org/ (the latest and GPU version are recommended)
2. Pytorch Image Model => https://pypi.org/project/timm/
3. Transformers => https://huggingface.co/docs/transformers/installation

## Testing
To run the trained model, you can run the following code.
```python
CUDA_VISIBLE_DEVICES=0 python main.py --model=1 --no-train
```
