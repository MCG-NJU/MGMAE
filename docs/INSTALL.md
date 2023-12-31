# VideoMAEv2 Installation

The required packages are in the file `requirements.txt`, and you can run the following command to install the environment

```
conda create --name mgmae python=3.8 -y
conda activate mgmae

conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 -c pytorch

pip install -r requirements.txt
```

### Note:
- [For MGMAE] If it failed to install `cupy`, you can specify the CUDA version. For example, for `CUDA=11.0`, run the command `pip install cupy-cuda110`.
- **The above commands are for reference only**, please configure your own environment according to your needs.
- We recommend installing **`PyTorch >= 1.12.0`**, which may greatly reduce the GPU memory usage.
- It is recommended to install **`timm == 0.4.12`**, because some of the APIs we use are deprecated in the latest version of timm.
- We have supported pre-training with `PyTorch 2.0`, but it has not been fully tested.
