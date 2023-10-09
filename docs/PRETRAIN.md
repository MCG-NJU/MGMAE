# Pre-training MGMAE

MGMAE is modified from VideoMAE V2. Please follow [the pre-training instructions of VideoMAE V2](https://github.com/OpenGVLab/VideoMAEv2/blob/master/docs/PRETRAIN.md) to learn how to pre-train the model.

 When we run `run_mgmae_pretraining.py` as the scripts in the [mgmae pre-train scripts folder](/scripts/mgmae), the pre-training will adapt the **motion guided masking**.  MGMAE defines some new custom args and their meaning could be found in `run_mgmae_pretraining.py`.

 To use the RAFT-small extracting optical flows, please download the [raft-small-clean.pth](https://drive.google.com/file/d/1xkui_y3r1R39zCgPd-iZd15SneeeTBBt/view?usp=sharing) and set `--flow_model '/path/of/the/downloaded/raft-small-clean.pth'` in the pre-training scripts. **raft-small-clean.pth** is modified from [princeton-vl/RAFT/raft-small.pth](https://drive.google.com/drive/folders/1sWDsfuZ3Up38EUQt7-JDTT1HcGHuJgvT)
