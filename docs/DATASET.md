# Data Preparation

MGMAE is modified from VideoMAE V2. The fine-tuning annotation files could be found in Google Drive.

| dataset  | data type | train videos | validation videos | data list file |
| :------: | :-------: | :----------: | :---------------: | :------------: |
| k400 | video | 240436 | 19796 | [k400_list.zip](https://drive.google.com/file/d/11US3KptpqHsZ5K4wQLzs-OA3Y50OWtPJ/view?usp=sharing) |
| ssv2 | rawframes | 168913 | 24777 | [sthv2_list.zip](https://drive.google.com/file/d/1OtQzj1S0HjgUciB7cZa4MCDHXQ20FpZg/view?usp=sharing) |

## Pre-train Dataset

The pretrain dataset loads the data list file, and then process each line in the list. The pre-training data list file is in the following format:

for video data line:
> video_path 0 -1

for rawframes data line:
> frame_folder_path start_index total_frames

For example, the k400 data list file:

```
# The path prefix 'your_path' can be specified by `--data_root ${PATH_PREFIX}` in scripts when training or inferencing.

your_path/k400/---QUuC4vJs.mp4 0 -1
your_path/k400/--VnA3ztuZg.mp4 0 -1
...
```

where the AVA and Something-Something data are rawframes and the rest are videos.

## Fine-tune Dataset

There are two implementations of our finetune dataset `VideoClsDataset` and `RawFrameClsDataset`, supporting video data and rawframes data, respectively. Where SSV2 uses `RawFrameClsDataset` by default and the rest of the datasets use `VideoClsDataset`.

`VideoClsDataset` loads a data list file with the following format:
> video_path label

while `RawFrameClsDataset` loads a data list file with the following format:
> frame_folder_path total_frames label

For example, video data list and rawframes data list are shown below:

```
# The path prefix 'your_path' can be specified by `--data_root ${PATH_PREFIX}` in scripts when training or inferencing.

# k400 video data validation list
your_path/k400/jf7RDuUTrsQ.mp4 325
your_path/k400/JTlatknwOrY.mp4 233
your_path/k400/NUG7kwJ-614.mp4 103
your_path/k400/y9r115bgfNk.mp4 320
your_path/k400/ZnIDviwA8CE.mp4 244
...

# ssv2 rawframes data validation list
your_path/SomethingV2/frames/74225 62 140
your_path/SomethingV2/frames/116154 51 127
your_path/SomethingV2/frames/198186 47 173
your_path/SomethingV2/frames/137878 29 99
your_path/SomethingV2/frames/151151 31 166
...
```
