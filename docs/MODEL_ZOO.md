# MGMAE Model Zoo

## Pre-train

| Model | Config | Dataset | Encoder Masking | Decoder Masking | Epoch | \#Frame | Ckpt | Log |
| :---: | :----  | :-----: | :-------------: | :-------------: | :---: | :-----: | :--: | :-: |
| ViT-base | [vit_b_k400_mgmae_800e]((/scripts/mgmae/vit_b_k400_mgmae.sh)) | K400 | MG masking (90%) | None | 800 | 16 | [vit_b_k400_mgmae_800e.pth](https://drive.google.com/file/d/10prrPjobDY4ayKpXhiMX8HHpxYqbLNCq/view?usp=sharing) | [vit_b_k400_mgmae_800e_log.txt](https://drive.google.com/file/d/1TqPs1XcApT6Qi9YECxtHx5Qm0JM-_vOS/view?usp=sharing) |
| ViT-base | [vit_b_k400_mgmae_1600e]((/scripts/mgmae/vit_b_k400_mgmae.sh)) | K400 | MG masking (90%) | None | 1600 | 16 | [vit_b_k400_mgmae_1600e.pth](https://drive.google.com/file/d/1EfcsX_vIYWLvm26mUXxJH8CZTbmr0ad9/view?usp=sharing) | [vit_b_k400_mgmae_1600e_log.txt](https://drive.google.com/file/d/1E5Ud4SX5C2xwL_9uPAmZysxYAJItojuk/view?usp=sharing) |
| ViT-base | [vit_b_ssv2_mgmae_800e]((/scripts/mgmae/vit_b_ssv2_mgmae.sh)) | SSV2 | MG masking (90%) | None | 800 | 16 | [vit_b_ssv2_mgmae_800e.pth](https://drive.google.com/file/d/14cEywYN0lMm5_he6bLlY1Zdsahw33dtG/view?usp=sharing) | [vit_b_ssv2_mgmae_800e_log.txt](https://drive.google.com/file/d/10HRqfQvUcwBINTvB1gZ9Ar5pomxocwXN/view?usp=sharing) |
| ViT-base | [vit_b_ssv2_mgmae_2400e]((/scripts/mgmae/vit_b_ssv2_mgmae.sh)) | SSV2 | MG masking (90%) | None | 2400 | 16 | [vit_b_ssv2_mgmae_2400e.pth](https://drive.google.com/file/d/1YR5N0LeqAO-fxYM1xb7kvM_x37yiTqKS/view?usp=sharing) | [vit_b_ssv2_mgmae_2400e_log.txt](https://drive.google.com/file/d/1ekK-cQyaMO3ZjfRO2C3YmQLl3erjiOXv/view?usp=sharing) |

## Fine-tune

| Model | Config | Dataset | Pre-train | Post-pre-train | \#Frame | Top-1 | Top-5 | Ckpt | Log |
| :---: | :----  | :-----: | :-------: | :------------: | :-----: | :---: | :---: | :--: | :-: |
| ViT-base | [vit_b_k400_mgmae_800e_k400_ft](/scripts/finetune/vit_b_k400_ft.sh) | K400 | K400 | None | 16x5x3 | 81.2 | 94.9 | [vit_b_k400_mgmae_800e_k400_ft.pth](https://drive.google.com/file/d/1_Z_085fN6ZM0lw6vn1vbebvyTTmVui0H/view?usp=sharing) | [vit_b_k400_mgmae_800e_k400_ft.pth](https://drive.google.com/file/d/1teswuUOvGgTMoDWp0cgOvH0ZTIuKGQTY/view?usp=sharing) |
| ViT-base | [vit_b_k400_mgmae_1600e_k400_ft](/scripts/finetune/vit_b_k400_ft.sh) | K400 | K400 | None | 16x5x3 | 81.8 | 95.0 | [vit_b_k400_mgmae_1600e_k400_ft.pth](https://drive.google.com/file/d/1VnbM9suJiRpR6d2djpYQeiinYKHUmDKz/view?usp=sharing) | [vit_b_k400_mgmae_1600e_k400_ft.pth](https://drive.google.com/file/d/1jAHz1v4Z3hW5RaAppU9e9LFbE4_1Jac9/view?usp=sharing) |
| ViT-base | [vit_b_ssv2_mgmae_800e_ssv2_ft](/scripts/finetune/vit_b_ssv2_ft.sh) | SSV2 | SSV2 | None | 16x2x3 | 71.0 | 93.1 | [vit_b_ssv2_mgmae_800e_ssv2_ft.pth](https://drive.google.com/file/d/1BGIQGlFPOITBF_0GB8ieoNZZiF7pPSsX/view?usp=sharing) | [vit_b_ssv2_mgmae_800e_ssv2_ft.pth](https://drive.google.com/file/d/1r6uaDnxHBGZmKZFQdfH4c45lphtFsZEA/view?usp=sharing) |
| ViT-base | [vit_b_ssv2_mgmae_2400e_ssv2_ft](/scripts/finetune/vit_b_ssv2_ft.sh) | SSV2 | SSV2 | None | 16x2x3 | 72.3 | 93.5 | [vit_b_ssv2_mgmae_2400e_ssv2_ft.pth](https://drive.google.com/file/d/1TnHDyqPVw84fVzVrpvTiDWT_HU3kNRCo/view?usp=sharing) | [vit_b_ssv2_mgmae_2400e_ssv2_ft.pth](https://drive.google.com/file/d/1L-qGFC7TG_sBtkIZkX3csz-0MulM53Xa/view?usp=sharing)  |

- We report the fine-tuning accuracy for **sparse sampling** on SSv2 and for **dense sampling** on K400.
- \#Frame = #input_frame x #clip x #crop.
- all the input resolution is $224^2$.
