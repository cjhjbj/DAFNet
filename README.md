# DAFNet
---
## Environment
This code has been tested with Python 3.12.7, PyTorch 2.6.0 and torchvision 0.21.0.

## Data Preparation
The content images are from the MS-COCO dataset, and the style images are from the WikiArt dataset.

- MS-COCO dataset: [https://cocodataset.org/](https://cocodataset.org/)
- WikiArt dataset: [https://huggingface.co/datasets/huggan/wikiart](https://huggingface.co/datasets/huggan/wikiart)
- DressCode dataset: [https://github.com/aimagelab/dress-code](https://github.com/aimagelab/dress-code)

Download the datasets and organize them as follows:
- Training content images: `/train2014`
- Training style images: `/style_image`
- Test content images: `datasets/contents/`
- Test style images: `datasets/styles/`

## Training
Download 'vgg_normalised.pth' from [here](https://drive.google.com/file/d/1BinnwM5AmIcVubr16tPTqxMjUCE8iu5M/view)

First, run `python -m visdom.server`

Run the following commands for training:
```bash
python train.py --content_path [path to content training dataset] \
                --style_path [path to style training dataset] \
                --name DAF_test \
                --model DAF \
                --dataset_mode unaligned \
                --no_dropout \
                --load_size 512 \
                --crop_size 256 \
                --image_encoder_path /other/vgg_normalised.pth \
                --gpu_ids 0 \
                --batch_size 2 \
                --n_epochs 2 \
                --n_epochs_decay 3 \
                --display_freq 1 \
                --display_port 8097 \
                --display_env DAF \
                --lambda_style 3 \
                --lambda_content 1 \
                --shallow_layer \
                --lambda_decouple_c 1 \
                --lambda_decouple_s 1
```
## Test
Run the following commands for testing:
```bash
python test.py --content_path datasets/contents \
               --style_path datasets/styles \
               --name DAF_test \
               --model DAF \
               --dataset_mode unaligned \
               --load_size 512 \
               --crop_size 512 \
               --image_encoder_path /other/vgg_normalised.pth \
               --gpu_ids 0 \
               --shallow_layer
```
