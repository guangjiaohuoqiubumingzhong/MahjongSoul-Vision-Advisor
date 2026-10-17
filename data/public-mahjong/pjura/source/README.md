---
task_categories:
- image-classification
license: apache-2.0
size_categories:
- 1k<n<10k
---

# Dataset Card for Mahjong Souls Tiles

## Dataset Details

### Dataset Description

This dataset, `pjura/mahjong_souls_tiles`, contains images of Mahjong tiles extracted from the game Mahjong Soul. It was created specifically for the purpose of training computer vision models to recognize Mahjong tiles within the game interface.

The dataset was used to fine-tune the `google/vit-base-patch16-224-in21k` model, resulting in the `pjura/mahjong_vision` model available on Hugging Face.

- **Curated by:** pjura
- **License:** apache-2.0

### Dataset Sources

- **Repository:** [https://huggingface.co/datasets/pjura/mahjong_souls_tiles](https://huggingface.co/datasets/pjura/mahjong_souls_tiles)
- **Associated Model:** [https://huggingface.co/pjura/mahjong_vision](https://huggingface.co/pjura/mahjong_vision)

## Uses

### Direct Use

This dataset is primarily intended for training or fine-tuning image classification models, specifically Vision Transformers (ViT) or similar architectures, for the task of recognizing Mahjong tiles from game screenshots or live feeds.

### Out-of-Scope Use

Using this dataset for tasks unrelated to Mahjong tile recognition might not yield meaningful results. The images are specific to the art style and presentation of the Mahjong Soul game.

## Dataset Structure

The dataset consists of labeled images of individual Mahjong tiles. It is structured as an image folder dataset suitable for use with libraries like `datasets` and `torchvision`.

## Associated Model Performance

The `pjura/mahjong_vision` model, fine-tuned on this dataset, achieved the following results on the evaluation set:

- Loss: 0.0466
- Accuracy: 0.9967
- F1: 0.9966
- Recall: 0.9967

## Associated Model Training Details

### Training Hyperparameters

The `pjura/mahjong_vision` model was trained with the following hyperparameters:

- learning_rate: 5e-05
- train_batch_size: 16
- eval_batch_size: 16
- seed: 42
- gradient_accumulation_steps: 4
- total_train_batch_size: 64
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- lr_scheduler_warmup_ratio: 0.1
- num_epochs: 250

### Framework Versions

- Transformers 4.34.1
- Pytorch 2.1.0+cu121
- Datasets 2.14.6
- Tokenizers 0.14.1

## Citation

```bibtex
@misc{mahjong_souls_tiles_dataset,
  author = {pjura},
  title = {Mahjong Souls Tiles Dataset},
  year = {2023},
  publisher = {Hugging Face},
  journal = {Hugging Face repository},
  howpublished = {\url{https://huggingface.co/datasets/pjura/mahjong_souls_tiles}}
}
```

## Dataset Card Contact

pjura on Hugging Face ([https://huggingface.co/pjura](https://huggingface.co/pjura))