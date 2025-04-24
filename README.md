# Show Attend Tell

## Overview 

The seminal ”Show, Attend and Tell” paper by Xu et al. (2015) introduced a novel approach to image captioning by integrating attention mechanisms with latent representations of images. The original architecture combined a convolutional neural network (CNN) encoder for image feature extraction with a long- short term memory network (LSTM) decoder with soft attention for textual sequence generation. I trained and evaluated a replication of the core architecture described in the original paper in order to establish a baseline reference point. From there, I extend beyond the methodology implemented in the original paper, improving the encoder, decoder, and training architectures to ultimately drive higher performance and signficantly reduce training time in comparison to the original paper. For a more extensive description of the contents and results of this project, please see my paper located within the root of the repository. 

* Note: Both the original and improved models are trained on the COCO dataset and created using PyTorch. Files were rearranged to be more organized and easier to navigate, but may need to be adjusted to run the code.

## Project Structure

```
show_attend_tell/
├── model/
├── model_comparison/
├── new_code/
│   ├── BLEU_Scores.py
│   ├── compare_models.py
│   ├── generate_caption.py
│   ├── model.py
│   └── train.py
├── original_code/
│   ├── attention.py
│   ├── dataset.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── generate_captions.py
│   ├── generate_json_data.py
│   ├── train.py
│   └── utils.py
├── runs/
├── ProjectReport.pdf
├── requirements.txt
└── README.md
```

## Descriptions

### model/

During training, the model weights are saved to this directory.

### model_comparison/

Contains the images, diagrams, and captions used for comparing the performance between the baseline and improved model.

### original_code/

Contains all scripts related to the baseline model. All credit goes to the original authors. This code was used to train the baseline model and served as a reference for the improved model.

### new_code/

#### BLEU_Scores.py 

Script for generating the BLEU diagrams.

#### compare_models.py

Script for querying the model weights and generating the attention diagrams and captions.

#### model.py

Script for improved model.

#### train.py

Script for improved training pipeline.

### ProjectReport.pdf

Contains the written report associated with this project.

### runs/

Contains the tensorboard logs for the training process.


## Sources

[Show Attend Tell](https://arxiv.org/pdf/1502.03044)

[Original Code's Github](https://github.com/AaronCCWong/Show-Attend-and-Tell/tree/master)

## Contact
Isabella Mercado - imercado@uchicago.edu

Project Link: https://github.com/imercadouchicago/show_attend_tell