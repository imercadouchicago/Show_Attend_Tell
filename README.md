# Show Attend Tell

This repository implements and improves the Show Attend Tell model for image captioning.

## Repository Structure

show_attend_tell/
├── model/
├── model_comparison/
├── runs/
├── original_code/
│   ├── attention.py
│   ├── dataset.py
│   ├── decoder.py
│   ├── encoder.py
│   ├── generate_captions.py
│   ├── generate_json_data.py
│   ├── train.py
│   └── utils.py
├── BLEU_Scores.py
├── compare_models.py
├── FinalReport.tex
├── generate_caption.py
├── model.py
├── requirements.txt
├── generate_caption.py
├── train.py
└── README.md

# Descriptions

## model/

Where all the weights are saved to during training.

## model_comparison/

Contains the images, diagrams, and captions for comparing the performance between the baseline model and the improved model.

## runs/

Contains the tensorboard logs for the training process.

## original_code/

Contains the code for the baseline model. All credit goes to the original authors. This code was used to train the baseline model and served as a reference for the improved model.

## BLEU_Scores.py

Contains the code for generating the BLEU diagrams.

## compare_models.py

Contains the code for querying the models and generating the attention diagrams and captions.

## FinalReport.tex

Contains the final report for the project.

## model.py

Contains the code for the improved model.


## train.py

Contains the improved training pipeline.


# Sources

[Show Attend Tell](https://arxiv.org/pdf/1502.03044)

[Original Code's Github](https://github.com/AaronCCWong/Show-Attend-and-Tell/tree/master)