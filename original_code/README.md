# Baseline Implementation

All of the code contained within the original_code directory is sourced from [this repository](https://github.com/AaronCCWong/Show-Attend-and-Tell/tree/master). This code closely follows the architecture of the original paper, but has been modified to work with PyTorch. I utilized this code to understand the architecture of the model, get a baseline for the performance of the model, and as a starting point for my implementation of an extended version of the model. 

# Instructions on how to run the code

## To Train

Download the COCO dataset training and validation
images (From here: [training](http://images.cocodataset.org/zips/train2014.zip) and [validation](http://images.cocodataset.org/zips/val2014.zip) images). Put them in `data/coco/imgs/train2014` and `data/coco/imgs/val2014` respectively. Put the COCO dataset split JSON file from [Deep Visual-Semantic Alignments](https://cs.stanford.edu/people/karpathy/deepimagesent/)
in `data/coco/`. It should be named `dataset.json`.

Run the preprocessing to create the needed JSON files:

```bash
python generate_json_data.py
```

Start the training by running:

```bash
python train.py
```

The models will be saved in `model/` and the training statistics will be saved in `runs/`. To see the training statistics, use:

```bash
tensorboard --logdir runs
```

## To Generate Captions

```bash
python generate_caption.py --img-path <PATH_TO_IMG> --model <PATH_TO_MODEL_PARAMETERS>
```
