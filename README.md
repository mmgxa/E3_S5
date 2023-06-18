<div align="center">

# Session 5

</div>


## Introduction

This work incorporates DVC into our PyTorch Lightning template named Hydra.

## Installation


To install in dev mode

```
pip install -e .
```

Then, you can view help using

```
dlearn_train --help
```

## Working

First, we create the data splits using the following command inside the `./scripts/` directory.

```
python split_dataset.py  ../data/PetImages

# to delete zero-size files
cd PetImages_split
find . -type f -empty -print -delete
```

Then, we initialize the git and DVC repository using 

```
git init .
dvc init
```

Our `data` folder is already in the `.gitignore` file.


To train your model, execute
```
dlearn_train experiment=cat_dog
```


For testing, checkpoint path is a mandatory argument. The following snippet extracts the latest run and evaluates using `trainer.test`


```
best_ckpt=$(ls -td -- ./outputs/*/* | head -n 1)'/checkpoints/best.ckpt'
dlearn_eval ckpt_path=$best_ckpt experiment=cat_dog
```

For inference, run the code

```
dlearn_infer experiment=cat_dog ckpt_path=$best_ckpt img_path=cat.jpg
dlearn_infer experiment=cat_dog ckpt_path=$best_ckpt img_path=dog.jpg
```

Note that both the checkpoint and the image path are mandatory arguments.


Add all files to git.

```
git add .
```
The add the data folder to DVC

```
dvc add data

# to add data.dvc to git
git add data.dvc
```

To create a local storage for DVC,

```
dvc remote add -d local ~/catsanddogs
dvc push -r local
```

This will print a message `24972 files pushed`