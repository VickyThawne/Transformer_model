# A Self-Crafted Transformer-Based text Generation Model

## Overview

Hello everyone! I've crafted this text generator. I built it from scratch.

This project is all about exploring and understanding the world of Transformer architecture. I wanted to create something unique and exciting, and this model does just that!

I've put my heart and soul into this project, and I'm so proud of what I've achieved. my little creation comes to life, generating awesome content that captures readers' attention.

you may check the journey in the jupyter notebook in the src.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Hyperparameters Configuration](#hyperparameters-configuration)
5. [Data Feeder](#data-feeder)
6. [Training](#training)
7. [Pretrained Models](#pretrained-models)
8. [Contributing](#contributing)
9. [References](#references)


## Introduction

Hey there! I'm thrilled to introduce you to The CyberForge, my personal exploration into the world of AI and language processing. This project is all about understanding and building my very own Transformer model, 

This Model utilizes the Transformer architecture to generate texts. It allows customization of hyperparameters and data feeding for better performance. The model is designed to be versatile, with options for fine-tuning and customization to suit various use cases.

## Installation
To use this project, first, clone the repository.

Next, install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage
To use the text Generation Model, follow these steps:

1. Prepare your dataset or use the provided `sherlocks_diary.txt` as a sample input.
2. Set up the hyperparameters in the `train.py`.
3. Run the training script:
    ```bash
    python train.py
    ```

4. The model will start training and save the best model at the end of training.

## Hyperparameters Configuration
The model can be configured parameters given in `train.py`. You can specify the following hyperparameters:

- `device`: The device for training (e.g., 'cuda' or 'cpu').
- `learning_rate`: The learning rate for the optimizer.
- `max_iters`: The maximum number of training iterations.
- `patience`: The patience for early stopping.
- `eval_iters` : the number of iteration when evaluation occurs.

## Data Feeder
The `data/process_data.py` script provides a DataFeeder class that can be used to split and encode the dataset for training. To use it, follow these steps:

1. Prepare your dataset and create a `sherlocks_diary.txt` file or modify the script to read your dataset.
2. Run the script to split and encode the data:

```bash
python data/process_data.py
```

This will create `train.bin` and `val.bin` files with the encoded data.

## Training
The `train.py` script contains the training code for the text Generation Model. It uses the model defined in `model.py`. Before running the script, make sure you have prepared the data and configured the hyperparameters.

## Pretrained Models
Currently, no pretrained models are provided in this repository. You can train the model with your data or use publicly available pretrained Transformer models.
 
## future contribution
- using this model I am going to make my own llm model on sanskrit 
- there are many usecases of the transformer model such as language translation,  

## Contributing
Contributions to this project are welcome! If you find any issues or have improvements to suggest, feel free to create a pull request or open an issue.

## References

To develop this project, I've found inspiration from some amazing sources:

1. **YouTube Video by Andrej Karpathy - "Let's build GPT: from scratch, in code, spelled out."**  
   Watch Here: [Let's build GPT: from scratch, in code, spelled out.](https://www.youtube.com/watch?v=kCc8FmEb1nY)

2. **Attention is All You Need - Original Transformer Paper**  
   Read the Paper: [Attention is All You Need](https://proceedings.neurips.cc/paper_files/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf)