# 7503_Project_Group_6

## Code Structure

- bert.py is the main file that contains the BERT model structure.
- train.py is the file that contains the training loop and evaluation functions.
- main.py is the file that contains the main function to run the training and evaluation.
- bert_moe.py defines the Mixture of Experts (MoE) attention structure, which is imported in to bert.py for attention structure choosing.

## Requirements
- torch, transformers, datasets
- costs 7GB of GPU memory for training

## Running
To run the training and evaluation, use the following command:

```python main.py``` 