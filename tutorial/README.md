# Tutorial Notebooks

This directory contains a collection of Jupyter notebooks designed to guide you through various aspects of the framework -- from understanding the core concepts to visualizing results.

## Notebook Overview

###  **Dataset Loading and Introduction**
 [`00_dataset.ipynb`](./00_dataset.ipynb) 
  Provides a basic introduction to the dataset, including how to load the data and visualize some example events.

### **Tokenization Strategy** (technical)
 [`01_tokenization.ipynb`](./01_tokenization.ipynb)

  Walks you visually through the tokenization strategy used in this work, starting with the usual tokenization strategy used in past works, and then moving on to the novel tokenization strategy (C-NMS) used in PoLAr-MAE.

### **Pretraining Strategy** (technical)
 [`02_pretraining.ipynb`](./02_pretraining.ipynb)
 
  Demonstrates how to pretrain a model in detail. Gives an overview of the architecture, as well as the training loop.

### **Pretraining Results**
 [`03_pretraining_results.ipynb`](./03_pretraining_results.ipynb)

  Demonstrates how to load pre-trained models and visualize the results of the pretraining process via (1) plotting group completions and (2) casting the embeddings to RGB space via PCA.

### **Fine-tuning Strategy** (technical)
 [`04_finetuning_training.ipynb`](./04_finetuning_training.ipynb)

  Demonstrates how to fine-tuning for semantic segmentation is done for this model in detail.

### **Fine-tuning Results**
[`05_finetuning_results.ipynb`](./05_finetuning_results.ipynb)

  Demonstrates how to load fine-tuned models and visualize the segmentation results.

## Getting Started

1. Ensure all dependencies are installed as outlined in the main repository’s [README](../README.md).
2. Activate your environment (e.g., using conda):
   ```bash
   conda activate polarmae
   ```
3. Launch Jupyter Notebook or JupyterLab within this directory:
   ```bash
   jupyter notebook
   ```
4. Open the notebook that interests you and follow the in-notebook instructions.

## Additional Resources

For further details on model configurations, experimental setups, and troubleshooting, please refer to the main documentation in the repository. If you have any questions or feedback, feel free to open an issue.

Happy learning!
