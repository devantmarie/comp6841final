# Exploring and Enhancing Non-Coding RNA Family Identification and Annotation Using Deep Learning on Primary Structure

This project implements a deep learning pipeline for classifying RNA sequences, with a focus on non-coding RNAs (ncRNAs). The pipeline handles multiple aspects of the machine learning workflow, including dataset preparation, model training, evaluation, and testing, all tailored to bioinformatics applications. 

The system supports various transformations, such as reverse complement augmentations, and integrates seamlessly with deep learning models for classifying biological sequences.

## Key Features
- **Flexible Dataset Handling**: Load multiple RNA sequence datasets from FASTA files, with support for combining data from multiple sources.
- **Data Augmentation**: Optionally apply reverse complement transformations to the sequences as part of the data augmentation process.
- **Customizable Parameters**: Configure various parameters like sequence length, batch size, learning rate, and data splits for training, validation, and testing.
- **Model Training & Evaluation**: Implements training, validation, and test workflows with customizable deep learning models.
- **Model Checkpointing**: Optionally save model checkpoints during training to resume or further evaluate the model.
- **Visualization**: Generate plots to track model performance during training, including loss curves and per-class accuracy.

## Project Structure

- **Dataset Class**: `NcRnaDataset` class for managing RNA sequence data, supporting sequence encoding and data transformations.
- **Training & Evaluation Class**: `ImplementDLEv` class for handling model training, evaluation, and testing.
- **Helper Functions**: Utility functions for sequence encoding (e.g., one-hot encoding) and performance plotting.

## Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/devantmarie/comp6841final.git
cd comp6841final
pip install biopython
pip install torch torchvision torchaudio
```
## Usage

For detailed instructions and code examples, check the notebook `Project_emilio_vello.ipynb`.

## Dataset Source

For this project, we use a subset of the Rfam database — one of the most comprehensive collections of manually curated non-coding RNA (ncRNA) sequences. This particular subset was selected by Fiannaca et al. (2017) and is available in FASTA format.

Each sequence in the dataset is labeled with its ncRNA class directly in the FASTA header (after the > symbol), and preprocessing is applied to extract the sequence and its corresponding label.

Three FASTA files are used:

Train_set.fasta – 6,320 ncRNA sequences

Validation_set.fasta – 2,600 ncRNA sequences

Test_set.fasta – 2,400 ncRNA sequences

Rather than keeping the original split, this project performs a random redistribution of the sequences into training, validation, and test sets. The proportions are configurable, allowing flexibility for different training scenarios.

Source and original files:
https://github.com/IcarPA-TBlab/nrc/tree/master?tab=readme-ov-file

