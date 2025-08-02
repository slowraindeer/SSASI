# SSASI (Style-Based Annotation Similarity & Structure Inference)

This repository contains Python scripts implementing SASSI, a framework for classifying annotation styles in binary segmentation masks. You can train pairwise classifiers and get a final classificaion for styles A, B, and C  while they are mixed, using the provided scripts.


## Repository Structure
├── AA_AB.py # Train a classifier to distinguish Style A vs. Style B

├── AA_AC.py # Train a classifier to distinguish Style A vs. Style C

├── BB_BC.py # Train a classifier to distinguish Style B vs. Style C

└── ABC.py # Final model: classify masks into Styles A, B, or C

## Features

- **Modular pairwise training**: Build binary classifiers for each style-pair (A–B, A–C, B–C) using `AA_AB.py`, `AA_AC.py`, `BB_BC.py`.  
- **Final multi-class classification**: Use `ABC.py` to combine pairwise outputs into a robust A/B/C classifier.  
- **Complete configuration**: Each script defines the model architecture, data loaders, and **all training hyperparameters** (learning rate, batch size, number of epochs, optimizer settings, etc.).

## Usage

1. Prepare your dataset folder with subdirectories for styles `A/`, `B/`, and `C/`, each applying segmentation masks into original images for training.
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
3. Using AA_AB.py, AA_AC.py, and BB_BC.py, please make the binary classifiers and get thresholds.
4. Prepare your dataset folder with mixed-dataset.
5. Using ABC.py, classify the three styles of segmenation masks, using three classifiers and thresholds.


## Datasets

All datasets used in this paper are publicly available and can be freely downloaded.


## Contact

For any questions about running the code or collaboration, please open an issue or send an email to the author.
