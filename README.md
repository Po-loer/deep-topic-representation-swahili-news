# Deep Topic Representation: Swahili News Articles

This repository implements deep topic representation using autoencoders and transformer embeddings on the Swahili News Articles dataset. It addresses tasks from the specified problem statement, including exploratory data analysis (EDA), text preprocessing, embedding generation, autoencoder training, visualization of latent spaces, experiments with model variations, and an extension to multilingual datasets for assessing language-invariant semantic clusters.

## Project Overview
The goal is to explore and model topics in Swahili news articles using deep learning techniques:
- **Dataset**: Swahili News Articles from Hugging Face (or similar sources like BBC Swahili).
- **Key Tasks**:
  1. Exploratory Data Analysis (EDA): Common terms, word frequency plots, text length distribution.
  2. Preprocessing and Embeddings: Clean text and generate sentence embeddings using BERT or Sentence-BERT.
  3. Modeling: Train a Deep Autoencoder to compress embeddings into a 2D/3D latent space. Analyze reconstruction loss and compression.
  4. Visualization: Use PCA, t-SNE, or UMAP to visualize latent clusters and interpret topic regions.
  5. Experiments: Vary encoder depths, activation functions, and latent dimensions.
  6. Extension: Train a Neural Topic Model (NTM) and compare with autoencoder results.
  7. Further Exploration: Apply dimensionality reduction on multilingual datasets (e.g., Swahili and English translations) to evaluate language-invariant semantic clusters.

The main implementation is in the Jupyter notebook `DeepTopicRepresentationSwahiliNews.ipynb`, which includes code for embeddings, dimensionality reduction, clustering analysis, and distance metrics between Swahili articles and their English translations to assess language invariance.

### Sample Outputs from the Notebook
- **Average Distance Between Translations**: 3.2074 (lower values indicate better language invariance).
- **Cluster Analysis**: Visualizes whether topics (e.g., sports, health) cluster independently of language.
- **Distance by Topic**:
  | Topic          | Distance (Lower = Better Cross-Language Understanding) |
  |----------------|-------------------------------------------------------|
  | Entertainment | 2.3816                                               |
  | Economy       | 4.7853                                               |
  | International | 1.7433                                               |
  | National      | 3.4880                                               |
  | Health        | 3.1911                                               |
  | Sports        | 2.6664                                               |

The notebook demonstrates that clusters are somewhat language-invariant, with topics like "international" showing strong cross-language similarity.

## Requirements
To run the notebook, install the dependencies:
