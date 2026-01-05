# GAF-CNN Network Traffic Classification

A network traffic classification system that combines Gramian Angular Field (GAF) image transformation with Convolutional Neural Networks (CNN) for identifying network traffic patterns.

## Overview

This project implements a novel approach to network traffic classification by:
1. Converting time-series network flow features into 2D GAF images
2. Applying DDPM-based data augmentation with Gaussian noise
3. Classifying the images using a custom CNN architecture (ETNet v2)

## Development Environment

This project is developed using **Spyder IDE** as the primary development environment.

## Requirements

- Python 3.x
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- tensorflow/keras

## Data Requirements

The script expects a CSV file named `02-15-2018.csv` in the root directory containing network flow data with the following columns:
- `Flow Duration`
- `Tot Fwd Pkts`
- `Tot Bwd Pkts`
- `TotLen Fwd Pkts`
- `TotLen Bwd Pkts`
- `Label` (target variable)

## Usage

Run the main script:
```bash
python gaf-cnn-method.py
```

The script will:
1. Load 50,000 rows from the CSV file
2. Process features and encode labels
3. Split data into train/test sets (80/20)
4. Train the CNN model for 10 epochs
5. Display training accuracy/loss plots
6. Show classification report and confusion matrix

## Model Architecture

The ETNet v2 CNN consists of:
- 3 convolutional blocks with batch normalization and dropout
- Dense layers with L2 regularization
- Input: 32x32x1 grayscale GAF images
- Output: Softmax classification layer

## Key Features

- **Dynamic GAF transformation**: Images are generated during batch processing
- **Data augmentation**: DDPM-based noise augmentation applied per epoch
- **Regularization**: Combines L2 penalty, dropout, and batch normalization
- **Batch size**: 16 samples per batch
- **Optimizer**: Adam with learning rate 0.001
