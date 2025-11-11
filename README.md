# Transfer Learning with VGG19 for Image Classification

## Author Information
- **Name:** Sanjay Sivaramakrishnan M
- **Registration Number:** 212223240151

## Project Overview

This project implements transfer learning using a pretrained VGG19 model for binary image classification. The model is trained to classify images as either "defect" or "notdefect". The experiment demonstrates how to leverage pretrained models for custom classification tasks with limited training data.

## Features

- **Transfer Learning:** Uses pretrained VGG19 model with frozen feature extractor layers
- **Image Classification:** Binary classification (defect vs notdefect)
- **Data Visualization:** Sample image visualization with denormalization
- **Model Training:** Training with validation loss tracking
- **Evaluation Metrics:** 
  - Test accuracy
  - Confusion matrix visualization
  - Classification report (precision, recall, F1-score)
- **Single Image Prediction:** Predict and visualize predictions on individual images

## Requirements

### Python Packages
- `torch` - PyTorch deep learning framework
- `torchvision` - Pre-trained models and datasets
- `matplotlib` - Plotting and visualization
- `numpy` - Numerical operations
- `scikit-learn` - Metrics (confusion matrix, classification report)
- `seaborn` - Enhanced visualization
- `torchsummary` - Model architecture summary

### Installation
```bash
pip install torch torchvision matplotlib numpy scikit-learn seaborn torchsummary
```

## Dataset Structure

The project expects the following dataset structure:
```
dataset/
├── train/
│   ├── defect/
│   └── notdefect/
└── test/
    ├── defect/
    └── notdefect/
```

- **Training samples:** 172 images
- **Image size:** 224x224 pixels (resized)
- **Classes:** 2 (defect, notdefect)

## Model Architecture

- **Base Model:** VGG19 (pretrained on ImageNet)
- **Transfer Learning Strategy:** 
  - Feature extractor layers are frozen (not trainable)
  - Only the final classifier layer is fine-tuned
- **Input Size:** 3 x 224 x 224 (RGB images)
- **Output:** 2 classes (defect, notdefect)

## Data Preprocessing

- **Image Resize:** 224x224 pixels
- **Normalization:** 
  - Mean: [0.485, 0.456, 0.406]
  - Std: [0.229, 0.224, 0.225]
- **Data Augmentation:** None (only basic transforms)

## Training Configuration

- **Optimizer:** Adam
- **Learning Rate:** 0.001
- **Loss Function:** CrossEntropyLoss
- **Batch Size:** 32
- **Number of Epochs:** 5
- **Device:** CUDA (if available), else CPU

## Results

### Training Performance
- Training loss decreased from 0.4288 to 0.1045 over 5 epochs
- Validation loss decreased from 0.2752 to 0.1551 over 5 epochs

### Test Performance
- **Test Accuracy:** 94.21%

### Classification Report
```
              precision    recall  f1-score   support

      defect       0.96      0.82      0.89        33
   notdefect       0.94      0.99      0.96        88

    accuracy                           0.94       121
   macro avg       0.95      0.90      0.92       121
weighted avg       0.94      0.94      0.94       121
```

## Usage

### Running the Notebook

1. **Setup Dataset:** Ensure your dataset is organized in the `./dataset/` directory with `train/` and `test/` subdirectories.

2. **Run the Notebook:** Execute all cells in `main.ipynb` sequentially.

3. **Training:** The model will train for 5 epochs by default. You can modify the number of epochs in the training function call.

4. **Evaluation:** After training, the model will automatically evaluate on the test set and display:
   - Test accuracy
   - Confusion matrix heatmap
   - Classification report

5. **Prediction:** Use the `predict_image()` function to predict on individual test images.

### Example Prediction
```python
# Predict on a specific test image
predict_image(model, image_index=55, dataset=test_dataset)
```

## Key Functions

### `denormalize(image_tensor)`
Denormalizes image tensors for visualization purposes.

### `show_sample_images(dataset, num_images=5)`
Displays sample images from the dataset with their class labels.

### `train_model(model, train_loader, test_loader, num_epochs=10)`
Trains the model and plots training/validation loss curves.

### `test_model(model, test_loader)`
Evaluates the model on test data and generates:
- Accuracy score
- Confusion matrix visualization
- Classification report

### `predict_image(model, image_index, dataset)`
Predicts the class of a single image and displays it with the predicted and actual labels.

## Project Structure
```
.
├── main.ipynb          # Main notebook with all code
├── README.md           # This file
└── dataset/            # Dataset directory
    ├── train/          # Training images
    └── test/           # Test images
```

## Notes

- The model uses transfer learning, which is effective for small datasets
- Feature extractor layers are frozen to preserve pretrained weights
- Only the final classifier layer is trained, making training faster and requiring less data
- The model achieves high accuracy (94.21%) despite the relatively small training set (172 samples)

## Future Improvements

- Data augmentation to increase dataset size
- Fine-tuning more layers for potentially better performance
- Hyperparameter tuning (learning rate, batch size, optimizer)
- Experimenting with different pretrained models (ResNet, EfficientNet, etc.)
- Cross-validation for more robust evaluation


