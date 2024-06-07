**README**

## Project Overview

This project involves designing and training a convolutional neural network (CNN) model for the image inpainting task, which involves filling in the missing parts of an image. The model will be applied to a preprocessed version of the CIFAR-100 dataset, containing RGB images resized to 28x28 pixels. The goal is to use an autoencoder network, composed of an encoder and a decoder, to learn how to compress and reconstruct these images.

## Dataset Description

The dataset for this assignment is derived from the CIFAR-100 dataset and includes:

- **Training Set**: Stored in a folder named "train"
- **Test Set**: Stored in a folder named "test"

Each folder contains images with a resolution of 28x28 pixels. The dataset is preprocessed to facilitate the inpainting task.

## Implementation Steps

### Custom PyTorch Dataset Class

1. **Initialization**:
   - Read images from the specified directory (`train` or `test`).
   - Store the images in an array for efficient access.

2. **Transformations**:
   - Resize images from 28x28 to 32x32 using bicubic interpolation with PIL.
   - Convert images to Tensor objects.
   - Normalize tensor values to the range (-1, 1).

3. **`__getitem__` Function**:
   - Return the preloaded images from the array when requested by the DataLoader object.

### Autoencoder Network

The autoencoder network consists of two parts: the encoder and the decoder.

1. **Encoder**:
   - Compresses the input image data.
   - Uses convolutional layers with 2D batch normalization and LeakyReLU activation (slope 0.2).
   - Employs max pooling in each layer to halve the resolution.
   - Convolution layers have a stride size of one.

2. **Decoder**:
   - Regenerates the image from its compressed version.
   - Uses transposed convolution (deconvolution) layers with stride two to increase the resolution back to the original size.
   - Applies the Tanh activation function in the last layer to bound the output pixel values to the range (-1, 1).

### Configuration

- **Kernel and Padding Sizes**: Free to choose based on the design requirements.
- **Activation Functions**: 
  - LeakyReLU with a slope of 0.2 for hidden layers.
  - Tanh for the output layer.
- **Normalization**: 2D batch normalization in each layer.

## Running the Project

1. **Dependencies**:
   - Python 3.10
   - PyTorch
   - PIL (Python Imaging Library)
   - Other standard libraries (numpy, matplotlib, etc.)

2. **Training and Testing**:
   - Implement the custom dataset class to handle data loading.
   - Define and initialize the autoencoder model.
   - Train the model using the training dataset.
   - Evaluate the model on the test dataset.
   - Visualize the reconstructed images to assess the inpainting quality.

3. **Visualization**:
   - Plot original and reconstructed images to compare the effectiveness of the inpainting.

## Contribution and Support

Contributions and feedback are welcome to enhance the functionality and performance of the autoencoder model. For any questions or support, please reach out to baha.yuzlu@gmail.com. Thank you for your interest and support!

---

This README provides a concise overview of the project, including the dataset description, implementation steps, configuration, and running instructions. Adjust the contact information as necessary.
