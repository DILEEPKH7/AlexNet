# AlexNet Image Classification

This repository contains a PyTorch implementation of the AlexNet model for image classification. The model is trained to classify whether a person in an image is smiling or not using the CelebA dataset.

## Prerequisites

Make sure you have the following dependencies installed:
- Python 3
- PyTorch
- matplotlib
- pandas

## Dataset

The CelebA dataset is used for training and testing the model. It consists of celebrity images labeled with various attributes. In this project, we focus on the "Smiling" attribute. The dataset is not included in this repository and should be downloaded separately.

1. Download the CelebA dataset from [http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html](http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html).
2. Extract the dataset and place it in the appropriate directory.

## Usage

1. Clone the repository: git clone https://github.com/DILEEPKH7/AlexNet.git
2. Update the file paths in the code to match the location of the dataset on your machine.
3. Run the code: python alexnet.py


## Results

The code trains the AlexNet model on the CelebA dataset to classify smiling and non-smiling faces. After training, it evaluates the model on a test set and displays sample images with their predicted labels.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

This implementation is based on the original AlexNet paper:
- Krizhevsky, A., Sutskever, I., & Hinton, G. E. (2012). ImageNet classification with deep convolutional neural networks. In Advances in neural information processing systems (pp. 1097-1105).

The CelebA dataset is created by:
- Liu, Z., Luo, P., Wang, X., & Tang, X. (2015). Deep learning face attributes in the wild. In Proceedings of the IEEE international conference on computer vision (pp. 3730-3738).

