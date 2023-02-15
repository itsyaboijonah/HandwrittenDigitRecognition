# Handwritten Digit Recognition

In this project, we designed and implemented a model for performing an image classification task using a modified version of the MNIST dataset.

Each image in the dataset contains three handwritten digits, and the goal was to design and validate a supervised classification model capable of identifying the highest-valued digit appearing in each image.

In order to complete the task, we preprocessed the data through binarization and connected component labelling to remove noise from the images.

We then used data augmentation before training, with random rotations and shears, to allow our model to generalize better on unseen data.

We implemented and tested two approaches: a CNN model created from scratch and the VGG16 model.

The CNN model created from scratch outperformed the VGG16 model and was able to achieve an accuracy of 98% on Kaggle after fine-tuning.
