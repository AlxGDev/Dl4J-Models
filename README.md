# Dl4J-Models
Neural Network Models created with DL4J

# Facial Expression Recognition
The neural network was trained using the fer2013 dataset. Data Augmentation was performed with Histogram Equalization and 10 fold crop and mirror, using the class ImagePreProcessing.

FERModelCreate is used to create the neural network, FERModelTraining is used to train it further. After training it for 10 epochs with a learning rate of 0.01 and 5 more epochs with a training rate of 0.001, the neural network achieved an accuracy of 60% against the validation set.
