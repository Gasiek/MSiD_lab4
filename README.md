#Fashion-MNIST
##Introduction
The clue of the task is to implement a model that allows classification of thumbnails of photos representing clothes
from Fashion-MNIST. Fashion-MNIST is a dataset of Zalando's article images-consisting of a training set of 60,000
examples and a test set of 10,000 examples. Each example is a 28x28 grayscale image, associated with a label from 10
classes. We'll be covering three classifiers: KNN and MLP.

Here we have an example of how the data looks:

![Image](https://github.com/zalandoresearch/fashion-mnist/raw/master/doc/img/fashion-mnist-sprite.png)

##Methods
First classifier we'll be covering is KNN (k-nearest neighbours). The Distance Metric I decided to use is Manhattan
Distance. The KNN algorithm assumes that similar things exist in close proximity. K parameter represents number of
neighbours we take into consideration. On my computer it takes around 30 minutes to calculate Manhattan distance matrix.

Next, we have MLP classifier (Multi-layer Perceptron), which is one of the simplest Neural Network classifier. 
Multi-layer Perceptron classifier consists of:
* input layer
* hidden layers
* output layer

Most of the magic happens in hidden layers. In my algorithm i use:
* Activation function I use in hidden layers is the ReLU(Rectified Linear Units) function, because it's cheap to 
compute as there is no complicated math and hence easier to optimize.
* Dropout, which is  a computationally cheap way to regularize our neural network. It's a technique where randomly 
selected neurons are ignored during training.
* Dense layer, which is fully connected with previous layer.
* Kernel regularizer to apply a penalty on the layer's kernel.

On output layer we use Softmax function because our output is multidimensional.


##Results
 Classifier | my Accuracy | Accuracy on benchmark
------------ | ------------- | -----      
KNN{k = 7} | 86,28% | 86%
MLP | 89,5% | 87,7%

##Usage
To run our classifier, we'll need to download Fashion-MNIST data from 
[here](https://github.com/zalandoresearch/fashion-mnist#get-the-data)
and put it in 'fashion' directory in the project. We need to have Tensorflow and Numpy installed. After these steps,
we can run our classifiers by running: 'classifierMLP.py', 'classifierKNN.py'. To use best fitted MLP classifier,
uncomment 57th line and comment 58th.