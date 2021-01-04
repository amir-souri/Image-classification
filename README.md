**Image classification**

In this project, I will experiment with different classification methods, including logistic regression, 
k-nearest neigbours, support vector machines and neural networks. The purpose is
to gain a practical understanding of how they compare for a
specific classification task. I will use neural networks through
PyTorch and other algorithms through Scikit-learn. The project
has two parts:

1. Basic models: Set up an environment for experimentation and perform initial tests using logistic regression, support
vector machines, and k-nearest neighbors.
2. Neural networks: Set up a training environment for neural networks using PyTorch and write an MLP (multi layer
perceptron) and a CNN (convolutional neural network). I will compare them to each other and the initial models used.



Overview:
This project revolves around image classification of clothes.
You may already know of the ImageNet object classification competition and 
dataset which has been at the forefront of much of the hype surrounding deep learning today. 
For this project, it would be entirely unrealistic to use ImageNet which contains 14
million images in approximately 20,000 classes. Instead, I use a much simpler dataset that is often used to benchmark classification
algorithms and which still allows one to discover and evaluate the differences between algorithms.
 
Dataset:
For this project I use the FashionMNIST dataset to train and
test the implemented algorithms. FashionMNIST is a database of
clothes article images (from Zalando), consisting of 28 × 28 pixel
grayscale images associated with one of ten classes of clothing articles. 
A total of 60, 000 training samples and 10, 000 test samples are provided.
FashionMNIST is an excellent starting point since it is not too
easy (as you will see) but is still small enough to allow you to train
classifiers within a reasonable timeframe.
To download the dataset, run the downloader.py script.

Framework:
Because I will experiment with many different classification
models, I provide an API for easily dealing with model initialisation, 
training, and evaluation. The abstract class Trainer defines
this API. You can read the details in trainer.py . The two subclasses 
SKLearnTrainer and PyTorchTrainer implement the API for
the Scikit-learn and PyTorch libraries. The actual classifier (Scikit-
learn algorithm or PyTorch module) is passed to the respective class
constructors when instantiating objects. 
For evaluation purposes, I provide a class MetricLogger , which
enables easy logging and calculation of performance metrics.


- Training and evaluating classifiers

The first part of the project is about setting up a base system
for training and evaluating machine learning models. I do the following steps:

1. Implement training and evaluation methods for the class in trainers.py .
2. Implement metric calculations in the MetricLogger SKLearnTrainer class.
3. Create a script for training different models, including a logistic
regression model, support vector machine and k-nearest neighbour model.
4. Evaluate the results (evaluation.ipynb).

The SKLearnTrainer class in trainers.py defines several instance variables for the FashionMNIST 
dataset loaded as Numpy arrays. Specifically, the following instance variables are available:

• ( X_train , y_train): For the training images and labels.
• ( X_val y_train ) : For the validation images and labels.
• ( X_test, y_test ): For the testing images and labels.

Task 1: Model training
I first have to implement the training method as well as a script for training models.


Implement the train() method: 
The method should train the Scikit-learn model self.model to the training data set. The
models provide a common API with model.fit(X, y) being used
for training.


Write training script: Create a script train_sklearn.py. In it,
create an instance of SKLearnTrainer with a LogisticRegression
model. Add calls to the train() and save() methods to train
and save the result.



Task 2: Metrics
I will use three metrics: accuracy, precision, and recall to evaluate
the models. However, since the presented problem contains 10 classes, the matrix for this problem is 10 × 10.



- Neural networks

In this part of the project I will implement a number of
neural network architectures as well as the code for training them.


TensorBoard:
Using a simple print-statements to check the status
when training a neural network is both cumbersome and
less than ideal for understanding how the model is progressing.
In this project, I will instead use TensorBoard which is a
utility for visualising data. Although TensorBoard was created
for TensorFlow (Google’s framework for neural networks), it is
supported by PyTorch as well. You will have to install it though.
Simply use the following command in a terminal after
activating the iaml environment:
pip install --user tensorboard

TensorBoard is started from the terminal, much like Jupyter, by
typing the following command:
tensorboard --logdir=runs

You may change the location of where to read logs from by chang-
ing --logdir but I use runs in this project because it is the
standard logging location for PyTorch. You can then open Tensor-
Board by opening the address localhost:6006 in a browser.


Task 1: Implement PyTorchTrainer

The first task is to implement the train() method for the PyTorchTrainer
class as I did with SKLearnTrainer in the previous project
part. Again, the class loads the datasets in the constructor. This
time, however, PyTorch DataLoader instances are used. The following fields are provided:

• train_data: For the training images and labels.
• val_data: For the validation images and labels.
• test_data: For the testing images and labels.

Additionally, the constructor of PyTorchTrainer takes a number
of extra arguments. Their use is specified in the docstring.


Implement a train(epochs) method:
The method should train the PyTorch nn.Module in self.model for the specified
number of epochs. For each epoch, iterate through the batches in
the training data ( self.train_data ). The optimiser passed to the
constructor is available as self.optimiser .

Log metrics: 
Create two MetricLogger objects in the train() method (with one_hot=false), one for training data and one for
validation data. Log results by using the log(prediction, label) method.

Logging to TensorBoard: 
Use the method self.logger.add_scalar(metric, scalar, step)
to log training results to TensorBoard. metric is a string used to
identify the metric, scalar is the actual value, and step is the
step value. Since scalars are shown as graphs in TensorBoard,
step is effectively the x-axis.


Implement the evaluate() method: 
The method should predict values for the test data self.test_data . Then log the
results using MetricLogger and return the logger object.

Write training script: Create a script train_pytorch.py. In it,
create an instance of PyTorchTrainer and train and save it as I
did for the SKLearnTrainer in part 1. Use the following for the
constructor parameters:


Task 2: Implement modules

I now have a setup that allows simple swapping of PyTorch
modules, optimisers, and preprocessing operations (transforms). In
this task, I will implement both a simple multilayer perceptron
(MLP) as well as a convolutional neural network (CNN).
Using networks.py to implement the modules. The script imports
a number of useful modules for easy access to layers ( torch.nn )
and functions ( torch.nn.functional ) (imported as F ).

Create MLP model: Implement a PyTorch nn.Module which
implements an MLP. The model should have two linear layers nn.Linear and use the ReLU activation function
for the first layer.

Create CNN model: Implement a module which implements a simple CNN. The module should have two convolutional layers
and two linear layers. I use ReLU for activations.
I use a single maxpooling operation
after the convolutional layers have been applied.
