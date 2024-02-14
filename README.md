# Activation_Function
Different types of activation functions

Some commonly-used activation functions in neural networks are:
1. Sigmoid activation function
   
Key features:

*The sigmoid function has an s-shaped graph.

*This is a non-linear function.

*The sigmoid function converts its input into a probability value between 0 and 1.

*It converts large negative values towards 0 and large positive values towards 1.

*It returns 0.5 for the input 0. The value 0.5 is known as the threshold value which can decide that a given input belongs to what type of two classes.

![image](https://github.com/abdullahsakib/Activation_Function/assets/54322794/8625852d-d3a0-4876-9579-e8d198b6ea13)

Usage:

*In the early days, the sigmoid function was used as an activation function for the hidden layers in MLPs, CNNs and RNNs.

*The sigmoid function must be used in the output layer when we build a binary classifier in which the output is interpreted as a class label depending on the probability value of input returned by the function.

*The sigmoid function is used when we build a multi label classification model in which each mutually inclusive class has two outcomes. 


Drawbacks:

*The sigmoid function has the vanishing gradient problem. This is also known as saturation of the gradients

*The sigmoid function has slow convergence.

*Its outputs are not zero-centered. Therefore, it makes the optimization process harder.

*This function is computationally expensive as an e^z term is included.


2. Tanh activation function
 
Key features:

*The output of the tanh (tangent hyperbolic) function always ranges between -1 and +1.

*Like the sigmoid function, it has an s-shaped graph. This is also a nonlinear function.

*One advantage of using the tanh function over the sigmoid function is that the tanh function is zero centered. This makes the optimization process much easier.

*The tanh function has a steeper gradient than the sigmoid function has.

Usage:
 
Until recently, the tanh function was used as an activation function for the hidden layers in MLPs, CNNs and RNNs.
However, the tanh function is still used in RNNs.
Currently, we do not usually use the tanh function for the hidden layers in MLPs and CNNs. Instead, we use ReLU or Leaky ReLU there.
We never use the tanh function in the output layer.
Drawbacks:
We do not usually use the tanh function in the hidden layers because of the following drawback.
The tanh function has the vanishing gradient problem.
This function is computationally expensive as an e^z term is included. 3. ReLU activation function




