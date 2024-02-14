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

![image](https://github.com/abdullahsakib/Activation_Function/assets/54322794/23081c7d-f0b0-4a12-ad47-2476474c2fa2)


Usage:
 
*Until recently, the tanh function was used as an activation function for the hidden layers in MLPs, CNNs and RNNs.


*The tanh function was never used in the output layer.

Drawbacks:

*The tanh function has the vanishing gradient problem.

*This function is computationally expensive as an e^z term is included. 3. ReLU activation function

3. ReLU activation function

Key features:

*The ReLU (Rectified Linear Unit) activation function is a great alternative to both sigmoid and tanh activation functions.

*This function does not have the vanishing gradient problem.

*This function is computationally inexpensive. It is considered that the convergence of ReLU is 6 times faster than sigmoid and tanh functions.

*If the input value is 0 or greater than 0, the ReLU function outputs the input as it is. If the input is less than 0, the ReLU function outputs the value 0.

*The output of the ReLU function can range from 0 to positive infinity.

*The convergence is faster than sigmoid and tanh functions. This is because the ReLU function has a fixed derivate (slope) for one linear component and a zero derivative for the other linear component.


*Calculations can be performed much faster with ReLU because no exponential terms are included in the function.

![image](https://github.com/abdullahsakib/Activation_Function/assets/54322794/5198a64e-8ebd-4fcd-a6b0-7c56875a8f6f)


Usage:

*The ReLU function is the default activation function for hidden layers in modern MLP and CNN neural network models.

Drawbacks:

*The main drawback of using the ReLU function is that it has a dying ReLU problem.

*The value of the positive side can go very high. That may lead to a computational issue during the training.


