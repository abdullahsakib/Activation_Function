# Activation_Function





Some commonly-used activation functions in neural networks are:

## 1. Sigmoid activation function
   
### Key features:

*The sigmoid function has an s-shaped graph.

*This is a non-linear function.

*The sigmoid function converts its input into a probability value between 0 and 1.

*It converts large negative values towards 0 and large positive values towards 1.

*It returns 0.5 for the input 0. The value 0.5 is known as the threshold value which can decide that a given input belongs to what type of two classes.

### Code: 
```python
import numpy as np

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

# Test
x_values = np.linspace(-10, 10, 100)  # create 100 points between -10 and 10
y_values = sigmoid(x_values)

# Plotting
import matplotlib.pyplot as plt

plt.plot(x_values, y_values)
plt.title("Sigmoid Activation Function")
plt.xlabel("Input")
plt.ylabel("Output")
plt.grid(True)
plt.show()

```
![sigmoid](https://github.com/abdullahsakib/Activation_Function/assets/54322794/1968de6d-86ea-4cd6-8cac-23e4fa995838)

### Usage:

*In the early days, the sigmoid function was used as an activation function for the hidden layers in MLPs, CNNs and RNNs.

*The sigmoid function must be used in the output layer when we build a binary classifier in which the output is interpreted as a class label depending on the probability value of input returned by the function.

*The sigmoid function is used when we build a multi label classification model in which each mutually inclusive class has two outcomes. 


#### Drawbacks:

*The sigmoid function has the vanishing gradient problem. This is also known as saturation of the gradients

*The sigmoid function has slow convergence.

*Its outputs are not zero-centered. Therefore, it makes the optimization process harder.

*This function is computationally expensive as an e^z term is included.


## 2. Tanh activation function
 
### Key features:

*The output of the tanh (tangent hyperbolic) function always ranges between -1 and +1.

*Like the sigmoid function, it has an s-shaped graph. This is also a nonlinear function.

*One advantage of using the tanh function over the sigmoid function is that the tanh function is zero centered. This makes the optimization process much easier.

*The tanh function has a steeper gradient than the sigmoid function has.

### Code:

```python
# Importing necessary libraries
import numpy as np
import matplotlib.pyplot as plt

# Define the tanh function
def tanh_function(x):
    return np.tanh(x)

# Generate an array of values from -10 to 10 to represent our x-axis
x = np.linspace(-10, 10, 400)

# Compute tanh values for each x
y = tanh_function(x)

# Plotting
plt.figure(figsize=(8, 6))
plt.plot(x, y, label='tanh(x)', color='blue')
plt.title('Hyperbolic Tangent Function (tanh)')
plt.xlabel('x')
plt.ylabel('tanh(x)')
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

# Setting the x and y axis limits
plt.axhline(y=0, color='black',linewidth=0.5)
plt.axvline(x=0, color='black',linewidth=0.5)
plt.legend()
plt.show()

```
![tanh](https://github.com/abdullahsakib/Activation_Function/assets/54322794/f265044d-65b4-4a9a-8327-edb027c24167)


### Usage:
 
*Until recently, the tanh function was used as an activation function for the hidden layers in MLPs, CNNs and RNNs.


*The tanh function was never used in the output layer.

### Drawbacks:

*The tanh function has the vanishing gradient problem.

*This function is computationally expensive as an e^z term is included. 3. ReLU activation function

## 3. ReLU activation function

### Key features:

*The ReLU (Rectified Linear Unit) activation function is a great alternative to both sigmoid and tanh activation functions.

*This function does not have the vanishing gradient problem.

*This function is computationally inexpensive. It is considered that the convergence of ReLU is 6 times faster than sigmoid and tanh functions.

*If the input value is 0 or greater than 0, the ReLU function outputs the input as it is. If the input is less than 0, the ReLU function outputs the value 0.

*The output of the ReLU function can range from 0 to positive infinity.

*The convergence is faster than sigmoid and tanh functions. This is because the ReLU function has a fixed derivate (slope) for one linear component and a zero derivative for the other linear component.


*Calculations can be performed much faster with ReLU because no exponential terms are included in the function.

```python
import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

# Generate data
x = np.linspace(-10, 10, 400)
y = relu(x)

# Visualization
plt.figure(figsize=(10, 6))
plt.plot(x, y, label='ReLU Function', color='blue')
plt.title('ReLU Activation Function')
plt.xlabel('x')
plt.ylabel('ReLU(x)')
plt.grid()
plt.legend()
plt.show()

```
![relu](https://github.com/abdullahsakib/Activation_Function/assets/54322794/63a10c61-d86c-4d01-9125-ce1b983c8e2c)


### Usage:

*The ReLU function is the default activation function for hidden layers in modern MLP and CNN neural network models.

### Drawbacks:

*The main drawback of using the ReLU function is that it has a dying ReLU problem.

*The value of the positive side can go very high. That may lead to a computational issue during the training.

#### 4. Leaky ReLU activation function



...
    python
    print("i")
...

