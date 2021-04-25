import numpy as np
from numpy import random
import matplotlib.pyplot as plt

#loading input file
print("\n")
user_input = input("Enter your input filename with extension (ex: input.txt): ")
file_input = np.loadtxt(user_input)

#defining input data
print("\n")
column_x = input("The data to predict (input x) are in column (ex: 1 or 2 or etc.): ")
column_y = input("The data to be predicted (input y) are in column (ex: 1 or 2 or etc.): ")
x = np.array(file_input[:, (int(column_x) - 1)])
y = np.array(file_input[:, (int(column_y) - 1)])
label_x = input("The label for input x: ")
label_y = input("The label for input y: ")
print("\n")
print("\n", label_x, "\n", x)
print("\n", label_y, "\n", y)
print("\n", "Length of ", label_x, ": ", len(x))
print("\n", "Length of ", label_y, ": ", len(y))
print("\n")

print("\n")
#defining initial parameter
slope = input("Input the initial guess of the slope (suggestion: 1): ")
intercept = input("Input the initial guess of the intercept (suggestion: 0): ")
learning_rate = input("Input the learning rate (alpha) (suggestion: 0.01): ")
minimum_gradient = input("Input the minimum gradient (suggestion: 0.0009): ")
maximum_iterations = input("Input the maximum number of iterations (suggestion: 10000): ")

#converting input to float
slope = float(slope)
intercept = float(intercept)
learning_rate = float(learning_rate)
minimum_gradient = float(minimum_gradient)
maximum_iterations = float(maximum_iterations)

#initial predicted line
model_x = np.arange((min(x) - (min(x) / 10)), (max(x) + (max(x) / 10)), (len(x) / 100))
model_y_initial = (slope * model_x) + intercept

eq_initial = "Initial Line: Y = (" + str(np.around(slope, 3)) + ")" + "X" + " + " + "(" + str(np.around(intercept, 3)) + ")"

#gradient descent function, using loss function = sum square residual
def gradient_descent(x, y, slope, intercept, learning_rate, minimum_gradient, maximum_iterations) :
    iterations = 0
    #looping
    looping = "continue"

    tot_gradient_slope = []
    tot_gradient_intercept = []
    tot_slope = []
    tot_intercept = []
    tot_mse = []

    while looping == "continue" :
        mse = np.mean((y - ((slope * x) + intercept)) * (y - ((slope * x) + intercept)))
        
        random_index = random.randint(len(x))
        x_random = x[random_index]
        y_random = y[random_index]
        
        gradient_slope = (-2 * x_random * (y_random - ((slope * x_random) + intercept))) / len(x)
        gradient_intercept = (-2 * (y_random - ((slope * x_random) + intercept))) / len(x)
        
        step_size_slope = gradient_slope * learning_rate
        step_size_intercept = gradient_intercept * learning_rate
        
        slope = slope - step_size_slope
        intercept = intercept - step_size_intercept

        if str(gradient_slope) == "nan" or str(gradient_intercept) == "nan" :
            slope = 1
            intercept = 0
            learning_rate = learning_rate / 10
            iterations = 0
            
            tot_gradient_slope = []
            tot_gradient_intercept = []
            tot_slope = []
            tot_intercept = []
            tot_mse = []
        
        if abs(gradient_slope) < minimum_gradient and abs(gradient_intercept) < minimum_gradient :
            looping = "stop"
        elif iterations > maximum_iterations :
            looping = "stop"

        tot_gradient_slope.append(gradient_slope)
        tot_gradient_intercept.append(gradient_intercept)
        tot_slope.append(slope)
        tot_intercept.append(intercept)
        tot_mse.append(mse)

        iterations = iterations + 1

        print("\n", iterations, "iteration(s) is done")
        print("\n", "The Mean Square Error (MSE): ", mse)
        print("\n", "The Gradient of The MSE and The Slope is: ", gradient_slope)
        print("\n", "The Gradient of The MSE and The Intercept is: ", gradient_intercept)
        print("\n", "The Slope: ", slope)
        print("\n", "The Intercept: ", intercept)
        print("\n", "The Learning Rate (alpha): ", learning_rate, "\n")
    
        if looping == "continue" :
            print("\n", "The Loop is Continuing . . .")
        else :
            print("\n", "The Loop is Stopped . . .")
    
    return slope, intercept, gradient_slope, gradient_intercept, tot_gradient_slope, tot_gradient_intercept, tot_slope, tot_intercept, tot_mse, learning_rate

slope, intercept, gradient_slope, gradient_intercept, tot_gradient_slope, tot_gradient_intercept, tot_slope, tot_intercept, tot_mse, learning_rate = gradient_descent(x, y, slope, intercept, learning_rate, minimum_gradient, maximum_iterations)

#linear regression line
model_x = np.arange((min(x) - (min(x) / 10)), (max(x) + (max(x) / 10)), (len(x) / 100))
model_y = (slope * model_x) + intercept
y_pred = (slope * x) + intercept

eq = "Predicted Line: Y = (" + str(np.around(slope, 3)) + ")" + "X" + " + " + "(" + str(np.around(intercept, 3)) + ")"

#correlation actual y against predicted y
r = np.corrcoef(y, y_pred)[0, 1]

#plotting
font1 = {"family":"serif","color":"#1D1D1D","size":14}
font2 = {"family":"serif","color":"#1D1D1D","size":12}

plt.subplot(2, 2, 1)
plt.plot(np.arange(0, len(tot_gradient_slope)), tot_gradient_slope, label = "The Gradient of The MSE The Slope", c = "crimson")
plt.plot(np.arange(0, len(tot_gradient_intercept)), tot_gradient_intercept, label = "The Gradient of The MSE and The Intercept", c = "midnightblue")
plt.plot(np.arange(0, len(tot_mse)), tot_mse, label = "The Mean Square Error (MSE)", c = "forestgreen")
plt.legend(loc = "lower right")
plt.xlabel("Number of Iterations", fontdict = font2)
plt.ylabel("Amplitude", fontdict = font2)
plt.title(("(Number of Iteration, Amplitude) " + ", alpha: " + str(learning_rate)), fontdict = font1)
plt.grid()

plt.subplot(2, 2, 2)
plt.scatter(np.arange(0, len(tot_slope)), tot_slope, label = "The Slope", c = "red")
plt.scatter(np.arange(0, len(tot_intercept)), tot_intercept, label = "The Intercept", c = "blue")
plt.legend(loc = "lower right")
plt.xlabel("Number of Iterations", fontdict = font2)
plt.ylabel("Amplitude", fontdict = font2)
plt.title(("(Number of Iteration, Amplitude) " + ", alpha: " + str(learning_rate)), fontdict = font1)
plt.grid()

plt.subplot(2, 2, 3)
plt.scatter(x, y, label = "Data", c = "dodgerblue")
plt.plot(model_x, model_y, label = eq, c = "black")
plt.plot(model_x, model_y_initial, label = eq_initial, c = "red")
plt.legend(loc = "lower right")
plt.xlabel(label_x, fontdict = font2)
plt.ylabel(label_y, fontdict = font2)
plt.title(("(" + label_x + ", " + label_y + ")"), fontdict = font1)
plt.grid()

plt.subplot(2, 2, 4)
plt.scatter(y, y_pred, c = "dodgerblue")
plt.xlabel(("Actual " + label_y), fontdict = font2)
plt.ylabel(("Predicted " + label_y), fontdict = font2)
plt.title(("(Actual " + label_y + ", " + "Predicted " + label_y + ")" + ", r = " + str(np.around(r, 3))), fontdict = font1)
plt.grid()

plt.subplots_adjust(hspace = 0.5)

if str(gradient_slope) == "nan" or str(gradient_intercept) == "nan" :
    print("\n", "Plot cannot be done . . .")
else :
    plt.show()

print("\n")
