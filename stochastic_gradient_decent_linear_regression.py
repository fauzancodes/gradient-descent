import random
import numpy as np
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
initial_a = input("Input the initial guess of the slope (suggestion: 1): ")
initial_b = input("Input the initial guess of the intercept (suggestion: 0): ")
learning_rate = input("Input the learning rate (alpha) (suggestion: 0.01): ")
minimum_gradient = input("Input the minimum gradient (suggestion: 0.0009): ")
maximum_iterations = input("Input the maximum number of iterations(suggestion: 10000): ")

#converting input to float
initial_a = float(initial_a)
initial_b = float(initial_b)
learning_rate = float(learning_rate)
minimum_gradient = float(minimum_gradient)
maximum_iterations = float(maximum_iterations)
iterations = 0

#initial predicted line
model_x = np.arange((min(x) - (min(x) / 10)), (max(x) + (max(x) / 10)), (len(x) / 100))
model_y_initial = (initial_a * model_x) - initial_b

eq_initial = "Initial Line: Y = (" + str(np.around(initial_a, 3)) + ")" + "X" + " + " + "(" + str(np.around(initial_b, 3)) + ")"

#gradient descent function, using loss function = sum square residual
def gradient_descent(x, y, initial_a, initial_b, learning_rate) :
    ssr = np.sum((y - ((initial_a * x) + initial_b)) * (y - ((initial_a * x) + initial_b)))

    random_scalar = random.randint(0, (len(x) - 1))

    x_random = x[random_scalar]
    y_random = y[random_scalar]

    gradient_a = -2 * x_random * (y_random - ((initial_a * x_random) + initial_b))
    gradient_b = -2 * (y_random - ((initial_a * x_random) + initial_b))

    step_size_a = gradient_a * learning_rate
    step_size_b = gradient_b * learning_rate

    new_a = initial_a - step_size_a
    new_b = initial_b - step_size_b

    return new_a, gradient_a, new_b, gradient_b, ssr

#looping
looping = "continue"

tot_gradient_a = []
tot_gradient_b = []
tot_a = []
tot_b = []
tot_ssr = []

while looping == "continue" :
    initial_a, gradient_a, initial_b, gradient_b, ssr  = gradient_descent(x, y, initial_a, initial_b, learning_rate)
    
    if str(gradient_a) == "nan" or str(gradient_b) == "nan" :
        looping = "stop"
        print("\n", "Since the gradient of the SSR and the slope or the intercept is approaching infinity," "\n", 
            "they produce nan (Not a Number) slope and intercept, so the calcualtion cannot be continued.", "\n",
            "so we will do some modification here, then we will recalculate the calculation,", "\n",
            "1. we will decrese the learning rate (alpha) to ", str(learning_rate / 10), "\n",
            "2. we will replace the slope value that has been calculated so far to 1", "\n",
            "3. we will replace the intercept value that has been calculated so far to 0", "\n")
        
        modif_aggreement = input("Do you aggree with these modification? (y/n) : ")

        if modif_aggreement == "y" :
            initial_a = 1
            initial_b = 0
            learning_rate = learning_rate / 10
            iterations = 0
            tot_gradient_a = []
            tot_gradient_b = []
            tot_a = []
            tot_b = []
            tot_ssr = []
            
            print("\n", "The Slope: ", initial_a)
            print("\n", "The Intercept: ", initial_b)
            print("\n", "The Learning Rate (alpha): ", learning_rate, "\n")

            continue_with_modif = input("Continue with these modification? (y/n) : ")

            if continue_with_modif == "y" :
                looping = "continue"
            else :
                looping = "stop"
                print("\n", "Calculation cannot be continued due to some problem . . .")

        else :
            looping = "stop"
            print("\n", "Calculation cannot be continued due to some problem . . .")

    if abs(gradient_a) < minimum_gradient and abs(gradient_b) < minimum_gradient :
        looping = "stop"
    elif iterations > maximum_iterations :
        looping = "stop"

    tot_gradient_a.append(gradient_a)
    tot_gradient_b.append(gradient_b)
    tot_a.append(initial_a)
    tot_b.append(initial_b)
    tot_ssr.append(ssr)

    iterations = iterations + 1

    print("\n", iterations, "iteration(s) is done")
    print("\n", "The Sum of Square Residual (SSR): ", ssr)
    print("\n", "The Gradient of The SSR and The Slope is: ", gradient_a)
    print("\n", "The Gradient of The SSR and The Intercept is: ", gradient_b)
    print("\n", "The Slope: ", initial_a)
    print("\n", "The Intercept: ", initial_b)
    
    if looping == "continue" :
        print("\n", "The Loop is Continuing . . .")
    else :
        print("\n", "The Loop is Stopped . . .")

#linear regression line
model_x = np.arange((min(x) - (min(x) / 10)), (max(x) + (max(x) / 10)), (len(x) / 100))
model_y = (initial_a * model_x) + initial_b
y_pred = (initial_a * x) + initial_b

eq = "Predicted Line: Y = (" + str(np.around(initial_a, 3)) + ")" + "X" + " + " + "(" + str(np.around(initial_b, 3)) + ")"

#correlation actual y against predicted y
r = np.corrcoef(y, y_pred)[0, 1]

#plotting
font1 = {"family":"serif","color":"#1D1D1D","size":14}
font2 = {"family":"serif","color":"#1D1D1D","size":12}

plt.subplot(2, 2, 1)
plt.plot(np.arange(0, len(tot_gradient_a)), tot_gradient_a, label = "The Gradient of The SSR The Slope", c = "crimson")
plt.plot(np.arange(0, len(tot_gradient_b)), tot_gradient_b, label = "The Gradient of The SSR and The Intercept", c = "midnightblue")
plt.plot(np.arange(0, len(tot_ssr)), tot_ssr, label = "The Sum of Square Residual (SSR)", c = "forestgreen")
plt.legend(loc = "lower right")
plt.xlabel("Number of Iterations", fontdict = font2)
plt.ylabel("Amplitude", fontdict = font2)
plt.title(("(Number of Iteration, Amplitude)" + ", alpha: " + str(learning_rate)), fontdict = font1)
plt.grid()

plt.subplot(2, 2, 2)
plt.plot(np.arange(0, len(tot_a)), tot_a, label = "The Slope", c = "red")
plt.plot(np.arange(0, len(tot_b)), tot_b, label = "The Intercept", c = "blue")
plt.legend(loc = "lower right")
plt.xlabel("Number of Iterations", fontdict = font2)
plt.ylabel("Amplitude", fontdict = font2)
plt.title(("(Number of Iteration, Amplitude)" + ", alpha: " + str(learning_rate)), fontdict = font1)
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

if str(gradient_a) == "nan" or str(gradient_b) == "nan" :
    print("\n", "Plot cannot be done . . .")
else :
    plt.show()

print("\n")