# Learning rate
l = 0.01

# Data
x = [1, 2, 3, 4]
y = [3, 5, 6, 7]

# Parameters
theta0 = 0
theta1 = 0

# Summations
sumh1 = 0
sumh2 = 0
sumh3 = 0

# Cost and gradients
j = 0
jtheta0 = 0
jtheta1 = 0

for i in range(len(x)):
    h0x = theta0 + theta1 * x[i]       # Hypothesis
    h1 = h0x - y[i]                    # Error
    h2 = h1 * h1                       # Squared error
    h3 = h1 * x[i]                     # Error * x

    sumh1 += h1
    sumh2 += h2
    sumh3 += h3

    j = sumh2 / (2 * len(x))
    jtheta0 = sumh1 / len(x)
    jtheta1 = sumh3 / len(x)

# Update parameters
theta0 = theta0 - l * jtheta0
theta1 = theta1 - l * jtheta1

# Output
print("summation of h0(x)-y =", sumh1)
print("summation of [h0(x)-y]^2 =", sumh2)
print("summation of (h0(x)-y)x =", sumh3)
print("J =", j)
print("dj/dtheta0 =", jtheta0)
print("dj/dtheta1 =", jtheta1)
print("updated theta0 =", theta0)
print("updated theta1 =", theta1)
