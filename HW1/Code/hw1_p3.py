__author__ = "Pratik Acharya"
__copyright__ = "Copyright 2022"
__UID__ = "117513615"


''' 

Instructions to run the code 
1. Make sure the csv file is in the same folder as the code and are named as given below
2. Run 'python hw1_p3.py' in an ubuntu terminal

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from random import randint

# Reading the data from csv file
data = pd.read_csv('ENPM673_hw1_linear_regression_dataset - Sheet1.csv')

# Converting the data to numpy arrays
ages = data['age'].to_numpy()
insurance_costs = data['charges'].to_numpy()

# Calculating the mean and initializing variables to store the sum
mean_a = np.mean(ages)
mean_i = np.mean(insurance_costs)
sum_ = 0
sum_a = 0
sum_i = 0

# Initializing the plots
fig, axes = plt.subplots(2, 2)
axes[0][0].plot(ages, insurance_costs, '.', color='black')
axes[0][1].plot(ages, insurance_costs, '.', color='black')
axes[1][0].plot(ages, insurance_costs, '.', color='black')
axes[1][1].plot(ages, insurance_costs, '.', color='black')

# Calculating the variances for the covariance matrix
for age, insurance_cost in zip(ages, insurance_costs):
    var_a = age - mean_a
    var_i = insurance_cost - mean_i
    sum_ += var_i * var_a
    sum_a += var_a ** 2
    sum_i += var_i ** 2

N = len(ages)

# Covariance Matrix
cov_mat = np.array([[sum_a / N, sum_ / N], [sum_ / N, sum_i / N]])

# Eigen values and vectors of the covariance matrix
eig_val, eig_vec = np.linalg.eig(cov_mat)

# Setting the origin for the eigen vectors
origin = [mean_a, mean_i]

eig_vec1 = eig_vec[:, 0]
eig_vec2 = eig_vec[:, 1]

# Plotting the eigen vectors
axes[0, 0].quiver(*origin, *eig_vec1, color=['r'], scale=21)
axes[0, 0].quiver(*origin, *eig_vec2, color=['b'], scale=21)


########## Standard Least Square ##########

# Defining the objective function
def objective(x, a, b):
    return a * x + b


# Function to calculate standard least square
def LeastSquare(x, y):

    # Converting arrays into column matrix
    x = np.asmatrix(x).T
    y = np.asmatrix(y).T

    # Appending with a column vector containing 1's
    x_new = np.hstack((x, np.ones((len(x), 1))))

    # Calculating the B matrix
    b_mat = np.linalg.inv((x_new.T * x_new)) * (x_new.T * y)

    # Returning the matrix as an array
    return [b_mat[0, 0], b_mat[1, 0]]


# Using the defined SLS function on the dataset
poly = LeastSquare(ages, insurance_costs)
fit = objective(ages, poly[0], poly[1])

# Plotting the line
axes[0, 1].plot(ages, fit, 'r', label="SLS")


########## Total Least Square ##########

# Function to calculate total least square
def TotalLeastSq(x, y):

    # Generating a matrix containing variances
    x = x - x.mean()
    y = y - y.mean()

    # Appending the matrix to form the U matrix
    u_mat = np.hstack((np.asmatrix(x).T, np.asmatrix(y).T))

    # Calculating the eigen values and vectors of the U matrix
    eig_tl, eig_vec_tl = np.linalg.eig(u_mat.T * u_mat)

    # Returning the eigen vector corresponding the smallest eigen value
    return np.asarray(eig_vec_tl[:, eig_tl.argmin()]).flatten()


# Using the TLS function on the given dataset
total_poly = TotalLeastSq(ages, insurance_costs)

# Calculating the points on the line
total_fit = ((total_poly[0] * ages.mean() + total_poly[1]) - total_poly[0] * ages) / total_poly[1]

# Plotting the line
axes[1, 0].plot(ages, total_fit, 'g', label="TLS")


########## RANSAC ##########

# Function to fit a line using RANSAC
def RANSAC(x, y, samples, dist):

    # Setting the max inliers in the threshold to zero
    max_inlier = 0

    # Looping for the given amount of samples
    for i in range(samples):

        # Selecting two random points
        point1 = randint(0, len(x) - 1)
        point2 = randint(0, len(x) - 1)

        # Initializing the variable to store the inliers for the above points
        inlier = 0

        # Making sure that same points are not selected
        if point2 != point1:

            # Converting the points into arrays
            p1 = np.asarray((x[point1], y[point1]))
            p2 = np.asarray((x[point2], y[point2]))

            # Iterating throughout the dataset and finding the points that lie within the threshold
            for point in range(len(x)):

                # Converting the point into an array
                p3 = np.asarray((x[point], y[point]))

                # Calculating the perpendicular distance of the point from the line
                d = np.linalg.norm(np.cross(p2 - p1, p1 - p3)) / np.linalg.norm(p2 - p1)

                # Checking if the point lies within the threshold
                if d < dist:
                    inlier += 1

        # Updating the line that has maximum inliers
        if inlier > max_inlier:
            max_inlier = inlier
            final_p1 = point1
            final_p2 = point2

    # Calculating slope for the best fit
    slope = (y[final_p2] - y[final_p1]) / (x[final_p2] - x[final_p1])

    # Returning the slope and intercept of the best fit line
    return [slope, y[final_p1] - slope * x[final_p1]]


# Using the RANSAC function on the given dataset
coefficients = RANSAC(ages, insurance_costs, 1000, 10)

# Initializing a polynomial with the RANSAC coefficients
polynomial = np.poly1d(coefficients)

# Calculating points on the line
ransac_y = polynomial(ages)

# Plotting the line
axes[1, 1].plot(ages, ransac_y, 'y', label='RANSAC')


# Labeling axes and adding legends to the plots
fig.text(0.5, 0.04, 'Ages', ha='center', va='center')
fig.text(0.06, 0.5, 'Insurance Costs', ha='center', va='center', rotation='vertical')
axes[0][1].legend(loc="upper right")
axes[1][0].legend(loc="upper right")
axes[1][1].legend(loc="upper right")
fig.tight_layout(pad=2.0)

# Displaying the plot
plt.show()
