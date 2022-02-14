__author__ = "Pratik Acharya"
__copyright__ = "Copyright 2022"
__UID__ = "117513615"


''' 

Instructions to run the code 
1. Run 'python hw1_p4.py' in an ubuntu terminal

'''

import numpy as np

# Initializing the variables to be used in the matrix
x1 = x4 = 5
x2 = x3 = 150

y1 = y2 = 5
y3 = y4 = 150

xp1 = xp4 = 100
xp2 = 200
xp3 = 220

yp1 = 100
yp2 = yp3 = 80
yp4 = 200

# Initializing the matrix
A = np.matrix([[-x1, -y1, -1, 0, 0, 0, x1*xp1, y1*xp1, xp1],
               [0, 0, 0, -x1, -y1, -1, x1*yp1, y1*yp1, yp1],
               [-x2, -y2, -1, 0, 0, 0, x2*xp2, y2*xp2, xp2],
               [0, 0, 0, -x2, -y2, -1, x2*yp2, y2*yp2, yp2],
               [-x3, -y3, -1, 0, 0, 0, x3*xp3, y3*xp3, xp3],
               [0, 0, 0, -x3, -y3, -1, x3*yp3, y3*yp3, yp3],
               [-x4, -y4, -1, 0, 0, 0, x4*xp4, y4*xp4, xp4],
               [0, 0, 0, -x4, -y4, -1, x4*yp4, y4*yp4, yp4]])

# Calculating the AA^t and A^tA matrices
W = A * A.T
X = A.T * A

# Finding the eigen values and eigen vectors of the calculated matrices
eig_u, U = np.linalg.eig(W)
eig_v, V = np.linalg.eig(X)

# Sorting the eigen values and vectors
W_sort = np.argsort(eig_u)[::-1]
eig_u = eig_u[W_sort]
U = U[:, W_sort]

X_sort = np.argsort(eig_v)[::-1]
eig_v = eig_v[X_sort]
V = V[:, X_sort]

# Calculating the S matrix
S = np.diag(np.sqrt(eig_u))
S = np.hstack((S, np.zeros((8, 1))))

# Displaying the homography matrix
print(np.asmatrix(V[:, eig_v.argmin()]).reshape(3, 3))
