__author__ = "Pratik Acharya"
__copyright__ = "Copyright 2022"
__UID__ = "117513615"

''' 

Instructions to run the code 
1. Make sure the videos are in the same folder as the code and are named as given below
2. Run 'python hw1_p2.py' in an ubuntu terminal

'''

import cv2
import numpy as np
import matplotlib.pyplot as plt

cap = cv2.VideoCapture("ball_video1.mp4")  # Video file 1
cap1 = cv2.VideoCapture("ball_video2.mp4")  # Video file 2

# Arrays to store the coordinates for ball positions
x_centers = np.array([])
y_centers = np.array([])
x_centers1 = np.array([])
y_centers1 = np.array([])

# Defining HSV range for red color
lower_red = np.array([160, 100, 50])
upper_red = np.array([180, 255, 255])

# Video 1
while cap.isOpened():
   ret, frame = cap.read()

   if ret:
       hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convert from BGR to HSV

       # Creating a mask to get binary image of the frame
       mask = cv2.inRange(hsv, lower_red, upper_red)

       # USing moments function to get the center of the ball
       M = cv2.moments(mask)
       center_x = int(M["m10"] / M["m00"])
       center_y = int(M["m01"] / M["m00"])

       # Drawing a circle at the center of the ball
       cv2.circle(frame, (center_x, center_y), 10, (0, 0, 0), -1)
       x_centers = np.append(x_centers, center_x)
       y_centers = np.append(y_centers, center_y)

       # Displaying the video with centers drawn
       # Uncomment the following lines to view the video
       # cv2.imshow('frame', frame)

       # if cv2.waitKey(25) & 0xFF == ord('q'):
       #     break
   else:
       break

# Video 2
while cap1.isOpened():
   ret1, frame1 = cap1.read()

   if ret1:
       hsv1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2HSV)

       mask1 = cv2.inRange(hsv1, lower_red, upper_red)

       M1 = cv2.moments(mask1)
       center_x1 = int(M1["m10"] / M1["m00"])
       center_y1 = int(M1["m01"] / M1["m00"])

       cv2.circle(frame1, (center_x1, center_y1), 10, (0, 0, 0), -1)

       x_centers1 = np.append(x_centers1, center_x1)
       y_centers1 = np.append(y_centers1, center_y1)

       # Uncomment the following lines to view the video with center drawn
       # cv2.imshow('frame1', frame1)

       # if cv2.waitKey(25) & 0xFF == ord('q'):
       #     break
   else:
       break


# Function to fit a curve using the least square method
def LeastSquare(x, y):

   # Column matrix containing squares of x elements
   x_2 = np.asmatrix(x * x).T

   # Converting arrays into column matrices
   x = np.asmatrix(x).T
   y = np.asmatrix(y).T

   # Joining the column matrices
   x_new = np.hstack((x_2, x))
   x_new = np.hstack((x_new, np.ones((len(x), 1))))

   # Calculating the coefficients for the curve
   b_mat = np.linalg.inv((x_new.T * x_new)) * (x_new.T * y)

   # Returning the coefficients of the curve
   return [b_mat[0, 0], b_mat[1, 0], b_mat[2, 0]]


# Using the LeaseSquares method to get the coefficients for both the videos
popt = LeastSquare(x_centers, y_centers)
popt1 = LeastSquare(x_centers1, y_centers1)

# Defining a second degree polynomial for the curve
polynomial = np.poly1d(popt)
polynomial2 = np.poly1d(popt1)

# Using the polynomial equations to get the curve
y_new = polynomial(x_centers)
y_new1 = polynomial2(x_centers1)

# Plotting the ball positions and fitted curves
figure, axis = plt.subplots(1,2)

axis[0].set_xlabel("X axis")
axis[1].set_ylabel("Y axis")

axis[0].plot(x_centers, y_centers, 'o', color='black', label="Ball position")
axis[0].plot(x_centers, y_new, 'r', label='Fitted curve')
axis[0].invert_yaxis()
axis[0].set_title("Without Noise")
axis[0].legend(loc="lower center")

axis[1].plot(x_centers1, y_centers1, 'o', color='black', label="Ball position")
axis[1].plot(x_centers1, y_new1, 'r', label='Fitted curve')
axis[1].legend(loc="lower center")
axis[1].invert_yaxis()
axis[1].set_title("With Noise")
plt.show()

cv2.destroyAllWindows()
cap.release()
