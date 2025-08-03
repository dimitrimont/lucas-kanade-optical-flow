# Dimitri Montgomery
# CSC 340
# 04/10/2025
# Optical Flow

import math
import numpy as np
import cv2


def Imatrices(emptyIMGIx, emptyIMGIy, emptyIMGIxx, emptyIMGIyy, emptyIMGIxy, emptyIMGIt, emptyIMGIxt, emptyIMGIyt):
    for i in range(1, rows1 - 1):
        for j in range(1, cols1 - 1):

            # Calculate x gradient using horizontal neighbors
            emptyIMGIx[i, j] = float(image1o[i, j + 1]) - float(image1o[i, j - 1])
            # Calculate y gradient using vertical neighbors
            emptyIMGIy[i, j] = float(image1o[i + 1, j]) - float(image1o[i - 1, j])


            # Calculate products of gradients
            emptyIMGIxx[i, j] = emptyIMGIx[i, j] * emptyIMGIx[i, j]
            emptyIMGIyy[i][j] = emptyIMGIy[i, j] * emptyIMGIy[i, j]
            emptyIMGIxy[i][j] = emptyIMGIx[i, j] * emptyIMGIy[i, j]


            # Calculate temporal gradient
            emptyIMGIt[i, j] = float(image2o[i, j]) - float(image1o[i, j])
            # Calculate products with temporal gradient
            emptyIMGIxt[i, j] = emptyIMGIx[i, j] * emptyIMGIt[i, j]
            emptyIMGIyt[i, j] = emptyIMGIy[i, j] * emptyIMGIt[i, j]


    return emptyIMGIx, emptyIMGIy, emptyIMGIxx, emptyIMGIyy, emptyIMGIxy, emptyIMGIt, emptyIMGIxt, emptyIMGIyt



def fillCornerMTX(emptyIMGIxx, emptyIMGIyy, emptyIMGIxy, emptyIMGIt, emptyIMGIxt, emptyIMGIyt, colored_img):
    for i in range(4, rows1 - 4):
        for j in range(4, cols1 - 4):

            # Initialize sums for 7x7 window
            sumIxx = 0
            sumIyy = 0
            sumIxy = 0
            sumIxt = 0
            sumIyt = 0

            # Define window boundaries
            startI = i - 3
            endI = i + 4
            startJ = j - 3
            endJ = j + 4

            # Sum gradient values in the window
            for x in range(startI, endI):
                for y in range(startJ, endJ):
                    sumIxx += emptyIMGIxx[x, y]
                    sumIyy += emptyIMGIyy[x, y]
                    sumIxy += emptyIMGIxy[x, y]
                    sumIxt += emptyIMGIxt[x, y]
                    sumIyt += emptyIMGIyt[x, y]


            # determinate
            det_M = (sumIxx * sumIyy) - (sumIxy * sumIxy)

            if det_M == 0:
                u = 0
                v = 0
            else:
                # Solve for optical flow (u,v)
                u = ((-(sumIyy)*(sumIxt)) + ((sumIxy)*(sumIyt))) / det_M
                v = (((sumIxy)*(sumIxt)) - ((sumIxx)*(sumIyt))) / det_M

            # Calculate the magnitude of the flow vector
            mag = math.sqrt(u**2 + v**2)


            # Store computed flow values
            flowIMGuX[i, j] = u
            flowIMGvY[i, j] = v
            flowIMGMag[i, j] = mag



            # Convert flow direction to color
            theta = math.atan2(v, u)  # Direction of flow vector
            blue = ((theta + math.pi) / (2 * math.pi))  # Map angle to blue channel [0,1]
            green = 1 - blue  # Inverse mapping for green channel
            colored_img[i][j][0] = blue * mag  # Scale blue by magnitude
            colored_img[i][j][1] = green * mag  # Scale green by magnitude

    return flowIMGuX, flowIMGvY, flowIMGMag, colored_img

def display_menu():
    print("\nOptical Flow Menu:")
    print("1. View Flow Color Image")
    print("2. View Flow U")
    print("3. View Flow V")
    print("4. View Flow Magnitude")
    print("5. Exit")
    return input("Enter your choice (1-5): ")



if __name__ == "__main__":
    # Load input images
    # image1o = cv2.imread("sphere1.jpg", 0)
    # image2o = cv2.imread("sphere2.jpg", 0)

    # image1o = cv2.imread("house1.bmp", 0)
    # image2o = cv2.imread("house2.bmp", 0)

    # image1o = cv2.imread("tree1.jpg", 0)
    # image2o = cv2.imread("tree2.jpg", 0)

    image1o = cv2.imread("coke1.jpg", 0)
    image2o = cv2.imread("coke2.jpg", 0)


    rows1 = image1o.shape[0]
    cols1 = image1o.shape[1]


    # Create empty images for Ix, Iy, Ixx, Iyy, and Ixy (all the same dimensions as your original image)
    emptyIMGIx = np.zeros((rows1, cols1), np.float32)
    emptyIMGIy = np.zeros((rows1, cols1), np.float32)
    emptyIMGIxx = np.zeros((rows1, cols1), np.float32)
    emptyIMGIyy = np.zeros((rows1, cols1), np.float32)
    emptyIMGIxy = np.zeros((rows1, cols1), np.float32)

    # Temporal gradient arrays
    emptyIMGIt = np.zeros((rows1, cols1), np.float32)
    emptyIMGIxt = np.zeros((rows1, cols1), np.float32)
    emptyIMGIyt = np.zeros((rows1, cols1), np.float32)

    # Initialize arrays for storing optical flow results
    flowIMGuX = np.zeros((rows1, cols1), np.float32)
    flowIMGvY = np.zeros((rows1, cols1), np.float32)
    flowIMGMag = np.zeros((rows1, cols1), np.float32)

    colored_img = np.zeros((rows1, cols1, 3), np.float32)


    # Calculate gradients and their values
    emptyIMGIx, emptyIMGIy, emptyIMGIxx, emptyIMGIyy, emptyIMGIxy, emptyIMGIt, emptyIMGIxt, emptyIMGIxt = Imatrices(emptyIMGIx, emptyIMGIy, emptyIMGIxx, emptyIMGIyy, emptyIMGIxy, emptyIMGIt, emptyIMGIxt, emptyIMGIyt)

    # Calculate optical flow vectors and visualizations
    flowIMGuX, flowIMGvY, flowIMGMag, colored_img = fillCornerMTX(emptyIMGIxx, emptyIMGIyy, emptyIMGIxy, emptyIMGIt, emptyIMGIxt, emptyIMGIyt, colored_img)

    while True:
        choice = display_menu()

        if choice == '1':
            cv2.imshow("Flow IMG Color", colored_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == '2':
            cv2.imshow("Flow IMG U", flowIMGuX)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == '3':
            cv2.imshow("Flow IMG V", flowIMGvY)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == '4':
            cv2.imshow("Flow IMG Magnitude", flowIMGMag)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        elif choice == '5':
            print("Exiting program...")
            break
        else:
            print("Invalid choice. Please try again.")

    cv2.imwrite("Sphere_IMG.jpg", colored_img)
    cv2.imwrite("Flow_IMG_U.jpg", flowIMGuX)
    cv2.imwrite("Flow_IMG_v.jpg", flowIMGvY)
    cv2.imwrite("Flow_IMG_Mag.jpg", flowIMGMag)

