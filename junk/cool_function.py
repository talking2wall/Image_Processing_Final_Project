# def aa(image):
#     # Convert the image to the HSV color space (for better color segmentation)
#     hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

#     # Define the lower and upper bounds of the color you want to extract (in HSV)
#     lower_bound = np.array([hue_min, saturation_min, value_min])
#     upper_bound = np.array([hue_max, saturation_max, value_max])

#     # Create a mask to extract the desired color range
#     mask = cv.inRange(hsv_image, lower_bound, upper_bound)

#     # Apply the mask to the original image
#     result = cv.bitwise_and(image, image, mask=mask)

#     # Display the result
#     cv.imshow('Result', result)
#     cv.waitKey(0)
#     cv.destroyAllWindows()

def cc(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply the Canny edge detector
    edges = cv.Canny(blurred, threshold1=30, threshold2=100)  # Adjust thresholds as needed

    # Display the result
    cv.imshow('Original Image', image)
    cv.imshow('Edges', edges)

    # Wait for a key press and then close the windows
    cv.waitKey(0)
    cv.destroyAllWindows()

def bb(image):
    # Convert the image to the HSV color space
    hsv_image = cv.cvtColor(image, cv.COLOR_BGR2HSV)

    # Split the HSV image into its individual channels
    h, s, v = cv.split(hsv_image)

    # Display the individual channels
    cv.imshow('Hue Channel', h)
    cv.imshow('Saturation Channel', s)
    cv.imshow('Value Channel', v)

    # Wait for a key press and then close the windows
    cv.waitKey(0)
    cv.destroyAllWindows()