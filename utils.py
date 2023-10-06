import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random


def rotate_image(image, angle):

    # Get the dimensions of the image
    height, width = image.shape[:2]

    # Calculate the center of the image (the rotation pivot point)
    center = (width // 2, height // 2)

    # Create the rotation matrix
    rotation_matrix = cv.getRotationMatrix2D(center, angle, 1.0)

    # Apply the rotation to the image
    rotated_image = cv.warpAffine(image, rotation_matrix, (width, height))

    # Append the rotated image to the list
    return rotated_image


def add_salt_and_pepper_noise(image, salt_prob=0.02, pepper_prob=0.02):
    noisy_image = np.copy(image)
    total_pixels = image.shape[0] * image.shape[1]

    # Add salt noise
    num_salt = int(total_pixels * salt_prob)
    salt_coordinates = [np.random.randint(0, i - 1, num_salt) for i in image.shape]
    noisy_image[salt_coordinates[0], salt_coordinates[1], :] = 255

    # Add pepper noise
    num_pepper = int(total_pixels * pepper_prob)
    pepper_coordinates = [np.random.randint(0, i - 1, num_pepper) for i in image.shape]
    noisy_image[pepper_coordinates[0], pepper_coordinates[1], :] = 0

    return noisy_image


def smooth_image(template_image, test_image):
    # covert images to gray-scale
    gray_template_img = cv.cvtColor(template_image,cv.COLOR_BGR2GRAY)
    gray_test_img = cv.cvtColor(test_image,cv.COLOR_BGR2GRAY)
    
    # apply median blur
    med_template = cv.medianBlur(gray_template_img,3)
    med_test = cv.medianBlur(gray_test_img,3)

    # apply gaussian blur
    gaus_template = cv.GaussianBlur(med_template,ksize=(3,3),sigmaX=1)
    gaus_test = cv.GaussianBlur(med_test,ksize=(3,3),sigmaX=1)

    return gaus_template, gaus_test


def create_mask(height, width, mask_ranges, mask_angle = 0):
    mask = np.ones((height, width), dtype=np.uint8)
    for h in mask_ranges:
        mask[h[0] : h[1], :] = 0
    if mask_angle != 0:
        mask = rotate_image(mask, mask_angle)
    return mask


def show_hist(template_image, test_image):
    template_image_hist = cv.calcHist([template_image],[0],None,[256],[0,256])
    test_image_hist = cv.calcHist([test_image],[0],None,[256],[0,256])
    plt.plot(template_image_hist, label='Working PCB')
    plt.plot(test_image_hist, label='Defected PCB')
    plt.legend()
    plt.show()

def load_image(path):
    image = cv.imread(path, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def showim(images, titles = []):
    if (len(images) == 1):
        f, ax = plt.subplots(figsize=(8 * len(images),8))
        ax.imshow(images[0], cmap='gray')
        ax.axis('off')
        if (len(titles) > 0):
            ax.title.set_text(titles[0])
        f.tight_layout()
    else:
        f, ax = plt.subplots(1,len(images),figsize=(8 * len(images),8))
        for i in range(0, len(images)):
            ax[i].imshow(images[i], cmap='gray')
            ax[i].axis('off')
            if (len(titles) > 0):
                ax[i].title.set_text(titles[i])
        f.tight_layout()
    plt.show() 


# Taken from: https://stackoverflow.com/questions/44650888/resize-an-image-without-distortion-opencv
# Thanks to: thewaywewere
def image_resize(image, width = None, height = None, inter = cv.INTER_AREA):
    # initialize the dimensions of the image to be resized and
    # grab the image size
    dim = None
    (h, w) = image.shape[:2]

    # if both the width and height are None, then return the
    # original image
    if width is None and height is None:
        return image

    # check to see if the width is None
    if width is None:
        # calculate the ratio of the height and construct the
        # dimensions
        r = height / float(h)
        dim = (int(w * r), height)

    # otherwise, the height is None
    else:
        # calculate the ratio of the width and construct the
        # dimensions
        r = width / float(w)
        dim = (width, int(h * r))

    # resize the image
    resized = cv.resize(image, dim, interpolation = inter)

    # return the resized image
    return resized


def detect_pcb_by_color(image):
    # lower_green = np.array([0, 35, 0], dtype="uint8")
    # upper_green = np.array([100, 255, 100], dtype="uint8")
    # Define the color range for the PCB (green color in BGR format)
    lower_green = np.array([10, 10, 10], dtype="uint8")
    upper_green = np.array([50, 255, 50], dtype="uint8")

    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen the image
    blurred = cv.GaussianBlur(gray, (5, 5), 5)

    # Apply Canny edge detection to find edges
    edges = cv.Canny(blurred, 50, 150)

    # Combine color-based detection (green) with edge detection
    color_mask = cv.inRange(image, lower_green, upper_green)
    combined_mask = cv.bitwise_or(color_mask, edges)

    # Apply morphological operations (e.g., closing) to further process the mask
    kernel = np.ones((5, 5), np.uint8)
    closed_mask = cv.morphologyEx(combined_mask, cv.MORPH_CLOSE, kernel)

    # Find the contours of the PCB in the closed mask
    contours, _ = cv.findContours(closed_mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialize variables to keep track of the largest contour
    largest_contour = None
    max_contour_area = 0

    # Iterate through the contours and hierarchy
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > max_contour_area:
            max_contour_area = area
            largest_contour = contour

    # Calculate the angle of the PCB
    angle = 0.0  # Default angle if no valid contour is found
    if largest_contour is not None:
        rect = cv.minAreaRect(largest_contour)
        angle = rect[2]

    # Draw the largest PCB contour on a copy of the original image
    result_image = image.copy()
    if largest_contour is not None:
        cv.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)

    return result_image, angle


def detect_pcb_by_edges(image):
    # Convert the image to grayscale
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)

    # Apply Gaussian blur to reduce noise and smoothen the image
    blurred = cv.GaussianBlur(gray, (5, 5), 0)

    # Apply Canny edge detection to find edges
    edges = cv.Canny(blurred, 30, 90)

    # Apply dilation to connect nearby edges and enhance contour detection
    kernel = np.ones((5, 5), np.uint8)
    dilated_edges = cv.dilate(edges, kernel, iterations=1)

    # Find the contours of the PCB in the edge-detected image
    contours, _ = cv.findContours(dilated_edges, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)

    # Initialize variables to keep track of the largest contour
    largest_contour = None
    max_contour_area = 0

    # Iterate through the contours and hierarchy
    for i, contour in enumerate(contours):
        area = cv.contourArea(contour)
        if area > max_contour_area:
            max_contour_area = area
            largest_contour = contour

    # Calculate the angle of the PCB
    if largest_contour is not None:
        rect = cv.minAreaRect(largest_contour)
        angle = rect[2]

        # Determine orientation based on the aspect ratio of the bounding box
        width, height = rect[1]
        aspect_ratio = max(width, height) / min(width, height)

        # Adjust the angle based on the aspect ratio
        
        if aspect_ratio > 1.2 and angle >= 90:
            angle -= 90
    else:
        angle = 0.0  # Default angle if no valid contour is found

    # Draw the largest PCB contour on the original image
    result_image = image.copy()
    if largest_contour is not None:
        cv.drawContours(result_image, [largest_contour], -1, (0, 255, 0), 2)

    return result_image, angle


def get_orientation(template_image, test_image):
    # Initialize the ORB detector
    orb = cv.ORB_create()

    # Find keypoints and descriptors for the template and test images
    keypoints_template, descriptors_template = orb.detectAndCompute(template_image, None)
    keypoints_test, descriptors_test = orb.detectAndCompute(test_image, None)

    # Create a Brute-Force Matcher
    bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)

    # Match the descriptors
    matches = bf.match(descriptors_template, descriptors_test)

    # Use the top matches to calculate the orientation
    num_matches_to_use = 10  # You can adjust the number of matches to use
    if len(matches) >= num_matches_to_use:
        # Collect the angles of matched keypoints
        angles = []
        for match in matches[:num_matches_to_use]:
            angle_template = keypoints_template[match.queryIdx].angle
            angle_test = keypoints_test[match.trainIdx].angle
            angles.append(angle_template - angle_test)

        # Calculate the orientation as the median of the angle differences
        orientation = np.median(angles)
        return orientation

    return None  # Not enough matches to reliably determine orientation