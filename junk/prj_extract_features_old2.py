import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True


def main():

    # load images
    template_image = load_image('template_pcb.jpg')
    test_image = load_image('defected_pcb.jpg')

    # resize images to make them lighter
    template_image = image_resize(template_image, height = 400)
    test_image = image_resize(test_image, height = 400)

    # smooth images and reduce noises
    gaus_template, gaus_test = smooth_image(template_image, test_image)


    f, ax = plt.subplots(1,2,figsize=(12,6))



    # create mask for usb detection
    height = gaus_test.shape[0]
    width = gaus_test.shape[1]
    usb_mask = create_mask(height, width, [(0,300),(390,400)])

    usb = extract_feature(test_image, gaus_test, gaus_template, usb_mask, (105,130), 'red')


    showim([usb])


def extract_feature(original_test_image, template_image, test_image, mask, color_range, output_color):

    # apply the mask on the image
    test_image = test_image + 1
    test_image_masked = test_image * mask
    template_image = template_image + 1
    template_image_masked = template_image * mask
    
    # extract the usb feature
    feature_test = cv.inRange(test_image_masked, color_range[0], color_range[1])
    feature_template = cv.inRange(template_image_masked, color_range[0], color_range[1])

    # enhance usb feature quality
    kernel = np.ones((15,15))
    feature_test_close = cv.morphologyEx(feature_test, cv.MORPH_CLOSE, kernel)
    feature_template_close = cv.morphologyEx(feature_template, cv.MORPH_CLOSE, kernel)


    # test & template images subtraction
    negative_feature  = feature_template_close - feature_test_close

    # enhance usb feature quality
    kernel = np.ones((15,15))
    negative_feature_open = cv.morphologyEx(negative_feature, cv.MORPH_OPEN, kernel)
    
    # switch colors
    if output_color == 'red':
        color_mask = [255, 0, 0]
    elif output_color == 'green':
         color_mask = [0, 255, 0]
    else:
        color_mask = [0, 0, 255]

    # paint the feature with the desired color
    defects = original_test_image.copy()
    defects[negative_feature_open == 255] = color_mask
    showim([original_test_image, defects, negative_feature_open])
    return defects

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

def create_mask(height, width, mask_ranges):
    mask = np.ones((height, width), dtype=np.uint8)
    for h in mask_ranges:
        mask[h[0] : h[1], :] = 0
    return mask


def show_plot(plot1, plot2, title1, title2):
    plt.plot(plot1, label=title1)
    plt.plot(plot2, label=title2)
    plt.legend()
    plt.show()


def load_image(path):
    image = cv.imread(path, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def showim(images, titles = []):
    print(len(images))
    if (len(images) == 1):
        f, ax = plt.subplots(figsize=(6 * len(images),6))
        ax.imshow(images[0], cmap='gray')
        ax.axis('off')
        if (len(titles) > 0):
            ax.title.set_text(titles[0])
    else:
        f, ax = plt.subplots(1,len(images),figsize=(6 * len(images),6))
        for i in range(0, len(images)):
            ax[i].imshow(images[i], cmap='gray')
            ax[i].axis('off')
            if (len(titles) > 0):
                ax[i].title.set_text(titles[i])
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

main()