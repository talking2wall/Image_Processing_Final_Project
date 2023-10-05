import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True


def main():

    # load images
    template_image = load_image('images/template_pcb.jpg')
    test_image = load_image('images/defected_pcb.jpg')

    # resize images to make them lighter
    template_image_resized = image_resize(template_image, height = 400)
    test_image_resized = image_resize(test_image, height = 400)
    
    # smooth images and reduce noises
    template_image_enhanced, test_image_enhanced = smooth_image(template_image_resized, test_image_resized)

    #showim([test_image_enhanced])
    #show_hist(template_image_enhanced, test_image_enhanced)

    # get image dimentions
    height = test_image_enhanced.shape[0]
    width = test_image_enhanced.shape[1]

    # usb feature extraction
    mask = create_mask(height, width, [(0,360),(397,400)])
    usb = extract_feature(test_image_resized, template_image_enhanced, test_image_enhanced, mask, (90,110), 'red')

    # capacitors feature extraction
    mask = create_mask(height, width, [(0,125),(165,400)])
    capacitors = extract_feature(test_image_resized, template_image_enhanced, test_image_enhanced, mask, (115,120), 'green')

    # chip feature extraction
    mask = create_mask(height, width, [(0,160),(280,400)])
    chip = extract_feature(test_image_resized, template_image_enhanced, test_image_enhanced, mask, (13,39), 'blue')

    showim([template_image_resized, test_image_resized, usb, capacitors, chip],
           ['Template PCB', 'Defected PCB', 'Find Missing USB', 'Find Missing Capacitor', 'Find Missing Chip'])


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
    kernel = np.ones((3,3))
    feature_test_close = cv.morphologyEx(feature_test, cv.MORPH_CLOSE, kernel)
    feature_template_close = cv.morphologyEx(feature_template, cv.MORPH_CLOSE, kernel)

    # test & template images subtraction
    negative_feature = feature_template_close - feature_test_close

    # enhance usb feature quality
    kernel = np.ones((7,7))
    negative_feature_open = cv.morphologyEx(negative_feature, cv.MORPH_CLOSE, kernel)
    
    # switch colors
    if output_color == 'red':
        color_mask = [255, 0, 0]
    elif output_color == 'green':
         color_mask = [0, 255, 0]
    else:
        color_mask = [0, 0, 255]

    # paint the feature with the desired color
    a = original_test_image.copy()
    a[negative_feature_open == 255] = color_mask
    
    return a


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

main()