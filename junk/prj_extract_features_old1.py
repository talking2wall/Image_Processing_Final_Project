import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['interactive'] == True





def main():

    # load images
    template_image = load_image('template_pcb.jpg')
    test_image = load_image('defected_pcb.jpg')

    # covert images to gray-scale
    gray_template_img = cv.cvtColor(template_image,cv.COLOR_BGR2GRAY)
    gray_test_img = cv.cvtColor(test_image,cv.COLOR_BGR2GRAY)
    
    # # plot histogram to show the difference in pixel intensity between the images
    # hist_template = cv.calcHist([gray_template_img],[0],None,[256],[0,256])
    # hist_test = cv.calcHist([gray_test_img],[0],None,[256],[0,256])
    # #show_plot(hist_template, hist_test, 'Template Image', 'Test Image')

    # apply median blur
    med_test = cv.medianBlur(gray_test_img,3)
    med_template = cv.medianBlur(gray_template_img,3)

    # apply gaussian blur
    gaus_test = cv.GaussianBlur(med_test,ksize=(3,3),sigmaX=1)
    gaus_template = cv.GaussianBlur(med_template,ksize=(3,3),sigmaX=1)

    # # plot histogram to show the difference in pixel intensity between the images, after bluring the images
    # hist_template = cv.calcHist([gray_template_img],[0],None,[256],[0,256])
    # hist_test = cv.calcHist([gray_test_img],[0],None,[256],[0,256])
    # #show_plot(hist_template, hist_test, 'Template Image', 'Test Image')




    showim([gray_template_img, gray_test_img], ['Template Image', 'Test Image'])


def show_plot(plot1, plot2, title1, title2):
    plt.plot(plot1, label=title1)
    plt.plot(plot2, label=title2)
    plt.legend()
    plt.show()


def load_image(path):
    image = cv.imread(path, 1)
    image = cv.cvtColor(image, cv.COLOR_BGR2RGB)
    return image


def smooth_image():
    pass


def extract_feature(template_image, test_image, mask):
    pass


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