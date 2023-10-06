import utils
import numpy as np
import cv2 as cv

def extract(template_image, test_image, all_in_one_image = True, mask_angle = 0):
    
    # rotate images if neccessary
    if mask_angle != 0:
        template_image = utils.rotate_image(template_image, mask_angle)

    # resize images to make them lighter
    template_image_resized = utils.image_resize(template_image, height = 400)
    test_image_resized = utils.image_resize(test_image, height = 400)

    # smooth images and reduce noises
    template_image_enhanced, test_image_enhanced = utils.smooth_image(template_image_resized, test_image_resized)

    # get image dimentions 
    height = test_image_enhanced.shape[0]
    width = test_image_enhanced.shape[1]

    if all_in_one_image == True:
        # copy the defect image
        final_image = test_image_resized.copy()

        # usb feature extraction
        mask = utils.create_mask(height, width, [(0,360),(397,400)], mask_angle)
        final_image = extract_feature(final_image, template_image_enhanced, test_image_enhanced, mask, (90,110), 'red')

        # capacitors feature extraction
        mask = utils.create_mask(height, width, [(0,125),(165,400)], mask_angle)
        final_image = extract_feature(final_image, template_image_enhanced, test_image_enhanced, mask, (115,120), 'green')

        # chip feature extraction
        mask = utils.create_mask(height, width, [(0,160),(280,400)], mask_angle)
        if mask_angle != 0:
            mask = utils.rotate_image(mask, mask_angle)
        final_image = extract_feature(final_image, template_image_enhanced, test_image_enhanced, mask, (13,39), 'blue')
    else:
        # usb feature extraction
        mask = utils.create_mask(height, width, [(0,360),(397,400)], mask_angle)
        if mask_angle != 0:
            mask = utils.rotate_image(mask, mask_angle)
        usb = extract_feature(test_image_resized, template_image_enhanced, test_image_enhanced, mask, (90,110), 'red')

        # capacitors feature extraction
        mask = utils.create_mask(height, width, [(0,125),(165,400)], mask_angle)
        if mask_angle != 0:
            mask = utils.rotate_image(mask, mask_angle)
        capacitors = extract_feature(test_image_resized, template_image_enhanced, test_image_enhanced, mask, (115,120), 'green')

        # chip feature extraction
        mask = utils.create_mask(height, width, [(0,160),(280,400)], mask_angle)
        if mask_angle != 0:
            mask = utils.rotate_image(mask, mask_angle)
        chip = extract_feature(test_image_resized, template_image_enhanced, test_image_enhanced, mask, (13,39), 'blue')

        # combine all 3 images to one big image
        final_image = np.hstack((usb,capacitors,chip))

    return final_image


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
    original_image_copy = original_test_image.copy()
    original_image_copy[negative_feature_open == 255] = color_mask

    return original_image_copy