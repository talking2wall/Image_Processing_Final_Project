import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import random
plt.rcParams['interactive'] == True
import utils
import feature_extraction


def main():

    # load images
    template_image = utils.load_image('images/template_pcb.jpg')
    test_image = utils.load_image('images/defected_pcb.jpg')


    # Extract features from the image (usb, capacitors, chip)
    image_features = feature_extraction.extract(template_image, test_image, False)
    utils.showim([image_features], ['Extacted Features (Defected PCB)'])


    # Extract features after applying noise
    test_image_noise = utils.add_salt_and_pepper_noise(test_image)
    image_noise = feature_extraction.extract(template_image, test_image_noise, True)
    utils.showim([test_image_noise, image_noise], ['Defected PCB (Noisy Image)', 'Detected Defects (Noisy Image)'])


    # Extract features after applying rotation
    test_image_45 = utils.rotate_image(test_image, 45)
    test_image_90 = utils.rotate_image(test_image, 90)
    test_image_180 = utils.rotate_image(test_image, 180)

    result_image_45 = feature_extraction.extract(template_image, test_image_45, True)
    result_image_90 = feature_extraction.extract(template_image, test_image_90, True)
    result_image_180 = feature_extraction.extract(template_image, test_image_180, True)

    utils.showim([result_image_45,  result_image_90, result_image_180],
                  ['Defected PCB 45° (Before Improvment)', 'Defected PCB 90° (Before Improvment)', 'Defected PCB 180° (Before Improvment)'])


    # Fixing feature extraction after applying roation
    extracted_angle_45 = utils.get_orientation(template_image, test_image_45)
    extracted_angle_90 = utils.get_orientation(template_image, test_image_90)
    extracted_angle_180 = utils.get_orientation(template_image, test_image_180)
    result_45_angle = feature_extraction.extract(template_image, test_image_45, True, extracted_angle_45)
    result_90_angle = feature_extraction.extract(template_image, test_image_90, True, extracted_angle_90)
    result_180_angle = feature_extraction.extract(template_image, test_image_180, True, extracted_angle_180)
    utils.showim([result_45_angle, result_90_angle, result_180_angle],
                 ['Defected PCB 45° (After Improvment)', 'Defected PCB 90° (After Improvment)', 'Defected PCB 180° (After Improvment)'])



main()