"""
@file : transforms.py
@author : s.aparajith@live.com
@date : 4.5.2021
@license : MIT
"""
import numpy as np
import preprocessing
import cv2
'''
CNNs are translation invariant but are not rotation, shear invariant, 
this means there can be situations where images are slightly tilted or skewed that the CNNs will fail to pick, 
sensible tilt values +/-5 degrees was tried.  
references : 
https://jvgemert.github.io/pub/kayhanCVPR20translationInvarianceCNN.pdf
https://stackoverflow.com/questions/40952163/are-modern-cnn-convolutional-neural-network-as-detectnet-rotate-invariant/40953261#40953261
'''


def transform(img,
              a_range,
              s_range):
    """
    @brief takes in an image and does some transforms on it.
    @param : Image
    @param : a_range: Range of angles for rotation in degrees
    @param : s_range: Range of values to apply affine transform to
    A Random uniform distribution is used to generate different parameters for transformation
    """
    # Rotation
    ang_rot = np.random.uniform(a_range) - a_range / 2
    rows, cols, ch = img.shape
    Rot_M = cv2.getRotationMatrix2D((cols / 2, rows / 2), ang_rot, 1)

    # Shear
    pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
    pt1 = 5 + s_range * np.random.uniform() - s_range / 2
    pt2 = 20 + s_range * np.random.uniform() - s_range / 2
    pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
    shear_M = cv2.getAffineTransform(pts1, pts2)

    img = cv2.warpAffine(img, Rot_M, (cols, rows))
    img = cv2.warpAffine(img, shear_M, (cols, rows))
    return img


def augmentData(img, count):
    """
    @param img - input image to be augmented
    @param count - number of images to augment
    @return list of augmented images
    """
    augmentedImages = []
    for i in range(count):
        augmentedImages.append(transform(img, 5, 1))
    return augmentedImages

# Playground to test this package
if __name__ == "__main__":

    '''
    for i in range(100):
        augimg = augmentData(image,1)
        cv2.imshow('sampleAugment',augimg[0])
        cv2.waitKey(100)
    '''
    import glob
    files = glob.glob("TEST/*.jpg")
    for file in files:
        image = cv2.imread(filename=file)
        image2=np.zeros(image.shape , dtype='float32')
        cv2.normalize(src=image.astype("float32"),
                      dst=image2,
                      norm_type=cv2.NORM_MINMAX)
        image2[59:129, 40:280, : ] = cv2.cvtColor(preprocessing.preprocess(image), cv2.COLOR_YUV2BGR)
        cv2.imshow('img and ROI', image2)
        cv2.waitKey(100)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
