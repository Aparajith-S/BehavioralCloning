"""
@file preprocessing.py
@brief image preprocessing techniques
@author s.aparajith@live.com
@date 23.03.2021
"""
from numpy import reshape, zeros
from cv2 import normalize, NORM_MINMAX
from cv2 import cvtColor, equalizeHist, COLOR_BGR2YUV

# variable to set final image size.
# do not change this unless it is taken care of in the model.py file.
imgShape = (70, 240, 3)


def Normalize(data):
    """
    @brief normalize the data.
    @param data : image data of shape imgShape
    @return data that is normalized
    """
    '''
    for channel in range(3):
        data[:, :, channel] = (data[:, :, channel] - 127.0) / (255.0)
    '''
    op = zeros(imgShape,dtype='float32')
    normalize(src=data,
              dst=op,
              norm_type=NORM_MINMAX)
    return reshape(op.astype('float32'), imgShape)


def histogram_equalize(image):
    """
    @brief convert the image into grayscale.
    @param image : expects a image in np.array of shape (160, 320, 1)
    @return image data that is normalized
    """
    data = cvtColor(image, COLOR_BGR2YUV)
    data[:, :, 0] = equalizeHist(data[:, :, 0])
    return reshape(data, imgShape)


def crop(image, h=(0, -1), w=(0, -1)):
    """
    @param image: input image
    @param h: height to be cropped from
    @param w: width to be cropped from
    @return: cropped image
    """
    return image[h[0]:h[1], w[0]:w[1], :]


def preprocess(img):
    """
    @brief preprocessing of input images of uint8, three channel RGB.
    @return preprocessed image in float
    """
    data = crop(img, h=(59, 129), w=(40, 280))
    data = histogram_equalize(data)
    data = Normalize(data.astype('float32'))
    return reshape(data, imgShape)
