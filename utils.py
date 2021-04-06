"""
@file : utils.py
@author : s.aparajith@live.com
@date : 4.5.2021
@license : MIT
@details : contains utilities to view images, debug models etc.
"""
from cv2 import imshow,cvtColor,COLOR_YUV2BGR,waitKey,destroyAllWindows


def playSome(imgs, CameraSide):
    """
        @brief play some images.
        @param imgs list of images
        @param CameraSide : 0 - center, 1 - left , 2 - right
    """
    for index in range(CameraSide, imgs.shape[0], 3):
        imshow('op', cvtColor(imgs[index], COLOR_YUV2BGR))
        waitKey(100)
    waitKey(0)
    destroyAllWindows()
