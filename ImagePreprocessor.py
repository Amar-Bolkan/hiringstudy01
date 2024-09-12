import cv2
import numpy as np


class ImagePreprocessor:
    """
    Class implementing the clean up steps described in this article on a received image
    https://tesseract-ocr.github.io/tessdoc/ImproveQuality.html#rotation--deskewing
    """

    def __init__(self):
        pass

    def remove_border(self, img):
        #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        ret, gray = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY)
        mask = np.zeros(gray.shape, dtype=np.uint8)

        #cv2.imshow("gray preprocessor", gray)
        #cv2.imshow("before thrsh", img)
        cnts = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]

        cv2.fillPoly(mask, cnts, [255, 255, 255])
        mask = 255 - mask
        #cv2.imshow("mask", mask)
        result = cv2.bitwise_or(gray, mask)

        #cv2.imshow("result", result)

        cv2.drawContours(gray, cnts, -1, (0, 255, 0), 20)
        #cv2.imshow("contour image", gray)
        cv2.waitKey(0)
        return result



