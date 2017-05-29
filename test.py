import cv2
import numpy as np


def overlay_image(foreground, background, rect):
    """
    overlay the foreground image on the background image
    at the specified location
    :param foreground: the foreground image, numpy array
    :param background: the background image, numpy array
    :param rect: a rectangle indicating where to overlay
    :return: overlaid image
    """
    row, col = foreground.shape[0:2]
    x, y, w, h = rect
    xc, yc = x + w/2, y + h/2
    ratio_x, ratio_y = w/float(col), h/float(row)
    ratio = max(ratio_x, ratio_y)
    row, col = int(ratio*row) & 0xfffe, int(ratio*col) & 0xfffe
    resized = cv2.resize(foreground, dsize=(col, row))
    ret = background.copy()
    ymin = max(yc-row/2, 0)
    ymax = min(yc+row/2, ret.shape[0])
    xmin = max(xc-col/2, 0)
    xmax = min(xc+col/2, ret.shape[1])
    size_x, size_y = xmax-xmin, ymax-ymin
    xo, yo = xmin-(xc-col/2), ymin-(yc-row/2)
    ret[ymin:ymax, xmin:xmax, :] = resized[yo:size_y, xo:size_x, :]
    return ret



face = cv2.imread('smiling_face.jpg')
face_detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture()
for i in range(10):
    if cap.open(i):
        print 'camera {} launched'.format(i)
        break
key = None
face_location = None
while key != ord('x'):
    status, image = cap.read()
    assert status, 'failed to grab image from camera'
    # convert color image to grayscale image
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gray, 1.1, 10)
    if len(faces) != 1:
        # use the previous detection result
        pass
    else:
        face_location = faces[0]
    if face_location is not None:
        x, y, w, h = face_location
#         image = cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
        image = overlay_image(foreground=face, background=image, rect=face_location)
    cv2.imshow('display', image)
    key = cv2.waitKey(30)
cv2.destroyWindow('display')
cap.release()
