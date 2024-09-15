from PIL import Image
# import pytesseract
import cv2
import numpy as np
# import mss

# from imagerec import getScreenShot


myconfig = r"--psm 8 --oem 3"

# img = getScreenShot(calibratedDims)



dims = {
    "left": 0,
    "top": 0,
    "width": 2560,
    "height": 1440,
}
img = cv2.imread('runeFull.jpg')
# scr = np.array(sct.grab(dims))
# scr_remove = scr[:,:,:3]


# threshold = .90


# res = cv2.matchTemplate(scr_remove, template, cv2.TM_CCOEFF_NORMED)


# w = template.shape[1]
# h = template.shape[0]



# yloc, xloc = np.where(res >= threshold)

# rectangles = []
# for (x,y) in zip(xloc, yloc):
#     rectangles.append([int(x), int(y), int(w), int(h)])
#     rectangles.append([int(x), int(y), int(w), int(h)])


# rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)


# for (x, y, w, h) in rectangles:
#     cv2.rectangle(scr, (x, y), (x + w, y + h), (0,255,255), 2)

# width = int(img.shape[1] * scale_percent / 100)
# height = int(img.shape[0] * scale_percent / 100)
# dim = (width, height)

# img = cv2.resize(img, dim, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)



# removes bluechannel
img[:,:,0] = np.zeros([img.shape[0], img.shape[1]])

# img = img[:,:,:3]
# img[np.all(img == (255, 0, 0), axis=-1)] = np.ui#nt8([255,255,255])

img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
threshold, img_processed = cv2.threshold(img, 110, 255, cv2.THRESH_BINARY )

cv2.imshow("Test", img_processed)
cv2.waitKey(0)



# test = pytesseract.image_to_string(img_processed, config=myconfig)