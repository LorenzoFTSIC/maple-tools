from PIL import Image
import pytesseract
import cv2
import numpy as np



# myconfig = r"--psm 6 --oem 3"
myconfig = r"--psm 8 --oem 3"

# text = pytesseract.image_to_string(Image.open("vellum3.jpg"), config=myconfig)

img = cv2.imread("vellum3.jpg")

scale_percent = 150 
width = int(img.shape[1] * scale_percent / 100)
height = int(img.shape[0] * scale_percent / 100)
dim = (width, height)

img = cv2.resize(img, dim, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)


img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_processed = cv2.medianBlur(img_processed, 9)

# kernel = np.ones((1, 1), np.uint8)
# img = cv2.dilate(img, kernel, iterations=1)
# img = cv2.erode(img, kernel, iterations=1)

threshold, img_processed = cv2.threshold(img_processed, 140, 255, cv2.THRESH_BINARY )

# img_processed = cv2.Canny(img_processed, 50, 150, apertureSize= 5, L2gradient=True)


height, width = img_processed.shape

boxes = pytesseract.image_to_boxes(img_processed, config=myconfig, )
for box in boxes.splitlines():
    box = box.split(" ")
    print(box)
    img_processed = cv2.rectangle(img_processed, (int(box[1]), height - int(box[2])), (int(box[3]), height-int(box[4])), (0,255,0), 2)

cv2.imshow("img", img_processed)
cv2.waitKey(0)
# print(boxes)
