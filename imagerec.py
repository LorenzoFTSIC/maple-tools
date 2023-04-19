from PIL import Image
import pytesseract
import cv2

# myconfig = r"--psm 6 --oem 3"
myconfig = r"--psm 8 --oem 3"

# text = pytesseract.image_to_string(Image.open("vellum3.jpg"), config=myconfig)
# print(text)


img = cv2.imread("vellum3.jpg")
img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
img_processed = cv2.medianBlur(img_processed,3)

height, width = img_processed.shape

boxes = pytesseract.image_to_boxes(img_processed, config=myconfig)
for box in boxes.splitlines():
    box = box.split(" ")
    print(box)
    img_processed = cv2.rectangle(img_processed, (int(box[1]), height - int(box[2])), (int(box[3]), height-int(box[4])), (0,255,0), 2)

cv2.imshow("img", img_processed)
cv2.waitKey(0)
# print(boxes)
