from PIL import Image
import pytesseract
import cv2
import numpy as np
import mss

myconfig = r"--psm 8 --oem 3"
# img = cv2.imread("vellum3.jpg")

sct = mss.mss()

def calibrateBar():

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    # cv2.rectangle(img,top_left, bottom_right, 255, 2)




    # img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # template_processed = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    # threshold, img_processed = cv2.threshold(img_processed, 150, 255, cv2.THRESH_BINARY )

    # img_processed = cv2.Canny(img_processed, 50, 150, apertureSize= 5, L2gradient=True)
    # img_processed = cv2.medianBlur(img_processed, 1)


    #post processing options not used
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
    dims = {
        "left": 0,
        "top": 0,
        "width": 2560,
        "height": 1440
    }
    scsht = sct.grab(dims)
    scr = np.ascontiguousarray(scsht, dtype=np.uint8)
    scr_remove = scr[:,:,:3]
    # img = cv2.imread("vhilla.jpg")
    template = cv2.imread("vhillaTemplateFull.jpg")
    threshold = .90

    #-------Template matching
    res = cv2.matchTemplate(scr_remove.astype(np.uint8).copy(), template, cv2.TM_CCOEFF_NORMED)
    # res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)

    # min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    w = template.shape[1]
    h = template.shape[0]
    # result = cv2.rectangle(img, max_loc, (max_loc[0] + w, max_loc[1] + h), (255,255,0), 2)

    yloc, xloc = np.where(res >= threshold)
    print(yloc, xloc)

    rectangles = []
    for (x,y) in zip(xloc, yloc):
        rectangles.append([int(x), int(y), int(w), int(h)])
        rectangles.append([int(x), int(y), int(w), int(h)])
    

    rectangles, weights = cv2.groupRectangles(rectangles, 1, 0.2)



    for (x, y, w, h) in rectangles:
        cv2.rectangle(scr_remove, (x, y), (x + w, y + h), (0,255,255), 2)
    
    # boss portrait is stored in rectangles as indecies 0,1 is top left coordinates, and indecies 2,3 are X and Y length
    # turned into dictionary using the following methadology
    #top left = x,y        | [0], [1]
    #top right = x+w, y      | [0]+[2], [1]
    #bot left = x, y+h      | [0], [1]+[3]
    #bot right = x+w, y+h   | [0]+[2], [1]+[3]

    bossPortrait = {
        "topLeft": [rectangles[0][0], rectangles[0][1]],
        "topRight": [(rectangles[0][0] + rectangles[0][2]), rectangles[0][1]],
        "botLeft": [rectangles[0][0], (rectangles[0][1] + rectangles[0][3])],
        "botRight": [(rectangles[0][0] + rectangles[0][2]), (rectangles[0][1] + rectangles[0][3])],
        "healthOffsets": [4, 38]
    }
    # print(bossPortrait["topLeft"], bossPortrait["topRight"], bossPortrait["botLeft"], bossPortrait["botRight"])

    #area of the health text % stored using reference to boss portrait
    healthBox = {
        "topLeft": [bossPortrait["topLeft"][0] + bossPortrait["healthOffsets"][0], bossPortrait["topLeft"][1] + bossPortrait["healthOffsets"][1]],
        "offsets": [28, 10]
    }


    cv2.rectangle(scr_remove, (healthBox["topLeft"]), (healthBox["topLeft"][0] + healthBox["offsets"][0], healthBox["topLeft"][1] + healthBox["offsets"][1]), (0,255,255), 2)

    
    #offsets from boss portrait to text area
    # [830, 163] [863, 163] [830, 196] [863, 196]
    # [834, 201] [862, 201] [834, 211] [862, 211]
    #  +04  +38 | +01  +38 | +04  +15 | -01  +15

    #test using manual offsets for finding location of health text
    # cv2.rectangle(img, (834, 201), (834 + 28, 201 + 10), (0,255,255), 2)


    # print(bossPortrait)
    # botLeftPortrait = bossPortrait[0]
    # botRightPortrait = "(" + str(bossPortrait[0] + bossPortrait[2]) + "," + str(bossPortrait[1] + bossPortrait[3]) + ")"
    # print(botRightPortrait)


    # scale_percent = 120 
    # width = int(result.shape[1] * scale_percent / 100)
    # height = int(result.shape[0] * scale_percent / 100)
    # dim = (width, height)
    # result_processed = cv2.resize(result, dim, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    

    # print(res)
    # cv2.imshow("template_processed", template_processed)
    # cv2.waitKey(0)
    print(healthBox)
    cv2.imshow("res", scr_remove)
    cv2.waitKey(0)
    return healthBox

def getScreenShot(dimensions=[]):
    print(dimensions)
    test = {
        "left": dimensions["topLeft"][0],
        "top": dimensions["topLeft"][1],
        "width": dimensions["offsets"][0],
        "height": dimensions["offsets"][1]
        # "left": 834,
        # "top": 201,
        # "width": 28,
        # "height": 10
    }
    scr = np.array(sct.grab(test))
    # scr_remove = scr[:,:,:3]
    cv2.imshow("screen shot", scr)
    cv2.waitKey(0)

    return


def getCurHP(img):

    img = cv2.imread(img)

    #post processing of captured image to increase OCR accuracy
    scale_percent = 150 
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)

    img = cv2.resize(img, dim, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)

    img_processed = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img_processed = cv2.medianBlur(img_processed, 9)

    threshold, img_processed = cv2.threshold(img_processed, 140, 255, cv2.THRESH_BINARY )


    #post processing options not used
    # kernel = np.ones((1, 1), np.uint8)
    # img = cv2.dilate(img, kernel, iterations=1)
    # img = cv2.erode(img, kernel, iterations=1)
    # img_processed = cv2.Canny(img_processed, 50, 150, apertureSize= 5, L2gradient=True)


    # string output of OCR rather than visualization
    test = pytesseract.image_to_string(img_processed, config=myconfig)
    # print(test)
    return test

    #-----------------
    #visualization of result for development/debugging/testing
    #
    # height, width = img_processed.shape

    # boxes = pytesseract.image_to_boxes(img_processed, config=myconfig)
    # for box in boxes.splitlines():
    #     box = box.split(" ")
    #     print(box)
    #     img_processed = cv2.rectangle(img_processed, (int(box[1]), height - int(box[2])), (int(box[3]), height-int(box[4])), (0,255,0), 2)

    # cv2.imshow("img", img_processed)
    # cv2.waitKey(0)

print(calibrateBar())

# dimensions = calibrateBar()
# getScreenShot(dimensions)

# getCurHP("vellum4.jpg")
