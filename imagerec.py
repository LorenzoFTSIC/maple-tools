from PIL import Image
import pytesseract

# myconfig = r"--psm 6 --oem 3"
myconfig = r"--oem 3"

text = pytesseract.image_to_string(Image.open("text.jpg"), config=myconfig)
print(text)
