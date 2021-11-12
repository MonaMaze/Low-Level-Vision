from tkinter import *
from tkinter import filedialog
from PIL import ImageTk , Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import imutils
import easyocr

def select_image():
    global panelA, panelB, f
    path = filedialog.askopenfilename()
    f.write("The card: " + path + "\n")
    
    img = cv2.imread(path)
    h = int(img.shape[0] * 650 / img.shape[1])
    img = cv2.resize(img,(650, h),interpolation = cv2.INTER_LINEAR)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    ret,thresh = cv2.threshold(gray,55,255,cv2.THRESH_BINARY_INV)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    reader = easyocr.Reader(['ar','en'])
    results = reader.readtext(gray)
    for (bbox, text, prob) in results:
        # display the OCR'd text and associated probability
        print("[INFO] {:.4f}: {}".format(prob, text))
        f.write(text + "\n")
        # unpack the bounding box
        (tl, tr, br, bl) = bbox
        tl = (int(tl[0]), int(tl[1]))
        tr = (int(tr[0]), int(tr[1]))
        br = (int(br[0]), int(br[1]))
        bl = (int(bl[0]), int(bl[1]))
        # cleanup the text and draw the box surrounding the text along
        # with the OCR'd text itself
        #text = cleanup_text(text)
        text = "".join([c if ord(c) < 400 else "" for c in text]).strip()
        res = cv2.rectangle(img1, tl, br, (0, 255, 0), 2)
        res = cv2.putText(img1, text, (tl[0], tl[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    f.write("\n")

    img = Image.fromarray(img)
    res = Image.fromarray(res)
    img = ImageTk.PhotoImage(img)
    res = ImageTk.PhotoImage(res)
    
    if panelA is None or panelB is None:
        panelA = Label(image=img)
        panelA.image = img
        panelA.pack(side='left', padx=10, pady=10)
        panelB = Label(image=res)
        panelB.image = res
        panelB.pack(side='right', padx=10, pady=10)
    else:
        panelA.configure(image=img)
        panelA.image = img
        panelB.configure(image=res)
        panelB.image = res
        
root = Tk()
f = open("cards.txt","a+", encoding='utf-8')
panelA = None
panelB = None
btn = Button(root, text='Select a card image', command=select_image)
btn.pack(side='bottom', fill='both', expand='yes', padx=10, pady=10)
root.mainloop()
