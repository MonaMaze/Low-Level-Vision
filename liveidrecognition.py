from tkinter import *
from tkinter import ttk
from ttkbootstrap import Style
from tkinter import filedialog
from PIL import ImageTk , Image
import cv2
from matplotlib import pyplot as plt
import numpy as np
import easyocr


def select_video():
    global panelA, f
    path = filedialog.askopenfilename()
    f.write("The card: " + path + "\n")
    
    cap = cv2.VideoCapture(0)
    while (cap.isOpened()):
        ret, frame = cap.read()
        img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        reader = easyocr.Reader(['ar','en'])
        results = reader.readtext(img)
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
            text = "".join([c if ord(c) < 400 else "" for c in text]).strip()
            cv2.rectangle(img, tl, br, (0, 255, 0), 2)
            cv2.putText(img, text, (tl[0], tl[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                
            f.write("\n")
            
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            if panelA is None:
                panelA = Label(image=img)
                panelA.image = img
                panelA.pack(side='left', padx=10, pady=10)
            else:
                panelA.configure(image=img)
                panelA.image = img
            
        if cv2.waitKey(1000) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()
    
            
style = Style(theme='minty')
root = style.master
root.title("Image Processing")
root.geometry("+1+50")
root.wait_visibility()
f = open("cards_video.txt","a+", encoding='utf-8')
panelA = None
btn = ttk.Button(root, text='Open the Camera', command=select_video)
btn.pack(side='bottom', fill='both', expand='yes', padx=10, pady=10)
txt1 = "* Click on the bottom to open your camera."
txt2 = "* If you want to close your camera, press 'q'."
lbl1 = ttk.Label(root, text = txt1)
lbl1.pack(side='bottom')
lbl2 = ttk.Label(root, text = txt2)
lbl2.pack(side='bottom')
root.mainloop()
