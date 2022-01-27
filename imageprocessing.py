# Build Tkinter APP With the ability to be Complete Image Processing Control Panel with ability to 
from tkinter import *
from tkinter import ttk
from ttkbootstrap import Style
from tkinter import filedialog
from PIL import ImageTk , Image
import cv2
import numpy as np
import imutils
import easyocr

def load_image():
    global panelA, panelB
    img1 = cv2.imread('panel_a.jpg')
    img2 = cv2.imread('panel_b.jpg')
    image1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
    image2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
    
    image1 = Image.fromarray(image1)  # convert the images to PIL format...
    image2 = Image.fromarray(image2)
    
    image1 = ImageTk.PhotoImage(image1)  # ...and then to tkinter format
    image2 = ImageTk.PhotoImage(image2)
    
    # if the panels are None, initialize them
    if panelA is None or panelB is None:
        panelA = Label(image=image1)
        panelA.image = image1
        panelA.pack(side='left', padx=10, pady=10)
        panelB = Label(image=image2)
        panelB.image = image2
        panelB.pack(side='right', padx=10, pady=10)

def select_image():
    global panelA, panelB, img
    
    path = filedialog.askopenfilename()
    img = cv2.imread(path)
    h = int(img.shape[0] * 500 / img.shape[1])
    img = cv2.resize(img,(500, h),interpolation = cv2.INTER_LINEAR)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    image = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    edged = gray
    
    image = Image.fromarray(image)  # convert the images to PIL format...
    edged = Image.fromarray(edged)
    
    image = ImageTk.PhotoImage(image)  # ...and then to tkinter format
    edged = ImageTk.PhotoImage(edged)
    
    # if the panels are None, initialize them
    if panelA is None or panelB is None:
        panelA = Label(image=image)
        panelA.image = image
        panelA.pack(side='left', padx=10, pady=10)
        panelB = Label(image=edged)
        panelB.image = edged
        panelB.pack(side='right', padx=10, pady=10)
    else:
        panelA.configure(image=image)
        panelA.image = image
        panelB.configure(image=edged)
        panelB.image = edged
        
def apply_trans():
    global img
    list_color = color.get().split()
    lower = np.array(list(map(int, list_color[:3])))
    upper = np.array(list(map(int, list_color[3:])))
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower, upper)
    img1 = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    if sum(lower) != 0:
        edged = cv2.bitwise_and(img, img, mask=mask)
        edged = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
    
    thresh_val = thresh.get()
    blur_val = blurr.get()
    morph_val = morph.get()
    edge_val = edge.get()
    dir_val = dirr.get()
    lower1 = lower_slide.get()
    upper1 = upper_slide.get()
    lower2 = lower_slid.get()
    upper2 = upper_slid.get()
    txt_val = txt.get()
    if sum(lower) == 0:
        edged = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kernel = np.ones((5,5),np.uint8)
        
        if thresh_val == 1:
            _, edged = cv2.threshold(edged,lower2,upper2,cv2.THRESH_BINARY)
        elif thresh_val == 2:
            edged = cv2.adaptiveThreshold(edged,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        elif thresh_val ==3:
            _, edged = cv2.threshold(edged,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
            
        if blur_val == 1:
            edged = cv2.blur(edged,(5,5))
        elif blur_val == 2:
            edged = blur = cv2.GaussianBlur(edged,(5,5),0)
        elif blur_val == 3:
            edged = cv2.medianBlur(edged,5)
            
        if morph_val == 1:
            edged = cv2.erode(edged, kernel, iterations = 1)
        elif morph_val == 2:
            edged = cv2.dilate(edged, kernel, iterations = 1)
        elif morph_val == 3:
            edged = cv2.morphologyEx(edged, cv2.MORPH_OPEN, kernel)
        elif morph_val == 4:
            edged = cv2.morphologyEx(edged, cv2.MORPH_CLOSE, kernel)
            
        if dir_val == 1:
            if edge_val == 1:
                edged = cv2.Sobel(edged,-1,1,0,ksize=5)
            elif edge_val == 2:
                edged = cv2.Scharr(edged,cv2.CV_64F,1,0,5)
            elif edge_val == 3:
                edged = cv2.Laplacian(edged,cv2.CV_64F,ksize=3)
            elif edge_val == 4:
                edged = cv2.Canny(edged, lower1, upper1)
        else:
            if edge_val == 1:
                edged = cv2.Sobel(edged,-1,0,1,ksize=5)
            elif edge_val == 2:
                edged = cv2.Scharr(edged,cv2.CV_64F,0,1,5)
            elif edge_val == 3:
                edged = cv2.Laplacian(edged,cv2.CV_64F,ksize=3)
            elif edge_val == 4:
                edged = cv2.Canny(edged, lower_, upper_)
        
        if txt_val == 1:
            reader = easyocr.Reader(['ar','en'])
            edged = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = reader.readtext(edged)
            for (bbox, text, prob) in results:
                print("[INFO] {:.4f}: {}".format(prob, text))
                # unpack the bounding box
                (tl, tr, br, bl) = bbox
                tl = (int(tl[0]), int(tl[1]))
                tr = (int(tr[0]), int(tr[1]))
                br = (int(br[0]), int(br[1]))
                bl = (int(bl[0]), int(bl[1]))
                # cleanup the text and draw the box surrounding the text along
                # with the OCR'd text itself
                text = "".join([c if ord(c) < 400 else "" for c in text]).strip()
                edged = cv2.rectangle(img1, tl, br, (0, 255, 0), 2)
                edged = cv2.putText(img1, text, (tl[0], tl[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
    
    edged = Image.fromarray(edged)
    edged = ImageTk.PhotoImage(edged)
    panelB.configure(image=edged)
    panelB.image = edged
    
# initialize the window toolkit along with the two image panels
style = Style(theme='minty')
root = style.master
root.title("Image Processing")
root.geometry("+1+50")
root.wait_visibility()
root.after_idle(load_image)
#logo_left = ImageTk.PhotoImage(file="panel_a.jpg")
#logo_right = ImageTk.PhotoImage(file="panel_b.jpg")
panelA = None   #ttk.Label(root, image=logo_left).pack(side="left", padx=10, pady=10)
panelB = None   #ttk.Label(root, image=logo_right).pack(side="left", padx=10, pady=10)

# Button to Upload image
btn1 = ttk.Button(root, text = 'Upload Photo', style='primary.TButton', command=select_image)
btn1.pack(side='top', fill='both', padx=10, pady=10)

left = ttk.LabelFrame(root)
left.pack(side='left', padx=10, pady=10)
# Radio Buttons for Red , Green , Blue Color Tracking or None 
label_color = ttk.LabelFrame(left, text='Color tracking', width=250)
label_color.pack(side='top', padx=10, pady=10)
color = StringVar(value='0 0 0 255 255 255')
color1 = ttk.Radiobutton(label_color, text='Red', variable=color, value='0 50 50 20 255 255')
color1.pack(anchor = W)
color2 = ttk.Radiobutton(label_color, text='Green', variable=color, value='30 50 50 90 255 255')
color2.pack(anchor = W)
color3 = ttk.Radiobutton(label_color, text='Blue', variable=color, value='100 50 50 180 255 255')
color3.pack(anchor = W)
color4 = ttk.Radiobutton(label_color, text='None', variable=color, value='0 0 0 255 255 255')
color4.pack(anchor = W)

# Binary Threshold Canny scales
upper_slid = ttk.Scale(left, from_=170, to=255, orient=HORIZONTAL)
upper_slid.set(255)
upper_slid.pack(side="bottom", padx=10, pady=10)
lower_slid = ttk.Scale(left, from_=20, to=170, orient=HORIZONTAL)
lower_slid.set(170)
lower_slid.pack(side="bottom", padx=10, pady=10)

# Radio Buttons for Binary ,Adaptive , Otsu  Threhsolding or None
label_thresh = ttk.LabelFrame(left, text='Thresholding    ', width=250)
label_thresh.pack(side='bottom', padx=10, pady=10)
thresh = IntVar(value=0)
thresh1 = ttk.Radiobutton(label_thresh, text='Binary', variable=thresh, value=1)
thresh1.pack(anchor = W)
thresh2 = ttk.Radiobutton(label_thresh, text='Adaptive', variable=thresh, value=2)
thresh2.pack(anchor = W)
thresh3 = ttk.Radiobutton(label_thresh, text='OTSU', variable=thresh, value=3)
thresh3.pack(anchor = W)
thresh4 = ttk.Radiobutton(label_thresh, text='None', variable=thresh, value=0)
thresh4.pack(anchor = W)

# Radio Buttons for Averaging , Gaussain , Median Blurring or None 
label_blur = ttk.LabelFrame(left, text='Image Blurring', width=250)
label_blur.pack(side='bottom', padx=10, pady=10)
blurr = IntVar(value=0)
blur1 = ttk.Radiobutton(label_blur, text='Averaging', variable=blurr, value=1)
blur1.pack(anchor = W)
blur2 = ttk.Radiobutton(label_blur, text='Gaussain', variable=blurr, value=2)
blur2.pack(anchor = W)
blur3 = ttk.Radiobutton(label_blur, text='Median', variable=blurr, value=3)
blur3.pack(anchor = W)
blur4 = ttk.Radiobutton(label_blur, text='None', variable=blurr, value=0)
blur4.pack(anchor = W)

# Radio Buttons for Erosion , Dilation , Openning , Closing or None
label_morph = ttk.LabelFrame(left, text='Morphology    ', width=250)
label_morph.pack(side='bottom', padx=10, pady=10)
morph = IntVar(value=0)
morph1 = ttk.Radiobutton(label_morph, text='Erosion', variable=morph, value=1)
morph1.pack(anchor = W)
morph2 = ttk.Radiobutton(label_morph, text='Dilation', variable=morph, value=2)
morph2.pack(anchor = W)
morph3 = ttk.Radiobutton(label_morph, text='Openning', variable=morph, value=3)
morph3.pack(anchor = W)
morph4 = ttk.Radiobutton(label_morph, text='Closing', variable=morph, value=4)
morph4.pack(anchor = W)
morph5 = ttk.Radiobutton(label_morph, text='None', variable=morph, value=0)
morph5.pack(anchor = W)

right = ttk.LabelFrame(root)
right.pack(side='right', padx=10, pady=10)
# Radio Buttons for Sobel , Scharr , Laplacian ,Canny or None
label_edge = ttk.LabelFrame(right, text='Edge Detection', width=250)
label_edge.pack(side='top', padx=10, pady=10)
edge = IntVar(value=0)
edge1 = ttk.Radiobutton(label_edge, text='Sobel', variable=edge, value=1)
edge1.pack(anchor = W)
edge2 = ttk.Radiobutton(label_edge, text='Scharr', variable=edge, value=2)
edge2.pack(anchor = W)
edge3 = ttk.Radiobutton(label_edge, text='Laplacian', variable=edge, value=3)
edge3.pack(anchor = W)
edge4 = ttk.Radiobutton(label_edge, text='Canny', variable=edge, value=4)
edge4.pack(anchor = W)
edge5 = ttk.Radiobutton(label_edge, text='None', variable=edge, value=0)
edge5.pack(anchor = W)

# Radio Buttons for horizontal or vertical edge detection
label_dirr = ttk.LabelFrame(right, text='Edge Direction', width=250)
label_dirr.pack(side='top', padx=10, pady=10)
dirr = IntVar(value=0)
dirr1 = ttk.Radiobutton(label_dirr, text='Horizontal', variable=dirr, value=1)
dirr1.pack(anchor = W)
dirr2 = ttk.Radiobutton(label_dirr, text='Vertical', variable=dirr, value=0)
dirr2.pack(anchor = W)

# Radio Buttons for reading text by easyocr
label_txt = ttk.LabelFrame(right, text='Text Reading', width=250)
label_txt.pack(side='bottom', padx=10, pady=10)
txt = IntVar(value=0)
txt1 = ttk.Radiobutton(label_txt, text='Text Reading', variable=txt, value=1)
txt1.pack(anchor = W)
txt2 = ttk.Radiobutton(label_txt, text='None', variable=txt, value=0)
txt2.pack(anchor = W)

# Edge detection Canny scales
upper_slide = ttk.Scale(right, from_=70, to=200, orient=HORIZONTAL)
upper_slide.set(100)
upper_slide.pack(side="bottom", padx=10, pady=10)
lower_slide = ttk.Scale(right, from_=20, to=70, orient=HORIZONTAL)
lower_slide.set(25)
lower_slide.pack(side="bottom", padx=10, pady=10)

# Button for Applying Transformation 
btn2 = ttk.Button(root, text = 'Applying Transformation', style='secondary.TButton', command=apply_trans)
btn2.pack(side='bottom', fill='both', padx=10, pady=10)

# kick off the GUI
root.mainloop()
