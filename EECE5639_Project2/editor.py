from random import random
from tkinter import messagebox
from copy import deepcopy
from PIL import Image, ImageTk
import cv2
import numpy as np
import tkinter as tk
import p2_main
import os

fp = os.path.dirname(__file__) 

sourceim = "DanaOffice/DSC_0308.JPG"

addim = "pika.jpg"

root = tk.Tk()

image = cv2.imread(f"{fp}/{sourceim}")
imagergb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
imagepil = Image.fromarray(imagergb)
imagetk = ImageTk.PhotoImage(imagepil)
panelIm = tk.Label(image=imagetk)

image2 = cv2.imread(f"{fp}/{addim}")
imagergb2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
imagepil2 = Image.fromarray(imagergb2)
print(image2.shape)
if image2.shape[1] > 400:
    imagepil2 = imagepil2.reduce(int(np.ceil(image2.shape[0] / 400)))
imagetk2 = ImageTk.PhotoImage(imagepil2)
panelImAdd = tk.Label(image=imagetk2)

dirvar = tk.StringVar(master=root, value="top-left in dest:")
dir = tk.Label(root, textvariable=dirvar)

points = []


def onClick(e):
    points.append((e.y, e.x))  # row,col
    if len(points) == 1:
        dirvar.set("top-right in dest:")
    elif len(points) == 2:
        dirvar.set("bottom-left in dest")
    elif len(points) == 3:
        dirvar.set("bottom-right in dest")
    elif len(points) == 4:
        dirvar.set("calculating...")
        # apply homography

        (imrows, imcols, _) = image2.shape
        pointpairs = \
            [((0, 0), points[0]),
             ((0, imcols), points[1]),
             ((imrows, 0), points[2]),
             ((imrows, imcols), points[3])]

        hg = p2_main.getHomography(*pointpairs)

        newimg = deepcopy(imagergb)
        for pr in range(0, imrows):
            for pc in range(0, imcols):
                (destr, destc, _) = p2_main.homogenize(np.matmul(hg, np.transpose(np.asarray([pr, pc, 1]))))
                newimg[int(destr), int(destc), :] = imagergb2[pr, pc, :]
        pilimage = Image.fromarray(newimg)
        tkim = ImageTk.PhotoImage(pilimage)
        panelIm.configure(image=tkim)
        panelIm.image = tkim
        dirvar.set("done!")
        name = "applied_%s.jpg" % (int(100*random()))
        pilimage.save(f"{fp}/insert/{name}", "JPEG")
        messagebox.showinfo("saved", "saved to %s!" % name)


panelIm.pack(side='left', padx=10, pady=10)
panelImAdd.pack(side='right', padx=10, pady=10)
dir.pack(side='bottom', pady=5)
panelIm.bind("<Button-1>", onClick)

if __name__ == '__main__':
    root.mainloop()
