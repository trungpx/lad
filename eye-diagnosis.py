import tkinter
import numpy as np
import PIL.Image, PIL.ImageTk
import time

import cv2
import tkinter as tk
from tkinter import *


white 		= "#ffffff"
lightBlue2 	= "#adc5ed"
font 		= "Constantia"
fontButtons = (font, 12)
maxWidth  	= 800
maxHeight 	= 480

#Graphics window
mainWindow = tk.Tk()
mainWindow.configure(bg=lightBlue2)
mainWindow.geometry('%dx%d+%d+%d' % (maxWidth,maxHeight,0,0))
mainWindow.resizable(0,0)
# mainWindow.overrideredirect(1)

mainFrame = Frame(mainWindow)
mainFrame.place(x=20, y=20)                

#Capture video frames
lmain = tk.Label(mainFrame)
lmain.grid(row=0, column=0)

cap = cv2.VideoCapture('test.avi')

def show_frame():
	ret, frame = cap.read()

	cv2image   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)

	img   = PIL.Image.fromarray(cv2image) #.resize((320, 240)) #760, 400
	imgtk = PIL.ImageTk.PhotoImage(image = img)
	lmain.imgtk = imgtk
	lmain.configure(image=imgtk)
	lmain.after(1, show_frame)

closeButton = Button(mainWindow, text = "CLOSE", font = fontButtons, bg = white, width = 20, height= 1)
closeButton.configure(command= lambda: mainWindow.destroy())              
closeButton.place(x=270,y=430)	

show_frame()  #Display
mainWindow.mainloop() #Starts GUI



'''
class App:
    def __init__(self, window, window_title, video_source=0):
        self.window = window
        self.window.title(window_title)
        self.video_source = video_source
       # open video source (by default this will try to open the computer webcam)
        self.vid = MyVideoCapture(self.video_source)
        # Create a canvas that can fit the above video source size
        self.canvas = tkinter.Canvas(window, width = self.vid.width, height = self.vid.height)
        self.canvas.pack()
        # Button that lets the user take a snapshot
        self.btn_snapshot=tkinter.Button(window, text="Snapshot", width=50, command=self.snapshot)
        self.btn_snapshot.pack(anchor=tkinter.CENTER, expand=True)
        # After it is called once, the update method will be automatically called every delay milliseconds
        self.delay = 15
        self.update()
        self.window.mainloop()
    def snapshot(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            cv2.imwrite("frame-" + time.strftime("%d-%m-%Y-%H-%M-%S") + ".jpg", cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
    def update(self):
        # Get a frame from the video source
        ret, frame = self.vid.get_frame()
        if ret:
            self.photo = PIL.ImageTk.PhotoImage(image = PIL.Image.fromarray(frame))
            self.canvas.create_image(0, 0, image = self.photo, anchor = tkinter.NW)
        self.window.after(self.delay, self.update)
class MyVideoCapture:
    def __init__(self, video_source=0): # 0
        # Open the video source
        self.vid = cv2.VideoCapture('test.avi') #video_source
        if not self.vid.isOpened():
            raise ValueError("Unable to open video source", video_source)
        # Get video source width and height
        self.width = self.vid.get(cv2.CAP_PROP_FRAME_WIDTH)
        self.height = self.vid.get(cv2.CAP_PROP_FRAME_HEIGHT)
    def get_frame(self):
        if self.vid.isOpened():
            ret, frame = self.vid.read()
            if ret:
                # Return a boolean success flag and the current frame converted to BGR
                return (ret, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            else:
                return (ret, None)
        else:
            return (ret, None)
    # Release the video source when the object is destroyed
    def __del__(self):
        if self.vid.isOpened():
            self.vid.release()
# Create a window and pass it to the Application object
App(tkinter.Tk(), "Tkinter and OpenCV")
'''

'''

root = tkinter.Tk()
root.title("Hello!")
#root.resizable(width="false", height="false")
#root.minsize(width=300, height=50)  
#root.maxsize(width=300, height=50)

simple_label = tkinter.Label(root, text="Easy, right?")  
another_label = tkinter.Label(root, text="More text")  
closing_button = tkinter.Button(root, text="Close window", command=root.destroy)  
another_button = tkinter.Button(root, text="Do nothing")
simple_label.grid(column=0, row=0, sticky="ew")  
another_label.grid(column=0, row=1, sticky="ew")  
closing_button.grid(column=1, row=0, sticky="ew")  
another_button.grid(column=1, row=1, sticky="ew")

#simple_label.pack()  
#closing_button.pack()

root.mainloop()
'''