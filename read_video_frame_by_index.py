# This file is just to show how to access the specific frame in video with cv2
import cv2
import ipdb

print('Program started!')
myFrameNumber = 50
cap = cv2.VideoCapture('predicted_video.avi') # Class1_left_000006.avi

# get total number of frames
totalFrames = cap.get(cv2.CAP_PROP_FRAME_COUNT)

# check for valid frame number
if myFrameNumber >= 0 & myFrameNumber <= totalFrames:
    # set frame position
    cap.set(cv2.CAP_PROP_POS_FRAMES,myFrameNumber)
#ipdb.set_trace()

while True:
    ret, frame = cap.read()
    current_frame = cap.get(cv2.CAP_PROP_POS_FRAMES)
    text = 'Frame: {}'.format(current_frame)
    print(text)
    cv2.putText(frame,text,(10,20),cv2.FONT_HERSHEY_SIMPLEX,.5,(0, 250, 0),1)
    cv2.imshow("Video", frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break
    
    ipdb.set_trace()
