import numpy as np
import cv2
from matplotlib import pyplot as plt

# FOR IMAGE
'''
#importing an image from video
img = cv2.imread('road_lane_img.jpg')

print(img.shape)

#converting img to graysacle
imgr = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

#edge detection:
canny = cv2.Canny(imgr, 80, 150)

#creating ROI
poly = np.array([[(482,704),(989,691),(746,600),(579,600)]])
mask = np.zeros_like(canny)
fillpoly = cv2.fillPoly(mask,poly,255)
msk_img = cv2.bitwise_and(canny,fillpoly)

#Hough Transform:
line = cv2.HoughLinesP(msk_img,rho=1,theta=np.pi/180,threshold=50,maxLineGap=900)
if line is not None:
    for lines in line:
        x1, y1, x2, y2 = lines[0]
        cv2.line(img, (x1, y1), (x2, y2), (100, 200, 0), 5)

cv2.imshow('result',img)
cv2.imshow('grayscale',imgr)
cv2.imshow('canny edge deection', canny)
cv2.imshow('roi',msk_img)
plt.imshow(img)
plt.show()

cv2.waitKey(0)
cv2.destroyAllWindows()
'''
'''
vid = cv2.VideoCapture('road_lane.mp4')


while True:
    ret,frame = vid.read()
    #frame = cv2.resize(frame, (970, 543))

    if not ret:
        vid = cv2.VideoCapture("road_lane.mp4")
        continue

    #convert to graysacle
    frame_gry = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    #edge detection
    canny = cv2.Canny(frame_gry,80,180)

    #creating a ROI
    polygon = np.array([[(400,694),(989,694),(746,600),(579,600)]])
    mask = np.zeros_like(frame_gry)
    fillpoly = cv2.fillPoly(mask,polygon,255)
    mask_vid = cv2.bitwise_and(canny,fillpoly)

    #Hough Transforms
    line = cv2.HoughLinesP(mask_vid, rho=1, theta=np.pi / 180, threshold=20, maxLineGap=400)
    if line is not None:
        for lines in line:
            x1, y1, x2, y2 = lines[0]
            cv2.line(frame, (x1, y1), (x2, y2), (100, 200, 0), 3)

    cv2.imshow('result', frame)
    cv2.imshow('grayscale', frame_gry)
    cv2.imshow('canny edge deection', canny)
    cv2.imshow('roi', mask_vid)
    #plt.imshow(img)
    #plt.show()

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("video ended")
        break
cv2.destroyAllWindows()

'''

# FOR A VIDEO

class Lane_Detection:

    def __init__(self, video_path):
        self.vid = cv2.VideoCapture(video_path)

    def videocapture(self):

        while True:
            ret, frame = self.vid.read()
            # frame = cv2.resize(frame, (970, 543))

            if not ret:
                self.vid = cv2.VideoCapture('road_lane.mp4')
                continue

            return frame

    def edgedet(self,frame):
            # convert to graysacle
            frame_gry = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # edge detection
            canny = cv2.Canny(frame_gry, 80, 180)

            return frame_gry,canny

    def Roi(self,frame_gry,canny):
            # creating a ROI
            polygon = np.array([[(400, 694), (989, 694), (746, 600), (579, 600)]])
            mask = np.zeros_like(frame_gry)
            fillpoly = cv2.fillPoly(mask, polygon, 255)
            mask_vid = cv2.bitwise_and(canny, fillpoly)

            return mask_vid

    def hough_transform(self,mask_vid,frame):

            line = cv2.HoughLinesP(mask_vid, rho=1, theta=np.pi / 180, threshold=20, maxLineGap=400)
            if line is not None:
                for lines in line:
                    x1, y1, x2, y2 = lines[0]
                    cv2.line(frame, (x1, y1), (x2, y2), (100, 200, 0), 3)

            return frame

videprocess = Lane_Detection('road_lane.mp4')

frame_width = int(videprocess.vid.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(videprocess.vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(videprocess.vid.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter('recorded_lane_detection.mp4', fourcc, fps, (frame_width, frame_height))


while True:
    frame = videprocess.videocapture()
    if frame is None:
        print('video ended')
        break

    frame_gry, edge_detect = videprocess.edgedet(frame)
    roi = videprocess.Roi(frame_gry, edge_detect)
    res_frame = videprocess.hough_transform(roi, frame)

    cv2.imshow('Lane Detection', res_frame)
    out_video.write(frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
         print('video closed')
         break

videprocess.vid.release()
cv2.destroyAllWindows()

