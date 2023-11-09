# Lane-Detection
<br>Lane detection helps in recognising the road lanes. The end goal is to detect lane markings accuratley in any conditions. This feature is very important in ADAS and Autonomous vehicle where it provides real time information of the road and the vehicle's position through which vehicle can safely cruise on the road.

## Aim:
<br>To detect the road lanes in a given video.

## Technologies used:
<br>Python(3.10.8) and OpenCV

## Workflow
<br>Step1: Getting the data from online, in this case it was a video which I downloaded from internet.
<br>Step2: Installing necessary libraries. 
<br>Step3: Importing required libraries into my workspace which is Pycharm IDE.
<br>Step4: Importing the video which was downloaded.
<br>Step5: Converting the frame to grayscale and performing edge detection using Canny Edge Detection. Edge is nothing but a sharp change in brightness and is helpful to detect the outlines in a video or image. And Canny is type of edge detection algorithm which is comparitivley better than other edge detection algorithms such as Sobel, Prewitt etc.
<br>Step5: Creating a Region Of Interset from a frame.
<br>Step6: Applying Hough Transform to detect the lines and drawing a line/boundary on detected lines.
<br>Step7: Concluding the project by saving the recorded video.

## Sources
<br>Check this drive to get the source video and final result
<br>https://drive.google.com/drive/folders/10vNVhgwKTP2DK63gTM5nvqj8fH1gDqEA?usp=sharing
