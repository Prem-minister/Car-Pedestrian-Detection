import cv2
#car image and video
img_file = 'car_image.PNG'
#car_video = cv2.VideoCapture('Tesla_Autopilot_Dashcam.mp4')
car_video = cv2.VideoCapture('Pedestrians_Compilation.mp4')

#pretrained car classifier
car_tracker_file ='cars.xml'
pedestrian_tracker_file = 'haarcascade_fullbody.xml'


#car classifier
car_tracker = cv2.CascadeClassifier(car_tracker_file)
pedestrian_tracker = cv2.CascadeClassifier(pedestrian_tracker_file)

#run forever
while True:
    #Read the current frame
    (read_successful,  frame) = car_video.read()
     
     #safe coding
    if read_successful:
        #must concert to grayscale
        grayscaled_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        break

    #detect cars and pedestrians
    cars = car_tracker.detectMultiScale(grayscaled_frame)
    pedestrians = pedestrian_tracker.detectMultiScale(grayscaled_frame)

    #draw rectangles around the cars detected
    for(x, y, w, h) in cars:
        cv2.rectangle(frame, (x+2, y+2), (x+w, y+h), (255, 0, 0), 2)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0,0,255), 2)

    for(x, y, w, h) in pedestrians:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 255), 2)



    #display the image with faces spotted
    cv2.imshow("Lucy's Car Detector", frame)

    #Dont autoclose (wait here in the code and listen for a key press)
    key = cv2.waitKey(1)

    #stop if q key is pressed
    if key==81 or key==113:
        break

#Relese the VideoCapture obj
car_video.release()

"""
#create opencv image
img = cv2.imread(img_file)

#car classifier
car_tracker = cv2.CascadeClassifier(classifier_file)

#converts to grayscale (needed for haar cascade)
black_and_white = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#detect cars
cars = car_tracker.detectMultiScale(black_and_white)

#draw rectangles around the cars detected
for(x, y, w, h) in cars:
    cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)


#display the image with faces spotted
cv2.imshow("Lucy's Car Detector", img)

#Dont autoclose (wait here in the code and listen for a key press)
cv2.waitKey()

"""

print("Code end")