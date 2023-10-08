import cv2

camera = cv2.VideoCapture(0)
camera.set(3, 660)
camera.set(4, 500)

file = "C:\\Users\\ALFA\\Desktop\\ARJUNA\\face.xml"

facedetector = cv2.CascadeClassifier(file)


while True:
    # Capture frame-by-frame
    retv, frame = camera.read()
    warna = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = facedetector.detectMultiScale(warna, 1.3, 5
    )

    # Draw a rectangle around the facesq
    # print(faces)
    for (x, y, w, h) in faces:        
        frame = cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)        
        #center_coordinates = x + w // 2, y + h // 2
        #radius = w // 2 # or can be h / 2 or can be anything based on your requirements
        #cv2.circle(frame, center_coordinates, radius, (0, 0, 100), 3)
        rec_muka = warna[y : y + h, x :x+w]

    # Display the resulting frame
    cv2.imshow('Video', frame)
    close = cv2.waitKey(1) & 0xff

    if close == 27 or close == ord('n'):
        break

# When everything is done, release the capture
camera.release()
cv2.destroyAllWindows()
