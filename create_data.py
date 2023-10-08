import cv2 as cv
import os
import time

img_path = "C:\\Users\\ALFA\\Desktop\\ARJUNA\\img"
cameraNo = 0
cameraBrightness = 100
moduleVal = 10
minBlur = 500
grayImage = False
saveData = True
showImage = True
imgWidth = 180
imgHeight = 120



global countFolder
cap = cv.VideoCapture(cameraNo)
cap.set(3, 640)
cap.set(4, 480)
cap.set(10, cameraBrightness)


count = 0
countSave = 0

def saveDataFunc():
    global countFolder
    countFolder = 0
    while os.path.exists(img_path + str(countFolder)):
        countFolder += 1
    os.makedirs(img_path + str(countFolder))

if saveData:
    saveDataFunc()


while True:
    success, img = cap.read()
    img = cv.resize(img, (imgWidth, imgHeight))

    if grayImage:
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    if saveData:
        blur = cv.Laplacian(img, cv.CV_64F).var()

        if count % moduleVal == 0 and blur > minBlur:
            nowTime = time.time()
            cv.imwrite(f"{img_path}{str(countFolder)}/{str(countSave)}_{str(int(blur))}_{str(nowTime)}.png", img)
            countSave += 1
            
        count += 1
    
    if showImage:
        cv.imshow("Image", img)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()