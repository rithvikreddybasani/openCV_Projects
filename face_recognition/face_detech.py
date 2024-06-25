import cv2 as cv

img = cv.imread('my_image2.jpg')

# cv.imshow("Person",img)

gray = cv.cvtColor(img,cv.COLOR_BGR2GRAY)
gray = cv.resize(img, (500,500), interpolation=cv.INTER_CUBIC)
cv.imshow('Gray Person',gray)

harr_cascade = cv.CascadeClassifier('haar_face.xml')
faces_rect = harr_cascade.detectMultiScale(gray,scaleFactor=1.1,minNeighbors=6)

print(faces_rect[0])

# calculating number of faces found

print(len(faces_rect))

for (x,y,w,h) in faces_rect:
  cv.rectangle(img,(x,y),(x+w,y+h),(0,255,0),thickness=2)
  
cv.imshow('Detected Faces ',gray)

cv.waitKey(0)