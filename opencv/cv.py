import cv2

face_cascade=cv2.CascadeClassifier('haarcascade_frontface_default.xml')
eye_cascade=cv2.CascadeClassifier('harcascade_eye.xml')

def detect(gray_img,color_img):
    faces=face_cascade.detectMultiScale(gray_img,1.3,5)

    for (x,y,w,h) in faces:
        cv2.rectangle(color_img, (x,y), (x+h,y+w), (0,0,255), 1)
        eyeimg_gray=gray_img[x:x+w, y:y+h]
        eyeimg_color=color_img[x:x+w, y:y+h]

        eyes=eye_cascade.detectMultiScale(eyeimg_gray,1.1,3)

        for (ex,ey,ew,eh) in eyes:
            cv2.rectangle(eyeimg_color,  (ex,ey), (ex+ew,ey+eh), (0,255,0), 1)

    return color_img

video_capture =cv2.VideoCapture(0)

while True:
    _,img=video_capture.read()
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    canvas=detect(gray,img)
    cv2.imshow("Video",canvas)

    if(cv2.waitKey(1) & 0xFF == ord('q')):
        break
video_capture.release()
cv2.destroyAllWindows()