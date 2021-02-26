import cv2 #pip install opencv_python

#set link frontal face detector xml file from python folder
face_directory = cv2.CascadeClassifier('C:/Python/Python36/Lib/site-packages/cv2/data/haarcascade_frontalface_default.xml')

def face_extractor(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) #Image convert color to gray
    faces = face_directory.detectMultiScale(gray, 1.3,5)

    if faces is():
        return None


    for(x,y,w,h) in faces:
        cropped_face = img[y: y+h, x:x+w]

    return cropped_face


cap = cv2.VideoCapture(0)
count = 0


while True:
    ret, frame = cap.read()
    if face_extractor(frame) is not None:
        count+=1
        face = cv2.resize(face_extractor(frame),(200,200)) #Resize image 200x200
        face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY) #image convert color to gray


        file_name_path = 'C:/Users/Nasir/PycharmProjects/faceRecognitaion/faces/user'+str(count)+'.jpg' #Save image in this directory with jpg format
        cv2.imwrite(file_name_path,face)

        cv2.putText(face,str(count),(50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,255,0),2)
        cv2.imshow('Face Cropper', face)

    else:
        print("Face Not Found") #if face not found show this message
        pass

    if cv2.waitKey(1)==13 or count==10: #capture image 10 0r press any key to stop.
        break

cap.release() # Release camera
cv2.destroyAllWindows() #Destroy All Window After complete counts
cv2.waitKey()
print('Collecting All Samples Completed!!!') #print this message t infome user



