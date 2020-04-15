import os
import cv2
import dlib
import numpy as np
from sklearn import preprocessing
import progressbar as bar
from sklearn.neighbors import KNeighborsClassifier



detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_recognition_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

face_list = []

#find everything in the data folder
for directory in os.listdir('data'):
    #find directories
    if os.path.isdir("data/"+directory):
        #find the file belongs to which folder 
        for file in os.listdir("data/"+directory):
            #find whether this file is a picture
            if file[-4:] in [".jpg",".png",".jpeg"]:
                face = {}
                face['person'] = directory
                face['path'] = "data/"+directory + "/" +file
                face_list.append(face)

for face in face_list:
    #print(face['path'])
    img = cv2.imread(face['path'])
    face_rect = detector(img, 1)[0]
    shape68 = predictor(img, face_rect)
    face_descriptor = face_recognition_model.compute_face_descriptor(img,shape68)
    face['descriptor'] =np.array(face_descriptor)

le= preprocessing.LabelEncoder()

le.fit([face['person'] for face in face_list])

X = [face['descriptor'] for face in face_list]
Y = le.transform([face['person'] for face in face_list])
#print(X)
#print(Y)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit([face['descriptor'] for face in face_list], le.transform([face['person'] for face in face_list]))

img = cv2.imread('data/test1.jpg')
# WARNING: ONLY THE FIRST DETECTED FACE IN THE IMAGE IS USED
face_rect = detector(img, 1)[0]
shape68 = predictor(img, face_rect)
face_descriptor = np.array(face_recognition_model.compute_face_descriptor(img, shape68))

predict_person = np.squeeze(le.inverse_transform(knn.predict([face_descriptor])))
person_classes = le.classes_
predict_proba = knn.predict_proba([face_descriptor])

print('\nPredict Person: {0}'.format(predict_person))
print('Probability:')
for i in range(len(person_classes)):
	print('{0}: {1}'.format(person_classes[i], np.squeeze(predict_proba)[i]))

cv2.rectangle(img, (face_rect.left(), face_rect.top()), (face_rect.right(), face_rect.bottom()), ( 0, 255, 0), 1, cv2.LINE_AA)
cv2.putText(img, str(predict_person), (face_rect.left(), face_rect.top() - 3), cv2.FONT_HERSHEY_DUPLEX, 1.1e-3 * img.shape[0], ( 0, 255, 0), 1, cv2.LINE_AA)
cv2.imshow('Facial Recognition', img)
cv2.waitKey()