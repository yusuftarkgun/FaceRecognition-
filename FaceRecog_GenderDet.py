import cv2
import face_recognition
import time


person1_image = face_recognition.load_image_file("images/yusuf.jpg")
person1_face_encoding = face_recognition.face_encodings(person1_image)[0]

person2_image = face_recognition.load_image_file("images/zeynep.jpg")
person2_face_encoding = face_recognition.face_encodings(person2_image)[0]

person3_image = face_recognition.load_image_file("images/rahma.jpg")
person3_face_encoding = face_recognition.face_encodings(person3_image)[0]

person4_image = face_recognition.load_image_file("images/talha.jpg")
person4_face_encoding = face_recognition.face_encodings(person4_image)[0]



known_face_encodings = [person1_face_encoding, person2_face_encoding, person3_face_encoding, person4_face_encoding]
known_face_names = ["Yusuf", "Zeynep", "Rahma", "Talha"]



prototxt_path = "/Users/yusuftarikgun/anaconda3/lib/python3.11/site-packages/cvlib/pre-trained/gender_deploy.prototxt"
caffemodel_path = "/Users/yusuftarikgun/anaconda3/lib/python3.11/site-packages/cvlib/pre-trained/gender_net.caffemodel"
gender_model = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)



net = cv2.dnn.readNetFromCaffe(prototxt_path, caffemodel_path)


cap = cv2.VideoCapture(0)

width = 640
height = 480

cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
start_time = time.time()

while True:

    ret, frame = cap.read()

    face_locations = face_recognition.face_locations(frame)
    face_encodings = face_recognition.face_encodings(frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):

        face = frame[top:bottom, left:right]
        blob = cv2.dnn.blobFromImage(face, 0.5, (227, 227), (78.4263377603, 87.7689143744, 114.895847746), swapRB=False)
        gender_model.setInput(blob)

        gender_preds = gender_model.forward()
        gender = "Erkek" if gender_preds[0][0] > gender_preds[0][1] else "Kadin"

        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        name = "Bilinmiyor"

        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

        cv2.putText(frame, f'Isim: {name}', (left, top-30), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(frame, f'Cinsiyet: {gender}', (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    cv2.imshow('Yuz Tanima ve Cinsiyet Belirleme', frame)
    cv2.imshow("Cinsiyet ve yuz tanima", frame)

    fps = 1.0 / (time.time() - start_time)
    print(f"FPS: {fps:.2f}")
    # 'q' tuşuna basarak çıkış yap
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

    start_time = time.time()
cap.release()
cv2.destroyAllWindows()