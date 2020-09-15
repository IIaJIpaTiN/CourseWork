import cv2
import face_recognition
import os

directory = 'data/video/Reformed/old_men_'
dirCnt = 3

for i in range(dirCnt):
    curDir = directory + str(i + 1)
    files = os.listdir(curDir)
    videos = filter(lambda x: x.endswith('.mp4'), files)
    count = 0
    for v in videos:
        video_capture = cv2.VideoCapture(curDir + '/' + v)
        print("Start:" + curDir + '/' + v)

        # Initialize variables
        face_locations = []

        while True:
            # Получение одного кадра видео
            exist, frame = video_capture.read()
            if not exist:
                break
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)

            # Конвертация изображения из BGR формата (используемого OpenCV) в RGB формат (для использования в библиотеке face_recognition)
            rgb_frame = small_frame[:, :, ::-1]

            # Нахождение всех лиц в текущим кадре видео
            face_locations = face_recognition.face_locations(rgb_frame)

            # Сохранение результата
            for top, right, bottom, left in face_locations:
                top *= 4; right *= 4; bottom *= 4; left *= 4
                count += 1
                cv2.imwrite(curDir + "/frame%d.jpg" % count, frame[top:bottom, left:right])

        print("End: frame count = " + str(count))
    print("\n")
