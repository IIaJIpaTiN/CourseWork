import cv2
import face_recognition
import os
import numpy as np
from PIL import Image
from my__model import create_model


def read_image(filename):
    img = Image.open(filename).convert('L')
    return np.array(img)

model = create_model((100, 100, 1))
model.load_weights('my_model_weights_4.h5')

personalCnt = 26
dim = 100
personalBase = np.zeros([personalCnt, 3, dim, dim, 1])
for i in range(personalCnt):
    for j in range(3):
        ind = np.random.randint(20)
        img_load = read_image('set/' + str(i + 1) + '/' + str(ind + 1) + '.jpg')
        personalBase[i, j, :, :, 0] = img_load
personalBase /= 255


print('Введите путь к видеофайлу')
video_path = input()
while(True) :
    if os.path.isfile(video_path):
        name = os.path.splitext(video_path)
        if len(name) > 0 and name[len(name) - 1] == '.mp4':
            break
    print('Некоректный путь, попробуйте ещё раз')
    video_path = input()

video_capture = cv2.VideoCapture(video_path)

face_locations = []
stat = [0 for i in range(personalCnt)]

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

    for top, right, bottom, left in face_locations:
        top *= 4; right *= 4; bottom *= 4; left *= 4

        # Приведение найденного лица к нужному размеру и формату данных
        curImg = Image.fromarray(frame[top:bottom, left:right])
        curImg = curImg.convert('L')
        percent = (100 / float(curImg.size[0]))
        hsize = int((float(curImg.size[1]) * float(percent)))
        curImg = curImg.resize((100, hsize), Image.ANTIALIAS)
        background = Image.new('L', (100, 100))
        background.paste(curImg)

        # Создание массива для сравнения найденного лица
        curBase = np.zeros([personalCnt, 1, dim, dim, 1])
        for i in range(personalCnt):
            curBase[i, 0, :, :, 0] = np.array(background)
        curBase /= 255

        # Классификация полученного лица нейросетью
        pred = np.zeros([personalCnt, 20])
        for i in range(3):
            p = model.predict([personalBase[:, i], curBase[:, 0]])
            pred[:, i] = p.reshape(26)

        # Сохранение 3х наиболее вероятных вариантов в массив со статистикой
        res = np.mean(pred, axis=1)
        personId = np.argmin(res)
        stat[personId] += 0.5
        res[personId] = 100
        personId = np.argmin(res)
        stat[personId] += 0.3
        res[personId] = 100
        personId = np.argmin(res)
        stat[personId] += 0.2

        # Отрисовка рамки вокруг распознанного лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

    # Отрисовка обработанного изображения
    cv2.imshow('Video', cv2.resize(frame, (0, 0), fx=0.5, fy=0.5))
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cv2.destroyAllWindows()
print("Наиболее вероятный Id: " + str(np.argmax(stat) + 1))