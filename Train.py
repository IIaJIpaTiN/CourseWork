import numpy as np
from PIL import Image
from my__model import create_model

from sklearn.model_selection import train_test_split

def read_image(filename):
    img = Image.open(filename).convert('L')
    return np.array(img)

total_sample_size = 10000

def get_data(total_sample_size):
    # read the image
    image = read_image('data/set/' + str(1) + '/' + str(1) + '.jpg')

    # get the new size
    dim1 = image.shape[0]
    dim2 = image.shape[1]

    count = 0

    # инициализация numpy массива с размерами [total_sample, no_of_pairs, dim1, dim2]
    # dim1, dim2 - размеры исходного изображения
    x_geuine_pair = np.zeros([total_sample_size, 2, dim1, dim2, 1])  # 2 для пар изображений
    y_genuine = np.zeros([total_sample_size, 1])

    for i in range(26):
        for j in range(int(total_sample_size / 26)):
            ind1 = 0
            ind2 = 0

            while ind1 == ind2:
                ind1 = np.random.randint(20)
                ind2 = np.random.randint(20)

            # считать два изображения как numpy массивы
            img1 = read_image('data/set/' + str(i + 1) + '/' + str(ind1 + 1) + '.jpg')
            img2 = read_image('data/set/' + str(i + 1) + '/' + str(ind2 + 1) + '.jpg')

            # сохранить изображения в инициализированном numpy массиве
            x_geuine_pair[count, 0, :, :, 0] = img1
            x_geuine_pair[count, 1, :, :, 0] = img2

            # для изображений из одной директории значение равно 1. (подлинная пара)
            y_genuine[count] = 1
            count += 1

    count = 0
    x_imposite_pair = np.zeros([total_sample_size, 2, dim1, dim2, 1])
    y_imposite = np.zeros([total_sample_size, 1])

    for i in range(int(total_sample_size / 20)):
        for j in range(20):

            # считавание изображений из разных директорий (ложные пары)
            while True:
                ind1 = np.random.randint(26)
                ind2 = np.random.randint(26)
                if ind1 != ind2:
                    break

            img1 = read_image('data/set/' + str(ind1 + 1) + '/' + str(j + 1) + '.jpg')
            img2 = read_image('data/set/' + str(ind2 + 1) + '/' + str(j + 1) + '.jpg')

            x_imposite_pair[count, 0, :, :, 0] = img1
            x_imposite_pair[count, 1, :, :, 0] = img2

            # для изображений из разных директорий значение равно 0. (ложные пары)
            y_imposite[count] = 0
            count += 1

    # объединение подлиных и ложных пар для получения общего массива данных
    X = np.concatenate([x_geuine_pair, x_imposite_pair], axis=0) / 255
    Y = np.concatenate([y_genuine, y_imposite], axis=0)

    return X, Y

print("Create Data")
X, Y = get_data(total_sample_size)
x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=.25)

print("Create Model")
model = create_model(x_train.shape[2:])

epo = 26
img_1 = x_train[:, 0]
img_2 = x_train[:, 1]

print("Train:")
model.fit([img_1, img_2], y_train, validation_split=.25, batch_size=128, verbose=2, epochs=epo)

pred = model.predict([x_test[:, 0], x_test[:, 1]])

def compute_accuracy(predictions, labels):
    return np.mean((predictions.ravel() < 0.5) == (labels.ravel() == 1.0))

print("Accuracy: " + str(compute_accuracy(pred, y_test)))

model.save_weights('my_model_weights_4.h5')
print("Saved model weights to disk")