from keras import backend as K
from keras.layers import Activation
from keras.layers import Input, Lambda, Dense, Dropout, Convolution2D, MaxPooling2D, Flatten
from keras.models import Sequential, Model
from keras.optimizers import RMSprop


def build_base_network(input_shp):
    seq = Sequential()

    nb_filter = [64, 128]
    kernel_size = 3

    # convolutional layer 1
    seq.add(Convolution2D(nb_filter[0], kernel_size, kernel_size, input_shape=input_shp,
                          padding='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))

    # convolutional layer 2
    seq.add(Convolution2D(nb_filter[1], kernel_size, kernel_size,
                          padding='valid'))
    seq.add(Activation('relu'))
    seq.add(MaxPooling2D(pool_size=(2, 2)))
    seq.add(Dropout(.25))

    # flatten
    seq.add(Flatten())
    seq.add(Dense(128, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(50, activation='relu'))
    return seq


def euclidean_distance(vects):
    x, y = vects
    return K.sqrt(K.maximum(K.sum(K.square(x - y), axis=1, keepdims=True), K.epsilon()))

def eucl_dist_output_shape(shapes):
    shape1, shape2 = shapes
    return (shape1[0], 1)

def contrastive_loss(y_true, y_pred):
    margin = 1
    return K.mean(y_true * K.square(y_pred) + (1 - y_true) * K.square(K.maximum(margin - y_pred, 0)))


def create_model(input_dim):
    img_a = Input(shape=input_dim)
    img_b = Input(shape=input_dim)

    base_network = build_base_network(input_dim)
    feat_vecs_a = base_network(img_a)
    feat_vecs_b = base_network(img_b)

    distance = Lambda(euclidean_distance, output_shape=eucl_dist_output_shape)([feat_vecs_a, feat_vecs_b])

    rms = RMSprop()
    model = Model([img_a, img_b], distance)
    model.compile(loss=contrastive_loss, optimizer=rms)

    return model