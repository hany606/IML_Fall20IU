from keras.callbacks import EarlyStopping, ReduceLROnPlateau, LearningRateScheduler
from keras.datasets import cifar10
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout, BatchNormalization
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adam, Nadam, RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19


(x_train, y_train), (x_test, y_test) = cifar10.load_data()


def make_model():
    # m = Sequential()

    # m.add(Conv2D(filters=96, kernel_size=5, 
    #              input_shape=(32,32,3), activation='relu'))
    # m.add(MaxPool2D(pool_size=2, strides=1))
    # m.add(BatchNormalization())

    # m.add(Conv2D(filters=128, kernel_size=3, 
    #              activation='relu'))
    # m.add(MaxPool2D(pool_size=2))
    # m.add(BatchNormalization())

    # m.add(Flatten())
    
    # m.add(Dense(512, activation="relu"))
    # m.add(BatchNormalization())
    # m.add(Dropout(0.2))
    
    # m.add(Dense(512, activation="relu"))
    # m.add(BatchNormalization())
    # m.add(Dropout(0.2))

    # m.add(Dense(10, activation="softmax"))

    # Transfer learning
    vgg19 = VGG19(include_top=False, input_shape=(32, 32, 3), weights='imagenet') # here return the NN without the top layers (high-level layers(the dense), only sharing the conv lauers)
    # turn off the trainable flags to the conv layers in order to freeze it from the traning and only trainable layers are the fully connected layers (that we are adding them by ourselves)
    for layer in vgg19.layers:
        layer.trainable = False

    x = Flatten()(vgg19.output)
    
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    x = Dense(512, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)

    x = Dense(10, activation="softmax")(x)

    m = Model(input=vgg19.input, output=x)

    return m

def schedule(epoch_idx, lr):
    if(epoch_idx == 10):
        return lr * 0.3
    
    elif(epoch_idx == 20):
        return lr * 0.5

    return lr


model = make_model()
model.compile(optimizer=Nadam(1e-4, clipnorm=1), loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.save("first_model.h5")


early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)

lr_schedule = ReduceLROnPlateau(monitor="val_accuracy", factor=0.3, patience=6)
# lr_schedule = LearningRateScheduler(schedule)

generator = ImageDataGenerator(rotation_range=20,
                               width_shift_range=0.1,
                               height_shift_range=0.1,
                               zoom_range=0.15,
                               horizontal_flip=True,
                               validation_split=0.15,)

# generator.fit(x_train) # form zero_weight=True, featurewise_std_normalization=True

# model.fit(x=x_train, y=y_train,
#           validation_split=0.15,
#           epochs=30, verbose=2,
#           batch_size=128,
#           callbacks=[early_stopping, lr_schedule])

model.fit(generator.flow(x_train, y_train, batch_size=128),
          validation_data=generator.flow(x_train, y_train, batch_size=128, subset='validation'),
          steps_per_epoch=len(x_train)/128,
          epochs=30, verbose=2,
          callbacks=[early_stopping, lr_schedule])



# print(model.evaluate(x_test, y_test))