from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D # convolutional filter, mengambil dari 4 pixel 
from keras.layers import Activation, Dropout, Flatten, Dense #activation = seperti fuction get, dense itu untuk menentukan berapa kelas misal 0-9 berarti desne 10
from keras import backend as K


# dimensions of our images.
img_width, img_height = 150, 150

train_data_dir = 'data/train'# direction of the folder
validation_data_dir = 'data/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 60
batch_size = 25#the bigger the batch_size the longger it takes to train the model

if K.image_data_format() == 'channels_first': #depth, width, height
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential() #create the model
model.add(Conv2D(32, (3, 3), input_shape=input_shape)) #create 1 layer that contains 32 node, (3,3) convolutional filter size
model.add(Activation('relu')) #relu: if the weight exceed the treshold it will be changed to linear
model.add(MaxPooling2D(pool_size=(2, 2)))# maxPooling: make the image smaller after being editen by Conv2D

model.add(Conv2D(32, (3, 3))) #no need to input shape because the input shape from the second layer is the output from first layer
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))#per 2x2 take the max value of it

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(36))
model.add(Activation('sigmoid')) #layer output must be sigmoid, clasification model uses sigmoid activation

#untuk compile modelnya
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',#for sampling
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,#bengkok
    zoom_range=0.2,
    )


# validation / test------------------------------------------
# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')#if more than 2 optput result we msut use categorical if just 2 output use binary

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

#start the training
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save('model.h5')
#model.save_weights('first_try.h5')

