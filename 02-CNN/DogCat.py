from keras.layers import Dense
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten

classifier = Sequential()

# Adding first convolutional layer
classifier.add(Convolution2D(32, 3,3, 
                             activation='relu',
                             input_shape = (64,64,3)))

# MaxPooling for first conv layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Adding second convolutional layer
classifier.add(Convolution2D(32,3,3,activation='relu'))

# Maxpooling for second conv layer
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Flatten your Layers
classifier.add(Flatten())

classifier.add(Dense(output_dim=128, activation='relu'))
classifier.add(Dense(output_dim=1, activation='sigmoid'))

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                   metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train = ImageDataGenerator(rescale = 1/255,
                           shear_range = 0.2,
                           zoom_range = 0.2,
                           horizontal_flip=True
                           )

test = ImageDataGenerator(rescale=1/255)

training_set = train.flow_from_directory('dataset/training_set',
                                                 target_size = (64, 64),
                                                 batch_size = 32,
                                                 class_mode = 'binary')

test_set = test.flow_from_directory('dataset/test_set',
                                            target_size = (64, 64),
                                            batch_size = 32,
                                            class_mode = 'binary')

classifier.fit_generator(training_set,
                         samples_per_epoch = 8000,
                         nb_epoch = 5,
                         validation_data = test_set,
                         nb_val_samples = 2000,
                         steps_per_epoch = 20)











