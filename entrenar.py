import sys
import os
#from PIL import Image
from tensorflow.python.keras.preprocessing.image import ImageDataGenerator
from tensorflow.python.keras import optimizers      #Optimizador para entrenar el algoritmo
from tensorflow.python.keras.models import Sequential   #Ayuda a hacer redes secuenciales
from tensorflow.python.keras.layers import Dropout, Flatten, Dense, Activation
from tensorflow.python.keras.layers import Convolution2D, MaxPooling2D #Capas para hacer convoluciones
from tensorflow.python.keras import backend as K    #Para matar background

K.clear_session()

data_entrenamiento = 'data/entrenamiento'
data_validation = 'data/validacion'

#Parametros

epocas = 20                 #Numero de veces en el que itera en el set de datos
altura, longitud = 100, 100 #Tamaño de imagenes
batch_size = 32
pasos = 1000            #Numero de veces en que se procesa la informacion en cada epoca
pasos_validacion = 200
filtrosConv1 = 32
filtrosConv2 = 64
tamaño_filtro1 = (3, 3)
tamaño_filtro2 = (2, 2)
tamaño_pool = (2, 2)
clases = 3
lr = 0.0005    #Es que tan seguido va a cambiar

#Pre_procesamiento_de_imagenes

entrenamiento_datagen = ImageDataGenerator(
    rescale= 1./255,  #Poner datos de 0 a 1
    shear_range = 0.3,      #Inclina las imagenes
    zoom_range = 0.3,
    horizontal_flip = True
)

#En la validacion las imagenes deben estar igual
validation_datagen = ImageDataGenerator(
    rescale = 1./255
)

imagen_entrenamiento = entrenamiento_datagen.flow_from_directory(
    data_entrenamiento,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode ='categorical'
)

imagen_validacion = validation_datagen.flow_from_directory(
    data_validation,
    target_size = (altura, longitud),
    batch_size = batch_size,
    class_mode = 'categorical'
 )

#Crear una red CNN

cnn = Sequential()

#Primera capa
cnn.add(Convolution2D(filtrosConv1, tamaño_filtro1, padding = 'same', input_shape=(altura, longitud, 3), activation='relu'))

cnn.add(MaxPooling2D(pool_size=tamaño_pool))

#Segunda capa
cnn.add(Convolution2D(filtrosConv2, tamaño_filtro2, padding='same', activation='relu'))
cnn.add(MaxPooling2D(pool_size=tamaño_pool))

cnn.add(Flatten())   #Aplanamiento de informacion
cnn.add(Dense(256, activation = 'relu'))
cnn.add(Dropout(0.5))
#Capa final
cnn.add(Dense(clases, activation= 'softmax'))

cnn.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr = lr), metrics=['accuracy'])

cnn.fit(imagen_entrenamiento, steps_per_epoch=pasos, epochs=epocas, validation_data=imagen_validacion, validation_steps=pasos_validacion)

dir = 'modelo/'

if not os.path.exists(dir):
    os.mkdir(dir)

cnn.save('modelo/modelo.h5')
cnn.save_weights('modelo/pesos.h5')








