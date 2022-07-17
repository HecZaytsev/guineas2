import matplotlib.pyplot as plt
import numpy as np
import os
import PIL
import tensorflow as tf
import os.path

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import pathlib

data_dir = pathlib.Path('C:\TA-ML\datasets\guineas2')

print(" ")
print("####----------Classificação de Porcos-----------####")
print(" ")
image_count = len(list(data_dir.glob('*/*.jpg')))

print('Porcos no dataset:')
print(image_count)

# Definições das dimensões das imagens
batch_size = 32
img_height = 180
img_width = 180

print(" ")
print("####----------Define Treino-----------####")
print(" ")
train_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.2,
  subset="training",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


print(" ")
print("####----------Define Validação-----------####")
print(" ")
val_ds = tf.keras.utils.image_dataset_from_directory(
  data_dir,
  validation_split=0.1,
  subset="validation",
  seed=123,
  image_size=(img_height, img_width),
  batch_size=batch_size)


print(" ")
print("####----------Classes existentes-----------####")
print(" ")
class_names = train_ds.class_names
print(class_names)



print(" ")
print("####----------Modelando Dados-----------####")
print(" ")
AUTOTUNE = tf.data.AUTOTUNE
train_ds = train_ds.cache().shuffle(1000).prefetch(buffer_size=AUTOTUNE)
val_ds = val_ds.cache().prefetch(buffer_size=AUTOTUNE)

normalization_layer = layers.Rescaling(1./255)

normalized_ds = train_ds.map(lambda x, y: (normalization_layer(x), y))
image_batch, labels_batch = next(iter(normalized_ds))
first_image = image_batch[0]
# Notice the pixel values are now in `[0,1]`.
print(np.min(first_image), np.max(first_image))


num_classes = len(class_names)

model = Sequential([
  layers.Rescaling(1./255, input_shape=(img_height, img_width, 3)),
  layers.Conv2D(16, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(32, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Conv2D(64, 3, padding='same', activation='relu'),
  layers.MaxPooling2D(),
  layers.Flatten(),
  layers.Dense(128, activation='relu'),
  layers.Dense(num_classes)
])


# Ajuste para melhorar Accuracy
# AUMENTO DE DADOS, gera novos dados parecidos
# Zooms aleatórios, flip na imagem
# Diminui overfitting

data_augmentation = keras.Sequential(
  [
    layers.RandomFlip("horizontal",
                      input_shape=(img_height,
                                  img_width,
                                  3)),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
  ]
)


# Dropout Regularization
# model = Sequential([
#    data_augmentation,
#    layers.Rescaling(1./255),
#    layers.Conv2D(16, 3, padding='same', activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(32, 3, padding='same', activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Conv2D(64, 3, padding='same', activation='relu'),
#    layers.MaxPooling2D(),
#    layers.Dropout(0.2),
#    layers.Flatten(),
#    layers.Dense(128, activation='relu'),
#    layers.Dense(num_classes)
#  ])


# Compila o modelo, adicionar tecnicas para aumentar accuracy acima
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])



print(" ")
print("####----------Treinar-----------####")
print(" ")
epochs=5
history = model.fit(
  train_ds,
  validation_data=val_ds,
  epochs=epochs
)



print(" ")
print("####----------Resultado | Utilizar IA -----------####")
print(" ")


dir_tst = "*"
while (dir_tst != 'Sair'):
    dir_tst = input('Digite o diretorio das imagens: ')

    mylist = os.listdir(dir_tst)

    # Reseta nome dos arquivos
    reset = 1
    for x in mylist:
      os.rename(dir_tst+"\\"+x, dir_tst+"\\"+str(reset)+".jpg")
      reset = reset +1

    # atualiza lista com novos nomes
    mylist = os.listdir(dir_tst)
    count = 1
    for x in mylist:
        img = tf.keras.utils.load_img(
        dir_tst+"\\"+x, target_size=(img_height, img_width)
        )
        img_array = tf.keras.utils.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0) # Create a batch

        predictions = model.predict(img_array)
        score = tf.nn.softmax(predictions[0])

        os.rename(dir_tst+"\\"+x      , dir_tst+"\\"+str(count)+ " - "+"{}".format(class_names[np.argmax(score)])+" - {:.2f}".format(100 * np.max(score))+".jpg")

        count = count + 1