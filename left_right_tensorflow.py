import os
import zipfile
base_dir = '/Users/michael/Desktop/chambit/tensorflow/cb/'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')

# 훈련에 사용되는 왼/오 이미지 경로
train_left_dir = os.path.join(train_dir, 'c')
train_right_dir = os.path.join(train_dir, 'b')
print(train_left_dir)
print(train_right_dir)

# 테스트에 사용되는 왼/오 이미지 경로
validation_left_dir = os.path.join(validation_dir, 'c')
validation_right_dir = os.path.join(validation_dir, 'b')
print(validation_left_dir)
print(validation_right_dir)

print('Total training left images :', len(os.listdir(train_left_dir)))
print('Total training right images :', len(os.listdir(train_right_dir)))

print('Total validation left images :', len(os.listdir(validation_left_dir)))
print('Total validation right images :', len(os.listdir(validation_right_dir)))




#텐서플로우 모델 구축
import tensorflow as tf


model = tf.keras.models.Sequential([
  tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(300, 300, 3)),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
  tf.keras.layers.MaxPooling2D(2,2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(1, activation='sigmoid')
])

model.summary()

#모델 컴파일 
from tensorflow.keras.optimizers import RMSprop

model.compile(optimizer=RMSprop(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics = ['accuracy'])


#이미지 전처리 
from tensorflow.keras.preprocessing.image import ImageDataGenerator


train_datagen = ImageDataGenerator( rescale = 1.0/255. )
test_datagen  = ImageDataGenerator( rescale = 1.0/255. )

train_generator = train_datagen.flow_from_directory(train_dir,
                                                  batch_size=20,
                                                  class_mode='binary',
                                                  target_size=(300, 300))
validation_generator =  test_datagen.flow_from_directory(validation_dir,
                                                       batch_size=20,
                                                       class_mode  = 'binary',
                                                       target_size = (300, 300))

#모델 훈련
history = model.fit(train_generator,
                    validation_data=validation_generator,
                    steps_per_epoch=10,#10
                    epochs=15,#40
                    validation_steps=20,#10
                    verbose=1) #verbose -> 모델 훈련 과정 인터페이스 


import matplotlib.pyplot as plt
#accuracy, loss 그래프 출력. 
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(len(acc))

plt.plot(epochs, acc, 'bo', label='Training accuracy')
plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs, loss, 'go', label='Training Loss')
plt.plot(epochs, val_loss, 'g', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()
#plt 그래프 출력
model.save("c_b_model.h5")

#지정경로 이미지 호출하여 테스트 







