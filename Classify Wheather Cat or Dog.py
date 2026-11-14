import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from tensorflow.keras.constraints import max_norm
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import SGD

def load_images_from_folder(folder, label):
    images, labels = [], []
    if os.path.exists(folder):
        for filename in os.listdir(folder):
            try:
                img_path = os.path.join(folder, filename)
                img = Image.open(img_path).convert('RGB')
                img = img.resize((128, 128))
                images.append(np.array(img))
                labels.append(label)
            except Exception:
                pass
    return images, labels

class_names = ['Cats', 'Dogs']
cat_images, cat_labels = load_images_from_folder('Cats', 0)
dog_images, dog_labels = load_images_from_folder('Dogs', 1)

x_data = np.array(cat_images + dog_images)
y_data = np.array(cat_labels + dog_labels).reshape(-1, 1)

shuffle_order = np.random.permutation(len(x_data))
x_data, y_data = x_data[shuffle_order], y_data[shuffle_order]

split_index = int(len(x_data) * 0.8)
x_train, x_test = x_data[:split_index], x_data[split_index:]
y_train, y_test = y_data[:split_index], y_data[split_index:]

print("Class names:", class_names)

for i in range(min(9, len(x_train))):
    plt.subplot(330 + 1 + i)
    plt.imshow(x_train[i])
plt.show()

num_classes = len(class_names)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

x_train = x_train.astype('float32') / 255.0
x_test = x_test.astype('float32') / 255.0

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

model = Sequential()
model.add(Input(shape=(128, 128, 3)))
model.add(Conv2D(16, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(Dropout(0.2))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dropout(0.2))
model.add(Dense(128, activation='relu', kernel_constraint=max_norm(3)))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation='softmax'))

opt = SGD(learning_rate=0.01, momentum=0.9, nesterov=False)
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
model.summary()

epochs = 20
batch_size = 64

history = model.fit(
    x_train, 
    y_train, 
    validation_data=(x_test, y_test), 
    epochs=epochs, 
    batch_size=batch_size, 
    verbose=1
)

scores = model.evaluate(x_test, y_test, verbose=0)
print(f"Test Accuracy: {scores[1]*100:.2f}%")