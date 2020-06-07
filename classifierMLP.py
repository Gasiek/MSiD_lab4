from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model
import mnist_reader

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

ModelCheck = ModelCheckpoint(filepath='model2.h5',
                             monitor='val_loss',
                             save_best_only=True)

EarlyStop = EarlyStopping(monitor='val_loss',
                          patience=5,
                          verbose=1)

X_train, y_train = mnist_reader.load_mnist('fashion', kind='train')
X_val, y_val = mnist_reader.load_mnist('fashion', kind='t10k')

X_train = X_train.astype('float32') / 255.0
X_val = X_val.astype('float32') / 255.0

y_train = to_categorical(y_train, len(class_names))
y_val = to_categorical(y_val, len(class_names))

model = Sequential()
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.3))
model.add(Dense(128, activation='relu',
                kernel_regularizer=regularizers.l2(0.01)))
model.add(Dropout(0.3))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])


def fit_the_model():
    model.fit(X_train,
              y_train,
              epochs=50,
              verbose=1,
              batch_size=1024,
              validation_split=0.2,
              callbacks=[EarlyStop, ModelCheck]
              )


# model = load_model('model.h5')
fit_the_model()
model.evaluate(X_val, y_val, batch_size=1024)
