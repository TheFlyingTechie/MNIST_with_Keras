#Imports
from keras.datasets import mnist
from keras.layers import Dense
from keras.models import Sequential
from keras.utils import to_categorical

#Loading and formatting the data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

image_size = 28*28
num_classes = 10
x_train = x_train.reshape(x_train.shape[0], image_size)
x_test = x_test.reshape(x_test.shape[0], image_size)

y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

#Creating the model
model = Sequential()

#Adding Layers to the model
model.add(Dense(units=32, activation='sigmoid', input_shape=(image_size,)))
model.add(Dense(units=num_classes, activation='softmax'))
#Compiling and training the model
model.compile(optimizer='sgd', loss='categorical_crossentropy', metrics=['accuracy'])
history = model.fit(x_train, y_train, batch_size=128, epochs=5, verbose=False, validation_split=.1)
#Evaluating the model's accuracy with new data
loss, accuracy = model.evaluate(x_test, y_test, verbose=False)

print(f'Test loss: {loss:.3}')
print(f'Test accuracy: {accuracy:.3}')
