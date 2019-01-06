from keras.models import Sequential
from keras.layers import Dense
from keras import optimizers
from keras.utils import plot_model
from training_plot import training_plot
import numpy as np

fpath = '/home/star/Desktop/idea/train/'

learning_rate, learning_rate_decay = 1e-4, 5e-6
fit_epochs, fit_batch_size, fit_verbose = 3, 64, 2

ds_x = np.load('%sds_x.npy' % fpath)
ds_y = np.load('%sds_y.npy' % fpath)
split_factor, length = 0.7, ds_x.shape[0]
split = int(split_factor * length)
ds_x_train, ds_x_test = ds_x[:split], ds_x[split:]
ds_y_train, ds_y_test = ds_y[:split], ds_y[split:]

model = Sequential()
model.add(Dense(28, input_shape=(1,), activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, decay=learning_rate_decay)
model.compile(optimizer=my_adam, loss='binary_crossentropy',
              metrics=['accuracy'])
history = model.fit(ds_x_train, ds_y_train, epochs=fit_epochs,
                    batch_size=fit_batch_size, verbose=fit_verbose,
                    validation_data=(ds_x_test, ds_y_test))

training_plot(history)

plot_model(model, to_file='ds_model.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')

model.save('/home/star/Desktop/idea/ds_model.h5')
