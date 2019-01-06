'''builds and trains the position model
'''

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import plot_model
from keras import optimizers
from keras import regularizers
from scipy import io
from numpy import reshape
from training_plot import training_plot

fpath = '../../MOT17/train/p/%s.mat'
learning_rate, learning_rate_decay = 1e-3, 5e-6
data_loss = 'mean_squared_error'
data_dim, timesteps, lstm_units = 1, 6, 12
fit_epochs, fit_batch_size, fit_verbose = 200, 64, 2

x_train = io.loadmat(fpath % 'p_x_train')['p_x_train']
x_train = reshape(x_train, (len(x_train), timesteps, data_dim))
x_test = io.loadmat(fpath % 'p_x_test')['p_x_test']
x_test = reshape(x_test, (len(x_test), timesteps, data_dim))
y_train = io.loadmat(fpath % 'p_y_train')['p_y_train']
y_test = io.loadmat(fpath % 'p_y_test')['p_y_test']

print('Build pos model...')
model = Sequential()
model.add(LSTM(lstm_units, input_shape=(timesteps, data_dim),
               kernel_regularizer=regularizers.l2(1e-3),
               recurrent_regularizer=regularizers.l2(1e-3), unroll=True))
model.add(Dense(data_dim))

my_adam = optimizers.Adam(lr=learning_rate, beta_1=0.9, beta_2=0.999,
                          epsilon=1e-08, decay=learning_rate_decay)
model.compile(optimizer=my_adam, loss=data_loss, metrics=['accuracy'])
history = model.fit(x_train, y_train, epochs=fit_epochs,
                    batch_size=fit_batch_size, verbose=fit_verbose,
                    validation_data=(x_test, y_test))

training_plot(history)

plot_model(model, to_file='p_model.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')

model.save('p_model.h5')
