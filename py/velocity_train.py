'''builds and trains the velocity model
'''

from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.utils import plot_model
from keras import optimizers
from keras import regularizers
from scipy import io
from training_plot import training_plot

# fpath = '../../MOT17/train/v/%s.mat'
fpath = '/home/star/Desktop/idea/train/v/%s.mat'

learning_rate, learning_rate_decay = 1e-4, 5e-7
data_loss = 'mean_squared_error'
data_dim, timesteps, lstm_units = 2, 6, 18


fit_epochs, fit_batch_size, fit_verbose = 200, 64, 2

x_train = io.loadmat(fpath % 'v_x_train')['v_x_train']
x_test = io.loadmat(fpath % 'v_x_test')['v_x_test']
y_train = io.loadmat(fpath % 'v_y_train')['v_y_train']
y_test = io.loadmat(fpath % 'v_y_test')['v_y_test']

print('Build velocity model...')
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

plot_model(model, to_file='v_model.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')

model.save('/home/star/Desktop/idea/v_model_LEN9.h5')
