from keras.models import load_model
from keras.utils import plot_model

'''
# v_model.h5
model = load_model('v_model.h5')
plot_model(model, to_file='v_model.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')
'''

'''
model = load_model('siamese.h5')
plot_model(model, to_file='siamese.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')
'''
'''
model = load_model('p_model.h5')
plot_model(model, to_file='p_model.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')
'''
'''
model = load_model('bottom_vgg.h5')
plot_model(model, to_file='bottom_vgg.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')
'''
model = load_model('ds_model.h5')
plot_model(model, to_file='ds_model.png', show_shapes=True,
           show_layer_names=True, rankdir='LR')