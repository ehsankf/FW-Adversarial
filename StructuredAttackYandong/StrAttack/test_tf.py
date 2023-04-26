import os

use_tfkeras = True

if use_tfkeras:
    from tensorflow.compat.v1.keras.applications.mobilenet import MobileNet
else:
    from keras.applications.mobilenet import MobileNet


def save_mobilenet_weights(alpha, filename):
    mobilenet = MobileNet(alpha=alpha, input_tensor=None, include_top=False, weights='imagenet', pooling=None)
    if use_tfkeras:
        mobilenet.save_weights(filepath=os.path.abspath(filename), overwrite=True, save_format='h5')
    else:
        mobilenet.save_weights(filepath=os.path.abspath(filename), overwrite=True)

def load_mobilenet_weights(alpha, filename):
    mobilenet = MobileNet(alpha=alpha, input_tensor=None, include_top=False, weights=None, pooling=None)
    mobilenet.load_weights(os.path.abspath(filename))


alpha = 0.75
filename = 'mobilenet_model.ckpt'

save_mobilenet_weights(alpha, filename)
load_mobilenet_weights(alpha, filename)
