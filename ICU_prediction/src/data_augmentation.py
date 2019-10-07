from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import numpy as np
import keras
from keras import backend
import tensorflow as tf
from tensorflow.python.platform import flags

from cleverhans.utils_mnist import data_mnist
from cleverhans.utils_tf import model_train, model_eval
from cleverhans.attacks import FastGradientMethod
from cleverhans.attacks import BasicIterativeMethod
from cleverhans.utils import AccuracyReport
from cleverhans.utils_keras import cnn_model
from cleverhans.utils_keras import KerasModelWrapper

FLAGS = flags.FLAGS

config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.1
sess = tf.Session(config=config)
keras.backend.set_session(sess)

def naive_mlp_model(input_shape, units = 256): 
    from keras.models import Model, Sequential
    from keras.layers import Input, Dense, Conv1D
    from keras.layers import Lambda, Dropout, Concatenate, Flatten, Activation
    from keras.layers import LSTM, GRU, SimpleRNN
    # 2 layers mlp
    model = Sequential()
    model.add(Dense(units, W_regularizer = 'l2', activation = "relu", input_shape=(input_shape, )))
    model.add(Dropout(0.5))
    model.add(Dense(units, W_regularizer = 'l2', activation = "relu"))
    model.add(Dropout(0.5))
    model.add(Dense(2, W_regularizer = 'l2', activation = "softmax"))
    return model

def rolling(x, y):
    icu_num = int(np.sum(y))
    print('icu_num =', icu_num, np.mean(y))
    icu = x[:icu_num]
    print('original size =', icu.shape[0])
    for i in range(6):
        icu = np.concatenate([icu, icu], axis = 0) # 32
    icu = icu[:icu_num*50]
    x = np.concatenate([icu, x[icu_num:]])
    y = np.ones(x.shape[0])
    y[:icu.shape[0]] *= 0
    print('rolling size =', icu.shape[0], x.shape[0])
    return x, y
    #exit()

def icu(nb_epochs=10, batch_size=1024, epsilon = 0.3,
        train_dir="/tmp", filename="icu_0106_test.ckpt", load_model=False, testing=False):
   
    ##################### initialization #####################
    keras.layers.core.K.set_learning_phase(0)
    # Object used to keep track of (and return) key accuracies
    report = AccuracyReport()
    # Set TF random seed to improve reproducibility
    tf.set_random_seed(1234)

    if not hasattr(backend, "tf"):
        raise RuntimeError("This tutorial requires keras to be configured to use the TensorFlow backend.")
    if keras.backend.image_dim_ordering() != 'tf':
        keras.backend.set_image_dim_ordering('tf')
        print("INFO: '~/.keras/keras.json' sets 'image_dim_ordering' to 'th', temporarily setting to 'tf'")
    # Create TF session and set as Keras backend session
    sess = tf.Session()
    keras.backend.set_session(sess)

    ###################### get_data #####################
    from read_data import read_saved_data
    from baseline import preprocess_training_data
    from keras.utils.np_utils import to_categorical
    model_type = 'mlp'
    window_size = 8
    basicInf_train, basicInf_valid, history_train, history_valid, y_train, y_valid = read_saved_data(window_size = window_size)
    x_train = preprocess_training_data(basicInf_train, history_train, model_type, window_size)
    x_valid = preprocess_training_data(basicInf_valid, history_valid, model_type, window_size)
    
    x_train, y_train = rolling(x_train, y_train)

    y_train = to_categorical(y_train, 2)
    y_valid = to_categorical(y_valid, 2)
    print(np.percentile(x_train, [0, 1, 50, 99, 100]))
    print(x_train[0], x_train[100])
   
    #exit()

    ###################### other preprocessing #####################
    # Use label smoothing
    label_smooth = .1
    y_train = y_train.clip(label_smooth / 9., 1. - label_smooth)
    
    # Define input TF placeholder
    input_shape = window_size*5+4
    x = tf.placeholder(tf.float32, shape=(None, input_shape))
    y = tf.placeholder(tf.float32, shape=(None, 2))

    # Define TF model graph
    model = naive_mlp_model(input_shape)
    preds = model(x)
    print("Defined TensorFlow model graph.")

    ###################### evaluation function #####################
    def evaluate():
        # Evaluate the accuracy of the MNIST model on legitimate test examples
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, x_valid, y_valid, args=eval_params)
        report.clean_train_clean_eval = acc
        print('Test accuracy on legitimate examples: %0.4f' % acc)

    # Train an MNIST model
    train_params = {'nb_epochs': nb_epochs, 'batch_size': batch_size, 'learning_rate': 1E-3, 'train_dir': train_dir, 'filename': filename}
    ckpt = tf.train.get_checkpoint_state(train_dir)
    ckpt_path = False if ckpt is None else ckpt.model_checkpoint_path

    rng = np.random.RandomState([2017, 8, 30])
    if load_model and ckpt_path:
        saver = tf.train.Saver()
        saver.restore(sess, ckpt_path)
        print("Model loaded from: {}".format(ckpt_path))
        evaluate()
    else:
        print("Model was not loaded, training from scratch.")
        model_train(sess, x, y, preds, x_train, y_train, evaluate=evaluate,args=train_params, save=False, rng=rng)

    # Calculate training error
    if testing:
        eval_params = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds, x_train, y_train, args=eval_params)
        report.train_clean_train_clean_eval = acc

    # Initialize the attack object and graph
    wrap = KerasModelWrapper(model)
    bim = BasicIterativeMethod(wrap, sess=sess)
    fgsm_params = {'eps': epsilon,'clip_min': -1.,'clip_max': 1.}
    adv_x = bim.generate(x, **fgsm_params)
    # Consider the attack to be constant
    adv_x = tf.stop_gradient(adv_x)
    preds_adv = model(adv_x)

    # Evaluate the accuracy of the MNIST model on adversarial examples
    eval_par = {'batch_size': batch_size}
    acc = model_eval(sess, x, y, preds_adv, x_valid, y_valid, args=eval_par)
    print('Test accuracy on adversarial examples: %0.4f\n' % acc)
    report.clean_train_adv_eval = acc

    # Calculating train error
    '''if testing:
        eval_par = {'batch_size': batch_size}
        acc = model_eval(sess, x, y, preds_adv, x_train, y_train, args=eval_par)
        report.train_clean_train_adv_eval = acc'''
    #return

    print("Repeating the process, using adversarial training")
    ####################### Redefine TF model graph ######################
    model_2 = naive_mlp_model(input_shape)
    preds_2 = model_2(x)
    wrap_2 = KerasModelWrapper(model_2)
    bim2 = BasicIterativeMethod(wrap_2, sess=sess)
    preds_2_adv = model_2(bim2.generate(x, **fgsm_params))

    def evaluate_2():
        # Accuracy of adversarially trained model on legitimate test inputs
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, x_valid, y_valid, args=eval_params)
        print('Test accuracy on legitimate examples: %0.4f' % accuracy)
        report.adv_train_clean_eval = accuracy

        # Accuracy of the adversarially trained model on adversarial examples
        accuracy = model_eval(sess, x, y, preds_2_adv, x_valid, y_valid, args=eval_params)
        print('Test accuracy on adversarial examples: %0.4f' % accuracy)
        report.adv_train_adv_eval = accuracy

    # Perform and evaluate adversarial training
    model_train(sess, x, y, preds_2, x_train, y_train,
                predictions_adv=preds_2_adv, evaluate=evaluate_2, args=train_params, save=False, rng=rng)
    import pdb
    pdb.set_trace()
    # Calculate training errors
    '''if testing:
        eval_params = {'batch_size': batch_size}
        accuracy = model_eval(sess, x, y, preds_2, X_train, Y_train,
                              args=eval_params)
        report.train_adv_train_clean_eval = accuracy
        accuracy = model_eval(sess, x, y, preds_2_adv, X_train,
                              Y_train, args=eval_params)
        report.train_adv_train_adv_eval = accuracy'''

    return report


def main(argv=None):
    #icu(nb_epochs=FLAGS.nb_epochs, batch_size=FLAGS.batch_size, learning_rate=FLAGS.learning_rate,
    #        train_dir=FLAGS.train_dir, filename=FLAGS.filename, load_model=FLAGS.load_model,  epsilon=epsilon)
    icu()


if __name__ == '__main__':
    flags.DEFINE_integer('nb_epochs', 30, 'Number of epochs to train model')
    flags.DEFINE_integer('batch_size', 128, 'Size of training batches')
    flags.DEFINE_float('learning_rate', 0.001, 'Learning rate for training')
    flags.DEFINE_string('train_dir', 'tmp', 'Directory where to save model.')
    flags.DEFINE_string('filename', 'mnist_BasicIterative_.ckpt', 'Checkpoint filename.')
    flags.DEFINE_boolean('load_model', True, 'Load saved model or train.')
    tf.app.run()