

import numpy as np
import onnx
from onnx_tf.backend import prepare
import tensorflow as tf
from sklearn.metrics import accuracy_score
from train_pytorch import preprocess, load_mnist


def load_converted_model():


    x_test, y_test = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    x_test = preprocess(x_test)
    x_test = x_test.astype(np.float32)
    print(x_test.shape)
    
    # Load a converted model
    # onnx-tf convert -i onnx-cnn-model.onnx -o tf-model-converted-from-onnx
    model = tf.saved_model.load('tf-model-converted-from-onnx')
    input_spec = model.signatures['serving_default'].structured_input_signature[1]['input']
    input_shape = input_spec.shape

    assert x_test.shape[1:] == input_shape[1:], f'x_test.shape[1:]: {x_test.shape[1:]} != input_shape[1:]: {input_shape[1:]}'

    # Get the output name (only one)
    outputs = model.signatures['serving_default'].structured_outputs
    output_name = list(outputs.keys())[0]

    infer = model.signatures["serving_default"]
    
    preds = infer(tf.constant(x_test))[output_name]

    print('preds.shape:', preds.shape)
    preds_onecold = np.argmax(preds, axis=1)
    print('accuracy:', accuracy_score(y_test, preds_onecold))




def load_onnx_model():

    x_test, y_test = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    x_test = preprocess(x_test)
    x_test = x_test.astype(np.float32)


    onnx_model = onnx.load("onnx-cnn-model.onnx")
    tf_model = prepare(onnx_model)

    preds = tf_model.run(x_test).output
    
    preds_onecold = np.argmax(preds, axis=1)
    print('accuracy:', accuracy_score(y_test, preds_onecold))


    
if __name__ == '__main__':
    
    print('Running converted model:')
    load_onnx_model()
    print('\n\n')

    print('Running ONNX model directly:')
    load_onnx_model()
    