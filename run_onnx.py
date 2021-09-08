""" 
Run saved model in the onnx runtime

https://github.com/onnx/onnx-docker/blob/master/onnx-ecosystem/inference_demos/resnet50_modelzoo_onnxruntime_inference.ipynb
"""
import gzip
import numpy as np
import onnxruntime
from sklearn.metrics import accuracy_score
from train_pytorch import load_mnist, preprocess


if __name__ == '__main__':

    x_test, y_test = load_mnist('t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz')
    x_test = preprocess(x_test)
    print(x_test.shape)
        
    # Run the model on the backend
    session = onnxruntime.InferenceSession('onnx-cnn-model.onnx', None)

    # Get input info
    input_node = session.get_inputs()[0]
    input_shape = tuple(input_node.shape[1:])
    input_name = input_node.name

    # Convert to the correct data type
    x_test = x_test.astype(np.float32)

    print('input_shape:', input_shape)
    assert x_test.shape[1:] == input_shape, f'x_test.shape: {x_test.shape} != input_shape: {input_shape}'

    # Predict
    preds = session.run([], {input_name: x_test})[0]

    # Compute accuracy
    preds_onecold = np.argmax(preds, axis=1)
    print('accuracy:', accuracy_score(y_test, preds_onecold))
    