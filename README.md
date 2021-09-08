# Model Zoo: ONNX example 

This example shows how to save, load and use models in the 
[Open Neural Network Exchange (ONNX)](onnx.ai) format.

We use the Fashion-MNIST dataset, which can be downloaded as so:
```
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/train-images-idx3-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-labels-idx1-ubyte.gz
wget http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/t10k-images-idx3-ubyte.gz
```

First, train a simple CNN using PyTorch:

```
python train_pytorch.py
```

This will save a model in the ONNX format: `onnx-cnn-model.onnx`.
We can now perform inference on test data using the ONNX runtime, by running

```
python run_onnx.py
```

Next, use the (TensorFlow backend for ONNX)[https://github.com/onnx/onnx-tensorflow]
to convert the saved model into TensorFlow format:

```
onnx-tf convert -i onnx-cnn-model.onnx -o tf-model-converted-from-onnx
```

Finally, we can 1) load the converted model, or 2) convert the ONNX model to
TensorFlow on the fly. Both methods are done in `load_in_tensorflow.py`:
```
python load_in_tensorflow.py
```
