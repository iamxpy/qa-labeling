# -*- coding: utf-8 -*-

"""## Setup

The Keras mixed precision API is available in TensorFlow 2.1.
"""
# pip install tf-nightly

from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.mixed_precision import experimental as mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)

"""The policy specifies two important aspects of a layer: the dtype the layer's computations are done in, and the dtype of a layer's variables. Above, you created a `mixed_float16` policy (i.e., a `mixed_precision.Policy` created by passing the string `'mixed_float16'` to its constructor). With this policy, layers use float16 computations and float32 variables. Computations are done in float16 for performance, but variables must be kept in float32 for numeric stability. You can directly query these properties of the policy."""

print('Compute dtype: %s' % policy.compute_dtype)
print('Variable dtype: %s' % policy.variable_dtype)

"""As mentioned before, the `mixed_float16` policy will most significantly improve performance on NVIDIA GPUs with compute capability of at least 7.0. The policy will run on other GPUs and CPUs but may not improve performance. For TPUs, the `mixed_bfloat16` policy should be used instead.

## Building the model

Next, let's start building a simple model. Very small toy models typically do not benefit from mixed precision, because overhead from the TensorFlow runtime  typically dominates the execution time, making any performance improvement on the GPU negligible. Therefore, let's build two large `Dense` layers with 4096 units each if a GPU is used.
"""

inputs = keras.Input(shape=(784,), name='digits')
if tf.config.list_physical_devices('GPU'):
  print('The model will run with 4096 units on a GPU')
  num_units = 4096
else:
  # Use fewer units on CPUs so the model finishes in a reasonable amount of time
  print('The model will run with 64 units on a CPU')
  num_units = 64
dense1 = layers.Dense(num_units, activation='relu', name='dense_1')
x = dense1(inputs)
dense2 = layers.Dense(num_units, activation='relu', name='dense_2')
x = dense2(x)

"""Each layer has a policy and uses the global policy by default. Each of the `Dense` layers therefore have the `mixed_float16` policy because you set the global policy to `mixed_float16` previously. This will cause the dense layers to do float16 computations and have float32 variables. They cast their inputs to float16 in order to do float16 computations, which causes their outputs to be float16 as a result. Their variables are float32 and will be cast to float16 when the layers are called to avoid errors from dtype mismatches."""

print('x.dtype: %s' % x.dtype.name)
# 'kernel' is dense1's variable
print('dense1.kernel.dtype: %s' % dense1.kernel.dtype.name)

"""Next, create the output predictions. Normally, you can create the output predictions as follows, but this is not always numerically stable with float16."""

# INCORRECT: softmax and model output will be float16, when it should be float32
outputs = layers.Dense(10, activation='softmax', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)

"""A softmax activation at the end of the model should be float32. Because the dtype policy is `mixed_float16`, the softmax activation would normally have a float16 compute dtype and output a float16 tensors.

This can be fixed by separating the Dense and softmax layers, and by passing `dtype='float32'` to the softmax layer
"""

# CORRECT: softmax and model output are float32
x = layers.Dense(10, name='dense_logits')(x)
outputs = layers.Activation('softmax', dtype='float32', name='predictions')(x)
print('Outputs dtype: %s' % outputs.dtype.name)

"""Passing `dtype='float32'` to the softmax layer constructor overrides the layer's dtype policy to be the `float32` policy, which does computations and keeps variables in float32. Equivalently, we could have instead passed `dtype=mixed_precision.Policy('float32')`; layers always convert the dtype argument to a policy. Because the `Activation` layer has no variables, the policy's variable dtype is ignored, but the policy's compute dtype of float32 causes softmax and the model output to be float32. 


Adding a float16 softmax in the middle of a model is fine, but a softmax at the end of the model should be in float32. The reason is that if the intermediate tensor flowing from the softmax to the loss is float16 or bfloat16, numeric issues may occur.

You can override the dtype of any layer to be float32 by passing `dtype='float32'` if you think it will not be numerically stable with float16 computations. But typically, this is only necessary on the last layer of the model, as most layers have sufficient precision with `mixed_float16` and `mixed_bfloat16`.

If the model does not end in a softmax, the outputs should still be float32. While unnecessary for this model, the model outputs can be cast to float32 with the following.:
"""

# The linear activation is an identity function. So this simply casts 'outputs'
# to float32. In this particular case, 'outputs' is already float32 so this is a
# no-op.
outputs = layers.Activation('linear', dtype='float32')(outputs)

"""Next, finish and compile the model, and generate input data."""

model = keras.Model(inputs=inputs, outputs=outputs)
model.compile(loss='sparse_categorical_crossentropy',
              optimizer=keras.optimizers.RMSprop(),
              metrics=['accuracy'])

(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
x_train = x_train.reshape(60000, 784).astype('float32') / 255
x_test = x_test.reshape(10000, 784).astype('float32') / 255

"""This example cast the input data from int8 to float32. We don't cast to float16 since the division by 255 is on the CPU, which runs float16 operations slower than float32 operations. In this case, the performance difference in negligible, but in general you should run input processing math in float32 if it runs on the CPU. The first layer of the model will cast the inputs to float16, as each layer casts floating-point inputs to its compute dtype.

The initial weights of the model are retrieved. This will allow training from scratch again by loading the weights.
"""

initial_weights = model.get_weights()

"""## Training the model with Model.fit

Next, train the model.
"""

history = model.fit(x_train, y_train,
                    batch_size=8192,
                    epochs=5,
                    validation_split=0.2)
test_scores = model.evaluate(x_test, y_test, verbose=2)
print('Test loss:', test_scores[0])
print('Test accuracy:', test_scores[1])

"""## GPU performance tips

Here are some performance tips when using mixed precision on GPUs.

### Increasing your batch size
If it doesn't affect model quality, try running with double the batch size when using mixed precision. As float16 tensors use half the memory, this often allows you to double your batch size without running out of memory. Increasing batch size typically increases training throughput, i.e. the training elements per second your model can run on.

### Ensuring GPU Tensor Cores are used

As mentioned previously, modern NVIDIA GPUs use a special hardware unit called Tensor Cores that can multiply float16 matrices very quickly. However, Tensor Cores requires certain dimensions of tensors to be a multiple of 8. In the examples below, an argument is bold if and only if it needs to be a multiple of 8 for Tensor Cores to be used.

* tf.keras.layers.Dense(**units=64**)
* tf.keras.layers.Conv2d(**filters=48**, kernel_size=7, stride=3)
  * And similarly for other convolutional layers, such as tf.keras.layers.Conv3d
* tf.keras.layers.LSTM(**units=64**)
  * And similar for other RNNs, such as tf.keras.layers.GRU
* tf.keras.Model.fit(epochs=2, **batch_size=128**)

You should try to use Tensor Cores when possible. If you want to learn more [NVIDIA deep learning performance guide](https://docs.nvidia.com/deeplearning/sdk/dl-performance-guide/index.html) describes the exact requirements for using Tensor Cores as well as other Tensor Core-related performance information.

### XLA

XLA is a compiler that can further increase mixed precision performance, as well as float32 performance to a lesser extent. See the [XLA guide](https://www.tensorflow.org/xla) for details.

## Cloud TPU performance tips
As on GPUs, you should try doubling your batch size, as bfloat16 tensors use half the memory. Doubling batch size may increase training throughput.

TPUs do not require any other mixed precision-specific tuning to get optimal performance. TPUs already require the use of XLA. They benefit from having certain dimensions being multiples of 128, but this applies equally to float32 as it does for mixed precision. See the [Cloud TPU Performance Guide](https://cloud.google.com/tpu/docs/performance-guide) for general TPU performance tips, which apply to mixed precision as well as float32.

## Summary
* You should use mixed precision if you use TPUs or NVIDIA GPUs with at least compute capability 7.0, as it will improve performance by up to 3x.
* You can use mixed precision with the following lines:
  ```
  # On TPUs, use 'mixed_bfloat16' instead
  policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16')
  mixed_precision.set_policy(policy)
  ```
* If your model ends in softmax, make sure it is float32. And regardless of what your model ends in, make sure the output is float32.
* If you use a custom training loop with `mixed_float16`, in addition to the above lines, you need to wrap your optimizer with a `tf.keras.mixed_precision.experimental.LossScaleOptimizer`. Then call `optimizer.get_scaled_loss` to scale the loss, and `optimizer.get_unscaled_gradients` to unscale the gradients.
* Double the training batch size if it does not reduce evaluation accuracy
* On GPUs, ensure most tensor dimensions are a multiple of 8 to maximize performance

For more examples of mixed precision using the `tf.keras.mixed_precision` API, see the [official models repository](https://github.com/tensorflow/models/tree/master/official). Most official models, such as [ResNet](https://github.com/tensorflow/models/tree/master/official/vision/image_classification) and [Transformer](https://github.com/tensorflow/models/tree/master/official/transformer/v2) will run using mixed precision by passing `--dtype=fp16`.
"""