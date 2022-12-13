import warnings
warnings.filterwarnings('ignore')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.applications import vgg19
from PIL import Image
img_nrows = 1
img_ncols = 1


def gram_matrix(x):
	# The gram matrix of an image tensor (feature-wise outer product)
    x = tf.transpose(x, (2, 0, 1))
    features = tf.reshape(x, (tf.shape(x)[0], -1))
    gram = tf.matmul(features, tf.transpose(features))
    return gram

def style_loss(style, main):
    C = gram_matrix(main)
    S = gram_matrix(style)
    channels = 3
    size = 100
    return tf.reduce_sum(tf.square(S - C)) / (4.0 * (channels**2) * (size**2))

def content_loss(base, main):
    return tf.reduce_sum(tf.square(main - base))
    
def total_variation_loss(x):
    a = tf.square(x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, 1:, : img_ncols - 1, :])
    b = tf.square(x[:, : img_nrows - 1, : img_ncols - 1, :] - x[:, : img_nrows - 1, 1:, :])
    return tf.reduce_sum(tf.pow(a + b, 1.25))

model = vgg19.VGG19(weights="imagenet", include_top=False)

# Get the symbolic outputs of each "key" layer (we gave them unique names).
outputs_dict = dict([(layer.name, layer.output) for layer in model.layers])

# Set up a model that returns the activation values for every layer in
# VGG19 (as a dict).
feature_extractor = keras.Model(inputs=model.inputs, outputs=outputs_dict)

# List of layers to use for the style loss.
style_layer_names = ["block1_conv1","block2_conv1","block3_conv1",
                     "block4_conv1","block5_conv1"]



# The layer to use for the content loss.
content_layer_name = "block5_conv2"


def compute_loss(combination_image, base_image, style_reference_image,
                 content_weight, style_weight, total_variation_weight):
    input_tensor = tf.concat(
        [base_image, style_reference_image, combination_image], axis=0)
    
    features = feature_extractor(input_tensor)

    # Initialize the loss
    loss = tf.zeros(shape=())

    # Add content loss
    layer_features = features[content_layer_name]
    base_image_features = layer_features[0, :, :, :]
    combination_features = layer_features[2, :, :, :]
    loss = loss + content_weight * content_loss(
        base_image_features, combination_features
    )
    # Add style loss
    for layer_name in style_layer_names:
        layer_features = features[layer_name]
        style_reference_features = layer_features[1, :, :, :]
        combination_features = layer_features[2, :, :, :]
        sl = style_loss(style_reference_features, combination_features)
        loss += (style_weight / len(style_layer_names)) * sl

    # Add total variation loss
    loss += total_variation_weight * total_variation_loss(combination_image)
    return loss


def compute_loss_and_grads(combination_image, base_image, style_reference_image):
    with tf.GradientTape() as tape:
        loss = compute_loss(combination_image, base_image, style_reference_image)
    grads = tape.gradient(loss, combination_image)
    return loss, grads
    
optimizer = keras.optimizers.SGD(
    keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=100.0, decay_steps=100, decay_rate=0.96
    )
)

# base_image = preprocess_image(base_image_path)
# style_reference_image = preprocess_image(style_reference_image_path)
# combination_image = tf.Variable(preprocess_image(base_image_path))

# iterations = 2000
# for i in range(1, iterations + 1):
#     loss, grads = compute_loss_and_grads(
#         combination_image, base_image, style_reference_image
#     )
#     optimizer.apply_gradients([(grads, combination_image)])
#     if i % 100 == 0:
#         print("Iteration %d: loss=%.2f" % (i, loss))
#         img = deprocess_image(combination_image.numpy())
#         fname = result_prefix + "_at_iteration_%d.png" % i
#         keras.preprocessing.image.save_img(fname, img)







