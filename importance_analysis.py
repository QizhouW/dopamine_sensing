# author: Qizhou Wang
# datetime: 27/3/2024 27/3/2024
# email: imjoewang@gmail.com
"""
This module 
"""
from cross_val import parse_and_split, train_test
import tensorflow as tf
import numpy as np


def get_gradients(input_vector, model):
    """Computes the gradients of model output w.r.t an input vector for regression.

    Args:
        input_vector: 2D tensor of shape (1, 8) representing the input vector.
        model: The regression model.

    Returns:
        Gradients of the model output w.r.t input_vector.
    """
    vector = tf.cast(input_vector, tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(vector)
        preds = model(vector)

    grads = tape.gradient(preds, vector)
    return grads


def get_integrated_gradients(input_vector, model, baseline=None, num_steps=50):
    """Computes Integrated Gradients for a regression model output.

    Args:
        input_vector (ndarray): Original 8-point input vector.
        model: The regression model.
        baseline (ndarray): The baseline vector to start with for interpolation.
        num_steps: Number of interpolation steps between the baseline
            and the input used in the computation of integrated gradients.

    Returns:
        Integrated gradients w.r.t input vector.
    """
    if baseline is None:
        baseline = np.zeros((8,)).astype(np.float32)
    else:
        baseline = baseline.astype(np.float32)

    input_vector = input_vector.astype(np.float32)
    interpolated_vectors = [
        baseline + (step / num_steps) * (input_vector - baseline)
        for step in range(num_steps + 1)
    ]
    interpolated_vectors = np.array(interpolated_vectors).astype(np.float32)

    grads = []
    for i, vec in enumerate(interpolated_vectors):
        vec = tf.expand_dims(vec, axis=0)
        grad = get_gradients(vec, model)
        grads.append(grad[0])
    grads = tf.convert_to_tensor(grads, dtype=tf.float32)

    grads = (grads[:-1] + grads[1:]) / 2.0
    avg_grads = tf.reduce_mean(grads, axis=0)

    integrated_grads = (input_vector - baseline) * avg_grads
    return integrated_grads

def random_baseline_integrated_gradients(input_vector, model, num_steps=50, num_runs=2):
    integrated_grads = []
    for run in range(num_runs):
        baseline = np.random.rand(8).astype(np.float32)  # Match the input vector dimensions
        igrads = get_integrated_gradients(
            input_vector=input_vector,
            model=model,
            baseline=baseline,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)


def zero_baseline_integrated_gradients(input_vector, model, num_steps=50, num_runs=2):
    integrated_grads = []
    for run in range(num_runs):
        igrads = get_integrated_gradients(
            input_vector=input_vector,
            model=model,
            baseline=None,
            num_steps=num_steps,
        )
        integrated_grads.append(igrads)
    integrated_grads = tf.convert_to_tensor(integrated_grads)
    return tf.reduce_mean(integrated_grads, axis=0)


