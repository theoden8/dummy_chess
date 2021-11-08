#!/usr/bin/env python3


import os
import sys
import tqdm
import typing

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as tfk
import tensorflow.keras.layers as tfkl

import dummy_chess


class NNUE:
    def __init__(self) -> None:
        def nnue_relu(x):
#            return tf.maximum(0, tf.minimum(127, tf.dtypes.cast(tf.math.floordiv(x, 64), tf.int32)))
            return tfk.activations.relu(x / 64, max_value=127)

        def clipped_relu(x):
            return tfk.activations.relu(x, max_value=127)

        halfkp1, halfkp2 = tfk.Input(shape=(41024,), sparse=True, dtype=tf.int8, name='persp1'), \
                           tfk.Input(shape=(41024,), sparse=True, dtype=tf.int8, name='persp2')
        feature_transformer = tfkl.Dense(units=256, activation=clipped_relu, name='feature_transformer')
        self.nnue = dummy_chess.NNUEDummy()
        self.model = tfk.Model(
            inputs=[halfkp1, halfkp2],
            outputs=tfkl.Dense(units=1, name='output')(
                tfkl.Dense(units=32, activation=nnue_relu, name='hidden_2')(
                    tfkl.Dense(units=32, activation=nnue_relu, name='hidden_1')(
                        tfkl.Concatenate()([feature_transformer(halfkp1), feature_transformer(halfkp2)]),
                    )
                )
            )
        )
        self.optimizer = tfk.optimizers.SGD(learning_rate=1e-3)
        self.affine_layer_inds = [2, 4, 5, 6]

    def __call__(self, x) -> tf.Tensor:
        return self.model(x) / (255 * 100)

    def load_weights(self, filename: str) -> None:
        self.nnue.load(filename)
        nnue_layers = self.nnue.layers
        for i in range(len(self.affine_layer_inds)):
            w, b = nnue_layers[i]
            self.model.layers[self.affine_layer_inds[i]].set_weights([w,b])

    def to_sparse_tensor(self, sparse: np.array):
        indices = np.pad(sparse.reshape(-1, 1), pad_width=[[0,0], [1,0]])
        values = np.ones(len(indices), dtype=np.float32)
        return tf.sparse.reorder(tf.sparse.SparseTensor(indices=indices, values=values, dense_shape=[1, 41024]))

    @tf.function
    def train_step(self, score: float, sparse1, sparse2):
        inputs = [sparse1, sparse2]
        with tf.GradientTape() as tape:
            predscore = self.model(inputs, training=True)
            loss = tf.square(predscore - score)
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss

    def save_weights(self, filename: str) -> None:
        # TODO set weights
        self.nnue.save(filename)


if __name__ == "__main__":
    to_here_path = os.path.relpath('.', os.path.abspath(os.path.dirname(sys.argv[0])))
    df = pd.read_hdf(os.path.join(to_here_path, 'data/sparse_dataset.h5'), key='sparse', mode='r')
    nnue = NNUE()
    epochs = 10
    dfscores = tf.convert_to_tensor(df.score, dtype=tf.float32)
    for epoch in range(epochs):
        for i in tqdm.trange(len(df)):
            sparse1 = nnue.to_sparse_tensor(np.array(df.sparse1[i], dtype=int))
            sparse2 = nnue.to_sparse_tensor(np.array(df.sparse2[i], dtype=int))
            score = dfscores[i]
            loss = nnue.train_step(score, sparse1, sparse2)
            if i % 50000 == 0:
                print(i, loss.numpy())
