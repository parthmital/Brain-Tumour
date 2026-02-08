import h5py
import numpy as np
import tensorflow as tf
import sys


def try_load(path):
    print(f"Loading {path}")
    with h5py.File(path, "r") as f:
        gw = f["model_weights"]
        base_model = tf.keras.applications.ResNet50(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
        x = tf.keras.layers.Dropout(0.5)(x)
        outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
        model = tf.keras.Model(inputs=base_model.input, outputs=outputs)
        print("Model built. Attempting manual weight load...")

        def find_ds(name, obj):
            if isinstance(obj, h5py.Dataset):
                if "conv2_block1_1_conv" in name:
                    print(f"DEBUG: Found {name} with shape {obj.shape}")

        gw.visititems(find_ds)


if __name__ == "__main__":
    try_load(sys.argv[1])
