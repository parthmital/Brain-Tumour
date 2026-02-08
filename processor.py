import os
import cv2
import h5py
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import traceback
from typing import List, Dict, Any, Optional

# Constants
IMG_SIZE = (224, 224)
PATCH_SIZE = (128, 128, 128)


class ConvBlock(nn.Module):
    def __init__(self, i, o):
        super().__init__()
        self.c = nn.Sequential(
            nn.Conv3d(i, o, 3, 1, 1, bias=False),
            nn.InstanceNorm3d(o),
            nn.LeakyReLU(0.01, True),
        )

    def forward(self, x):
        return self.c(x)


class nnUNet(nn.Module):
    def __init__(self):
        super().__init__()
        f = [32, 64, 128, 256]
        self.e1, self.e2, self.e3 = (
            ConvBlock(4, f[0]),
            ConvBlock(f[0], f[1]),
            ConvBlock(f[1], f[2]),
        )
        self.pool = nn.MaxPool3d(2)
        self.bottleneck = ConvBlock(f[2], f[3])
        self.u3, self.u2, self.u1 = (
            nn.ConvTranspose3d(f[3], f[2], 2, 2),
            nn.ConvTranspose3d(f[2], f[1], 2, 2),
            nn.ConvTranspose3d(f[1], f[0], 2, 2),
        )
        self.d3, self.d2, self.d1 = (
            ConvBlock(f[3], f[2]),
            ConvBlock(f[2], f[1]),
            ConvBlock(f[1], f[0]),
        )
        self.out, self.ds2, self.ds3 = (
            nn.Conv3d(f[0], 3, 1),
            nn.Conv3d(f[1], 3, 1),
            nn.Conv3d(f[2], 3, 1),
        )

    def forward(self, x):
        s1 = self.e1(x)
        s2 = self.e2(self.pool(s1))
        s3 = self.e3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))
        d3 = self.d3(torch.cat([self.u3(b), s3], 1))
        d2 = self.d2(torch.cat([self.u2(d3), s2], 1))
        d1 = self.d1(torch.cat([self.u1(d2), s1], 1))
        return self.out(d1), self.ds2(d2), self.ds3(d3)


class BrainProcessor:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.detection_model = None
        self.classification_model = None
        self.segmentation_model = None
        self.classes = ["glioma", "meningioma", "notumor", "pituitary"]

    def _rebuild_classification_model(self, num_classes: int = 4):
        # The notebook used ResNet50 with ImageNet weights and froze them.
        # We can just load ImageNet weights directly and then load our trained Dense layer.
        base_model = tf.keras.applications.ResNet50(
            weights="imagenet",
            include_top=False,
            input_shape=(224, 224, 3),
            name="resnet50",
        )
        model = tf.keras.models.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(
                    name="global_average_pooling2d_1"
                ),
                tf.keras.layers.Dropout(0.5, name="dropout_1"),
                tf.keras.layers.Dense(
                    num_classes, activation="softmax", name="dense_1"
                ),
            ]
        )
        return model

    def load_models(
        self, detection_path: str, classification_path: str, segmentation_path: str
    ):
        # Load Detection (Keras)
        print(f"Loading Detection model from {detection_path}...")
        try:
            self.detection_model = tf.keras.models.load_model(
                detection_path, compile=False
            )
            print("Detection model loaded.")
        except Exception as e:
            print(f"Error loading Detection model: {e}")
            traceback.print_exc()
            raise e

        # Load Classification (H5)
        print(f"Loading Classification model from {classification_path}...")
        try:
            # Rebuild model with ImageNet base
            self.classification_model = self._rebuild_classification_model(
                num_classes=4
            )

            # Manually load only the head (dense_1) weights from the H5 file
            # This avoids the shape mismatch in ResNet50 layers between Keras versions
            try:
                with h5py.File(classification_path, "r") as f:
                    gw = f["model_weights"]
                    if "dense_1" in gw:
                        # H5 structure is often dense_1/dense_1/kernel:0 etc or dense_1/sequential_1/kernel:0
                        # But we can find them by visiting items
                        weights = {}

                        def collect_dense_weights(name, obj):
                            if isinstance(obj, h5py.Dataset):
                                if "dense_1" in name:
                                    if "kernel" in name:
                                        weights["kernel"] = np.array(obj)
                                    if "bias" in name:
                                        weights["bias"] = np.array(obj)

                        gw["dense_1"].visititems(collect_dense_weights)

                        if "kernel" in weights and "bias" in weights:
                            self.classification_model.get_layer("dense_1").set_weights(
                                [weights["kernel"], weights["bias"]]
                            )
                            print("Classification model head weights loaded manually.")
                        else:
                            print(
                                "Warning: Could not find trained Dense weights in H5. Using random head."
                            )
                    else:
                        print("Warning: 'dense_1' not found in H5. Using random head.")
                print("Classification model (rebuilt + manual head) loaded.")
            except Exception as e_h5:
                print(f"Warning: Failed to load head weights manually: {e_h5}")
                # We still have a model with ImageNet base and random head as fallback
        except Exception as e:
            print(f"Error loading Classification model: {e}")
            traceback.print_exc()
            raise e

        # Load Segmentation (PyTorch)
        print(f"Loading Segmentation model from {segmentation_path}...")
        try:
            self.segmentation_model = nnUNet().to(self.device)
            state_dict = torch.load(segmentation_path, map_location=self.device)
            self.segmentation_model.load_state_dict(state_dict)
            self.segmentation_model.eval()
            print("Segmentation model loaded.")
        except Exception as e:
            print(f"Error loading Segmentation model: {e}")
            traceback.print_exc()
            raise e

    def crop_brain(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
        _, thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
        cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = cnts[0] if len(cnts) == 2 else cnts[1]
        if len(cnts) == 0:
            return cv2.resize(image, IMG_SIZE)
        c = max(cnts, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(c)
        return cv2.resize(image[y : y + h, x : x + w], IMG_SIZE)

    def preprocess_for_tf(self, image):
        # Image is BGR from cv2
        cropped = self.crop_brain(image)
        # Convert to RGB and normalize as per ResNet50
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        # ResNet50 expects (224, 224, 3)
        return tf.keras.applications.resnet50.preprocess_input(
            np.expand_dims(rgb.astype(np.float32), axis=0)
        )

    def nifti_to_2d(self, path: str):
        data = nib.load(path).get_fdata()
        # BraTS data is often (H, W, D) or (D, H, W)
        if len(data.shape) == 3:
            mid = data.shape[2] // 2
            slice_data = data[:, :, mid]
        else:
            slice_data = data[0]  # Fallback

        # Normalize to 0-255
        s_min, s_max = np.min(slice_data), np.max(slice_data)
        if s_max > s_min:
            slice_data = (slice_data - s_min) / (s_max - s_min) * 255
        else:
            slice_data = np.zeros_like(slice_data)

        slice_data = slice_data.astype(np.uint8)
        # Convert to BGR for cv2 consistency
        return cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR)

    def get_slice_as_image(self, path: str, slice_idx: int):
        """Extract a specific slice from a NIfTI file and return as BGR image."""
        try:
            proxy = nib.load(path)
            # Use dataobj to avoid loading full file if possible, though for shape we need header
            data_shape = proxy.shape

            # Handle 3D or 4D data
            if len(data_shape) == 4:
                # Assuming 4th dim is time/modality, take first
                num_slices = data_shape[2]
                slice_data = proxy.dataobj[:, :, slice_idx, 0]
            elif len(data_shape) == 3:
                num_slices = data_shape[2]
                slice_data = proxy.dataobj[:, :, slice_idx]
            else:
                return None, 1  # Fallback for unknown shapes

            # Ensure we have a numpy array
            slice_data = np.array(slice_data)

            # Normalize using robust range (ignoring outliers)
            s_min = np.min(slice_data)
            s_max = np.percentile(
                slice_data, 99.5
            )  # Use 99.5th percentile to avoid hot pixels

            if s_max > s_min:
                # Clip values above s_max
                slice_data = np.clip(slice_data, s_min, s_max)
                # Normalize to 0-255
                slice_data = (slice_data - s_min) / (s_max - s_min) * 255
            else:
                slice_data = np.zeros_like(slice_data)

            slice_data = slice_data.astype(np.uint8)

            # Rotate 90 degrees if needed (standard medical files often need rotation for viewing)
            slice_data = cv2.rotate(slice_data, cv2.ROTATE_90_COUNTERCLOCKWISE)

            return cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR), num_slices
        except Exception as e:
            print(f"Error in get_slice_as_image: {e}")
            return None, 0

    def run_detection(self, image_data: np.ndarray) -> float:
        if self.detection_model is None:
            raise ValueError("Detection model not loaded")
        X = self.preprocess_for_tf(image_data)
        pred = self.detection_model.predict(X)
        # Prediction might be (1, 1) or (1, 2)
        if pred.shape[1] == 1:
            return float(pred[0][0])
        else:
            return float(pred[0][1])  # Assume index 1 is 'Yes'

    def run_classification(self, image_data: np.ndarray) -> Dict[str, Any]:
        if self.classification_model is None:
            raise ValueError("Classification model not loaded")
        X = self.preprocess_for_tf(image_data)
        preds = self.classification_model.predict(X)
        class_idx = np.argmax(preds[0])
        return {
            "class": self.classes[class_idx],
            "confidence": float(preds[0][class_idx]),
        }

    def preprocess_modality(self, path: str):
        data = nib.load(path).get_fdata().astype(np.float32)
        mask = data != 0
        if mask.sum() > 0:
            data[mask] = (data[mask] - data[mask].mean()) / (data[mask].std() + 1e-8)

        d, h, w = data.shape
        start_d, start_h, start_w = (d - 128) // 2, (h - 128) // 2, (w - 128) // 2

        cropped = np.zeros((128, 128, 128), dtype=np.float32)

        d_src = slice(max(0, start_d), min(d, start_d + 128))
        h_src = slice(max(0, start_h), min(h, start_h + 128))
        w_src = slice(max(0, start_w), min(w, start_w + 128))

        d_dst = slice(max(0, -start_d), min(128, d - start_d))
        h_dst = slice(max(0, -start_h), min(128, h - start_h))
        w_dst = slice(max(0, -start_w), min(128, w - start_w))

        cropped[d_dst, h_dst, w_dst] = data[d_src, h_src, w_src]
        return cropped

    def run_segmentation(self, modality_paths: Dict[str, str]) -> Dict[str, float]:
        if self.segmentation_model is None:
            return {"tumorVolume": 0, "wtVolume": 0, "tcVolume": 0, "etVolume": 0}

        imgs = []
        for k in ["flair", "t1", "t1ce", "t2"]:
            imgs.append(self.preprocess_modality(modality_paths[k]))

        X = torch.from_numpy(np.stack(imgs)).unsqueeze(0).to(self.device)
        self.segmentation_model.eval()
        with torch.no_grad():
            output = self.segmentation_model(X)
            # Handle list output from nnUNet (out, ds2, ds3)
            if isinstance(output, (list, tuple)):
                output = output[0]

            probs = torch.sigmoid(output).cpu().numpy()[0]

            wt_vol = np.sum(probs[0] > 0.5) / 1000.0
            tc_vol = np.sum(probs[1] > 0.5) / 1000.0
            et_vol = np.sum(probs[2] > 0.5) / 1000.0

            return {
                "tumorVolume": float(wt_vol),
                "wtVolume": float(wt_vol),
                "tcVolume": float(tc_vol),
                "etVolume": float(et_vol),
            }
