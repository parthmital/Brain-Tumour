import os
import cv2
import numpy as np
import tensorflow as tf
import torch
import torch.nn as nn
import torch.nn.functional as F
import nibabel as nib
import traceback
from typing import List, Dict, Any, Optional

IMG_SIZE = 224, 224
PATCH_SIZE = 128, 128, 128


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

    def _rebuild_classification_model(self):
        # Rebuilding architecture to avoid "received 2 input tensors" error in .keras config
        base_model = tf.keras.applications.ResNet50(
            weights=None, include_top=False, input_shape=(224, 224, 3)
        )
        model = tf.keras.models.Sequential(
            [
                base_model,
                tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling2d"),
                tf.keras.layers.Dropout(0.5, name="dropout"),
                tf.keras.layers.Dense(4, activation="softmax", name="dense"),
            ]
        )
        return model

    def load_models(
        self, detection_path: str, classification_path: str, segmentation_path: str
    ):
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
        print(f"Loading Classification model from {classification_path}...")
        try:
            self.classification_model = self._rebuild_classification_model()
            self.classification_model.load_weights(classification_path)
            print("Classification model loaded successfully (rebuilt + weights).")
        except Exception as e:
            print(f"Error loading Classification model: {e}")
            traceback.print_exc()
            raise e
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
        cropped = self.crop_brain(image)
        rgb = cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB)
        return tf.keras.applications.resnet50.preprocess_input(
            np.expand_dims(rgb.astype(np.float32), axis=0)
        )

    def nifti_to_2d(self, path: str):
        data = nib.load(path).get_fdata()
        if len(data.shape) == 3:
            mid = data.shape[2] // 2
            slice_data = data[:, :, mid]
        else:
            slice_data = data[0]
        s_min, s_max = np.min(slice_data), np.max(slice_data)
        if s_max > s_min:
            slice_data = (slice_data - s_min) / (s_max - s_min) * 255
        else:
            slice_data = np.zeros_like(slice_data)
        slice_data = slice_data.astype(np.uint8)
        return cv2.cvtColor(slice_data, cv2.COLOR_GRAY2BGR)

    def get_slice_as_image(self, path: str, slice_idx: int):
        try:
            proxy = nib.load(path)
            data_shape = proxy.shape
            if len(data_shape) == 4:
                num_slices = data_shape[2]
                slice_data = proxy.dataobj[:, :, slice_idx, 0]
            elif len(data_shape) == 3:
                num_slices = data_shape[2]
                slice_data = proxy.dataobj[:, :, slice_idx]
            else:
                return None, 1
            slice_data = np.array(slice_data)
            s_min = np.min(slice_data)
            s_max = np.percentile(slice_data, 99.5)
            if s_max > s_min:
                slice_data = np.clip(slice_data, s_min, s_max)
                slice_data = (slice_data - s_min) / (s_max - s_min) * 255
            else:
                slice_data = np.zeros_like(slice_data)
            slice_data = slice_data.astype(np.uint8)
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
        if pred.shape[1] == 1:
            return float(pred[0][0])
        else:
            return float(pred[0][1])

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
            data[mask] = (data[mask] - data[mask].mean()) / (data[mask].std() + 1e-08)
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

    def run_segmentation(
        self, modality_paths: Dict[str, str], save_path: Optional[str] = None
    ) -> Dict[str, float]:
        if self.segmentation_model is None:
            return {"tumorVolume": 0, "wtVolume": 0, "tcVolume": 0, "etVolume": 0}
        imgs = []
        for k in ["flair", "t1", "t1ce", "t2"]:
            imgs.append(self.preprocess_modality(modality_paths[k]))
        X = torch.from_numpy(np.stack(imgs)).unsqueeze(0).to(self.device)
        self.segmentation_model.eval()
        with torch.no_grad():
            output = self.segmentation_model(X)
            if isinstance(output, (list, tuple)):
                output = output[0]
            probs = torch.sigmoid(output).cpu().numpy()[0]
            wt_vol = np.sum(probs[0] > 0.5) / 1e3
            tc_vol = np.sum(probs[1] > 0.5) / 1e3
            et_vol = np.sum(probs[2] > 0.5) / 1e3
            if save_path:
                try:
                    ref_img = nib.load(modality_paths["flair"])
                    d, h, w = ref_img.shape
                    affine = ref_img.affine

                    # Create a single 3D uint8 mask with discrete labels
                    # Label 1: WT (Whole Tumor), Label 2: TC (Tumor Core), Label 3: ET (Enhancing Tumor)
                    seg_mask = np.zeros((d, h, w), dtype=np.uint8)

                    start_d, start_h, start_w = (
                        (d - 128) // 2,
                        (h - 128) // 2,
                        (w - 128) // 2,
                    )
                    d_dst = slice(max(0, start_d), min(d, start_d + 128))
                    h_dst = slice(max(0, start_h), min(h, start_h + 128))
                    w_dst = slice(max(0, start_w), min(w, start_w + 128))

                    d_src = slice(max(0, -start_d), min(128, d - start_d))
                    h_src = slice(max(0, -start_h), min(128, h - start_h))
                    w_src = slice(max(0, -start_w), min(128, w - start_w))

                    # We map probabilities to discrete labels
                    # More specific labels overwrite general ones
                    local_wt = (probs[0, d_src, h_src, w_src] > 0.5).astype(np.uint8)
                    local_tc = (probs[1, d_src, h_src, w_src] > 0.5).astype(np.uint8)
                    local_et = (probs[2, d_src, h_src, w_src] > 0.5).astype(np.uint8)

                    local_seg = np.zeros((128, 128, 128), dtype=np.uint8)
                    local_seg[local_wt == 1] = 1
                    local_seg[local_tc == 1] = 2
                    local_seg[local_et == 1] = 3

                    seg_mask[d_dst, h_dst, w_dst] = local_seg[d_src, h_src, w_src]

                    # Save as uncompressed .nii for better compatibility and to rule out Gzip issues
                    final_path = save_path.replace(".nii.gz", ".nii")
                    nib.save(nib.Nifti1Image(seg_mask, affine), final_path)
                    print(f"Segmentation mask saved to {final_path}")
                except Exception as e:
                    print(f"Failed to save segmentation mask: {e}")
                    traceback.print_exc()
            return {
                "tumorVolume": float(wt_vol),
                "wtVolume": float(wt_vol),
                "tcVolume": float(tc_vol),
                "etVolume": float(et_vol),
            }

    def get_segmentation_slice(self, path: str, slice_idx: int):
        try:
            if not os.path.exists(path):
                return
            proxy = nib.load(path)
            # Now expecting (W, H, D) uint8
            if len(proxy.shape) != 3:
                print(f"Unexpected mask shape: {proxy.shape}")
                return
            slice_data = proxy.dataobj[:, :, slice_idx]
            slice_data = np.array(slice_data)
            slice_data = np.rot90(slice_data, 1)

            # Map labels to RGB for visualization in the 2D slice viewer.
            # Verified against BraTS2020 notebook: WT=Ch0(R), TC=Ch1(G), ET=Ch2(B).
            # Regions ET ⊂ TC ⊂ WT, so labels are mapped to all parent channels.
            h, w = slice_data.shape
            rgb_mask = np.zeros((h, w, 3), dtype=np.uint8)

            # Label 1: WT (Whole Tumor)
            rgb_mask[slice_data == 1, 0] = 255
            # Label 2: TC (Tumor Core + WT)
            rgb_mask[slice_data == 2, 0] = 255
            rgb_mask[slice_data == 2, 1] = 255
            # Label 3: ET (Enhancing Tumor + TC + WT)
            rgb_mask[slice_data == 3, 0] = 255
            rgb_mask[slice_data == 3, 1] = 255
            rgb_mask[slice_data == 3, 2] = 255

            return rgb_mask
        except Exception as e:
            print(f"Error in get_segmentation_slice: {e}")
            return
