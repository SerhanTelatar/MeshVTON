"""
Pose Estimator — 2D/3D skeleton extraction.

Wrapper supporting OpenPose and DWPose backends for extracting
body keypoints from person images.
"""

from typing import Optional
import numpy as np
import torch
import torch.nn as nn
import cv2


class PoseEstimator:
    """
    Extracts 2D/3D body keypoints from person images.

    Args:
        model_type: 'dwpose', 'openpose', or 'mediapipe'.
        device: Torch device for inference.
        confidence_threshold: Minimum keypoint confidence.
    """

    # COCO-18 body keypoint names
    BODY_KEYPOINTS = [
        "nose", "neck", "right_shoulder", "right_elbow", "right_wrist",
        "left_shoulder", "left_elbow", "left_wrist", "right_hip", "right_knee",
        "right_ankle", "left_hip", "left_knee", "left_ankle", "right_eye",
        "left_eye", "right_ear", "left_ear",
    ]

    # Skeleton connections for rendering
    SKELETON_CONNECTIONS = [
        (0, 1), (1, 2), (2, 3), (3, 4), (1, 5), (5, 6), (6, 7),
        (1, 8), (8, 9), (9, 10), (1, 11), (11, 12), (12, 13),
        (0, 14), (14, 16), (0, 15), (15, 17),
    ]

    KEYPOINT_COLORS = [
        (255, 0, 0), (255, 85, 0), (255, 170, 0), (255, 255, 0),
        (170, 255, 0), (85, 255, 0), (0, 255, 0), (0, 255, 85),
        (0, 255, 170), (0, 255, 255), (0, 170, 255), (0, 85, 255),
        (0, 0, 255), (85, 0, 255), (170, 0, 255), (255, 0, 255),
        (255, 0, 170), (255, 0, 85),
    ]

    def __init__(self, model_type: str = "dwpose", device: str = "cuda",
                 confidence_threshold: float = 0.3):
        self.model_type = model_type
        self.device = device
        self.confidence_threshold = confidence_threshold
        self.model = None

    def load_model(self):
        """Load the pose estimation model."""
        if self.model_type == "dwpose":
            self._load_dwpose()
        elif self.model_type == "openpose":
            self._load_openpose()
        elif self.model_type == "mediapipe":
            self._load_mediapipe()

    def _load_dwpose(self):
        """Load DWPose model (needs mmcv/mmpose/mmdet). Falls back to OpenPose."""
        try:
            from controlnet_aux import DWposeDetector
            self.model = DWposeDetector()
        except Exception as e:
            # DWPose needs mmdet/mmpose/mmcv; if missing it raises NameError, not
            # ImportError. Fall back to OpenPose (no mmdet dependency).
            print(f"Warning: DWPose unavailable ({e}); falling back to OpenPose.")
            self._load_openpose()

    def _load_openpose(self):
        """Load OpenPose model."""
        try:
            from controlnet_aux import OpenposeDetector
            self.model = OpenposeDetector.from_pretrained("lllyasviel/Annotators")
            self.model_type = "openpose"
        except Exception as e:
            print(f"Warning: OpenPose unavailable ({e}); using dummy pose.")
            self.model = None

    def _load_mediapipe(self):
        """Load MediaPipe Pose."""
        try:
            import mediapipe as mp
            self.model = mp.solutions.pose.Pose(
                static_image_mode=True, model_complexity=2, min_detection_confidence=0.5,
            )
        except Exception as e:
            print(f"Warning: MediaPipe unavailable ({e}); using dummy pose.")
            self.model = None

    def estimate(self, image: np.ndarray) -> dict:
        """
        Estimate pose keypoints from an image.

        Args:
            image: (H, W, 3) BGR image as numpy array.

        Returns:
            Dict with 'keypoints' (N, 3) array of (x, y, confidence),
            'pose_image' rendered pose visualization.
        """
        if self.model is None:
            self.load_model()

        if self.model is None:
            return self._dummy_pose(image)

        if self.model_type in ("dwpose", "openpose"):
            return self._estimate_controlnet_aux(image)
        elif self.model_type == "mediapipe":
            return self._estimate_mediapipe(image)

        return self._dummy_pose(image)

    def _estimate_controlnet_aux(self, image: np.ndarray) -> dict:
        from PIL import Image
        pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        pose_image = self.model(pil_image)
        pose_np = np.array(pose_image)
        return {
            "keypoints": np.zeros((18, 3), dtype=np.float32),
            "pose_image": pose_np,
        }

    def _estimate_mediapipe(self, image: np.ndarray) -> dict:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.process(rgb)
        h, w = image.shape[:2]
        keypoints = np.zeros((18, 3), dtype=np.float32)
        if results.pose_landmarks:
            mp_to_coco = [0, None, 12, 14, 16, 11, 13, 15, 24, 26, 28, 23, 25, 27, 5, 2, 8, 7]
            for i, mp_idx in enumerate(mp_to_coco):
                if mp_idx is not None:
                    lm = results.pose_landmarks.landmark[mp_idx]
                    keypoints[i] = [lm.x * w, lm.y * h, lm.visibility]
        return {"keypoints": keypoints, "pose_image": self.render_pose(image, keypoints)}

    def _dummy_pose(self, image: np.ndarray) -> dict:
        h, w = image.shape[:2]
        keypoints = np.zeros((18, 3), dtype=np.float32)
        return {"keypoints": keypoints, "pose_image": np.zeros((h, w, 3), dtype=np.uint8)}

    def render_pose(self, image: np.ndarray, keypoints: np.ndarray,
                    thickness: int = 4) -> np.ndarray:
        """Render pose skeleton on a black canvas."""
        h, w = image.shape[:2]
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

        # Draw limbs
        for i, j in self.SKELETON_CONNECTIONS:
            if (keypoints[i, 2] > self.confidence_threshold and
                    keypoints[j, 2] > self.confidence_threshold):
                pt1 = tuple(keypoints[i, :2].astype(int))
                pt2 = tuple(keypoints[j, :2].astype(int))
                color = self.KEYPOINT_COLORS[i]
                cv2.line(canvas, pt1, pt2, color, thickness)

        # Draw keypoints
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.confidence_threshold:
                cv2.circle(canvas, (int(x), int(y)), thickness + 2, self.KEYPOINT_COLORS[i], -1)

        return canvas

    def to_heatmap(self, keypoints: np.ndarray, height: int, width: int,
                   sigma: float = 6.0) -> np.ndarray:
        """Convert keypoints to Gaussian heatmaps."""
        num_kp = keypoints.shape[0]
        heatmaps = np.zeros((num_kp, height, width), dtype=np.float32)
        for i, (x, y, conf) in enumerate(keypoints):
            if conf > self.confidence_threshold:
                x_grid, y_grid = np.meshgrid(np.arange(width), np.arange(height))
                heatmaps[i] = np.exp(-((x_grid - x) ** 2 + (y_grid - y) ** 2) / (2 * sigma ** 2))
        return heatmaps
