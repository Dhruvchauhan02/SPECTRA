import cv2

from ai_models.face_recognition.detector import RetinaFaceDetector
from ai_models.deepfake_detection.efficientnet_detector import EfficientNetDeepfakeDetector
from ai_models.deepfake_detection.clip_detector import CLIPDetector
from ai_models.deepfake_detection.fusion import ScoreFusion


class DeepfakePipeline:
    def __init__(self, device="cpu"):
        self.face_detector = RetinaFaceDetector()

        # 🔥 MAIN VISUAL MODEL (your trained EfficientNet)
        self.visual = EfficientNetDeepfakeDetector("efficientnet_b0_spectra.pth")

        # 🔹 CLIP semantic support
        self.clip = CLIPDetector(device=device)

        # 🔹 Fusion logic
        self.fusion = ScoreFusion()

    def analyze(self, image_path):
        img_bgr = cv2.imread(image_path)
        assert img_bgr is not None, "Image not loaded"

        # ----------------------------
        # 🔹 Face detection
        # ----------------------------
        faces = self.face_detector.detect(img_bgr)

        if faces:
            face = faces[0].crop_img
            if face is not None and face.size > 0:
                img_for_analysis = face
            else:
                img_for_analysis = img_bgr
        else:
            img_for_analysis = img_bgr

        # ----------------------------
        # 🔹 Visual model (EfficientNet)
        # ----------------------------
        p_visual = self.visual.predict_proba(img_for_analysis)

        # ----------------------------
        # 🔹 CLIP support
        # ----------------------------
        p_clip = self.clip.predict_proba(img_for_analysis)

        # ----------------------------
        # 🔹 Frequency disabled for now
        # ----------------------------
        p_freq = 0.0

        # ----------------------------
        # 🔹 Fusion
        # ----------------------------
        final_p, verdict = self.fusion.fuse(
            p_freq=p_freq,
            p_visual=p_visual,
            p_clip=p_clip
        )

        return {
            "p_visual": round(p_visual, 3),
            "p_freq": round(p_freq, 3),
            "p_clip": round(p_clip, 3),
            "final_p": round(final_p, 3),
            "verdict": verdict
        }
