import cv2

from ai_models.face_recognition.detector import RetinaFaceDetector
from ai_models.deepfake_detection.vision_hf import VisionHFDetector
from ai_models.deepfake_detection.freq_detector import FrequencyDetector
from ai_models.deepfake_detection.clip_detector import CLIPDetector
from ai_models.deepfake_detection.fusion import ScoreFusion


class DeepfakePipeline:
    def __init__(self, device="cpu"):
        self.face_detector = RetinaFaceDetector()

        self.vision = VisionHFDetector(device=device)
        self.freq = FrequencyDetector()
        self.clip = CLIPDetector(device=device)

        self.fusion = ScoreFusion()

    def analyze(self, image_path):
        img_bgr = cv2.imread(image_path)
        assert img_bgr is not None, "Image not loaded"

        # 🔹 Full image analysis
        p_visual = self.vision.predict_proba(img_bgr)
        p_freq = self.freq.predict_proba(img_bgr)
        p_clip = self.clip.predict_proba(img_bgr)

        # 🔹 Face-based refinement (VERY IMPORTANT)
        faces = self.face_detector.detect(img_bgr)
        if faces:
            face = faces[0].crop_img
            if face is not None and face.size > 0:
                p_visual_face = self.vision.predict_proba(face)
                p_clip_face = self.clip.predict_proba(face)

                p_visual = max(p_visual, p_visual_face)
                p_clip = max(p_clip, p_clip_face)
        # 🔹 Forensic fusion
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
