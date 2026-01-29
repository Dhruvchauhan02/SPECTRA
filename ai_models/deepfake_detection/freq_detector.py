import cv2
import numpy as np


class FrequencyDetector:
    def predict_proba(self, img_bgr):
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)

        f = np.fft.fft2(gray)
        fshift = np.fft.fftshift(f)
        magnitude = np.log(np.abs(fshift) + 1)

        h, w = magnitude.shape
        center = (h // 2, w // 2)
        radius = min(h, w) // 8

        mask = np.ones((h, w), np.uint8)
        cv2.circle(mask, center, radius, 0, -1)

        high_freq = magnitude * mask

        score = np.mean(high_freq)

        # Normalize to probability-like range
        p_fake = min(1.0, max(0.0, (score - 2.0) / 4.0))

        return float(p_fake)
