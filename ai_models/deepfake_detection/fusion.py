class ScoreFusion:
    def __init__(self):
        # Decision thresholds (calibrated for CPU models)
        self.fake_thresh = 0.65
        self.real_thresh = 0.35

    def fuse(self, p_freq, p_visual, p_clip):
        """
        Forensic decision logic
        Returns: final_probability, verdict
        """

        # 1️⃣ Strong visual CNN confidence → trust it
        if p_visual >= 0.75:
            return p_visual, "FAKE"

        if p_visual <= 0.20 and p_freq < 0.6:
            return p_visual, "REAL"

        # 2️⃣ Frequency is a suspicion signal, not a judge
        suspicious = p_freq >= 0.8

        # 3️⃣ Weighted forensic combination
        final_p = (
            0.5 * p_visual +
            0.3 * p_clip +
            0.2 * p_freq
        )

        # 4️⃣ Verdict with uncertainty zone
        if final_p >= self.fake_thresh:
            verdict = "FAKE"
        elif final_p <= self.real_thresh:
            verdict = "REAL"
        else:
            verdict = "UNCERTAIN"

        return final_p, verdict
