from __future__ import annotations
import time
from dataclasses import dataclass
from typing import Optional, List


import cv2
import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as T
from dataclasses import dataclass
import math

from .utils.math import wrap_angle_deg

from .utils.math import wrap_angle_deg


# Simple NBV scoring = novelty(embedding vs history) - cost(turn + energy)
# Replace with your decision graph later.


@dataclass
class NBVDecision:
    heading_delta_deg: float
    novelty_score: float
    cost_score: float


class PlannerAgent:
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(self.device)
        self.model.eval()
        self.tf = T.Compose([
            T.ToTensor(),
            T.Resize((224, 224)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.emb_history: List[np.ndarray] = []
        self.step = 0  # for exploration schedule



    @torch.inference_mode()
    def embed(self, frame_bgr: np.ndarray) -> np.ndarray:
        x = self.tf(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(self.device)
        feat = self.model(x).detach().cpu().numpy()[0]
        # L2-normalize
        n = np.linalg.norm(feat) + 1e-6
        return feat / n


    def score_novelty(self, emb: np.ndarray) -> float:
        if not self.emb_history:
            return 1.0
        sims = [float(np.dot(emb, e)) for e in self.emb_history[-16:]]
        # lower similarity → higher novelty
        return float(1.0 - max(sims))


    def score_cost(self, heading_delta_deg: float, battery: float) -> float:
        turn_cost = abs(heading_delta_deg) / 180.0
        energy_cost = (1.0 - battery)
        return 0.5 * turn_cost + 0.5 * energy_cost


    def decide(self, emb: np.ndarray, curr_heading_deg: float, battery: float) -> NBVDecision:
        self.step += 1
        candidates = np.linspace(-45, 45, 13)  # tighten for smoothness

        novelty = self.score_novelty(emb)

        # --- ε-greedy exploration ---
        # shrink epsilon over time, floor at 0.05
        eps = max(0.05, 0.25 * math.exp(-self.step / 350.0))  # slightly more eager exploration, faster decay
        if novelty < 0.05 or np.random.rand() < eps:
            # gentle oscillation to induce parallax & new content
            # alternate left/right every ~2s assuming ~20 Hz loop
            swing_amp = 15.0                                      # a bit larger motion to change the view
            swing = swing_amp * (1 if (self.step // 40) % 2 == 0 else -1)
            return NBVDecision(heading_delta_deg=wrap_angle_deg(swing), novelty_score=novelty, cost_score=self.score_cost(swing, battery))

        best, best_score, best_cost = 0.0, -1e9, 0.0
        for d in candidates:
            cost = self.score_cost(d, battery)
            score = 2.0 * novelty - 1.0 * cost
            if score > best_score:
                best, best_score, best_cost = d, score, cost

        return NBVDecision(heading_delta_deg=wrap_angle_deg(best), novelty_score=novelty, cost_score=best_cost)

    def update_history(self, emb: np.ndarray):
        self.emb_history.append(emb)
        if len(self.emb_history) > 256:
            self.emb_history = self.emb_history[-256:]