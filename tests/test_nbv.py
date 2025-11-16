from __future__ import annotations
import numpy as np
import cv2
from pathlib import Path


from src.planner_agent import PlannerAgent


# Smoke test: ensure we can embed a frame and get a decision

def main():
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.putText(img, "NBV Test", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 3)


    planner = PlannerAgent()
    emb = planner.embed(img)
    dec = planner.decide(emb, curr_heading_deg=0.0, battery=0.9)

    print({
        "heading_delta_deg": dec.heading_delta_deg,
        "novelty": dec.novelty_score,
        "cost": dec.cost_score,
        "emb_dim": int(emb.shape[0])
    })


if __name__ == "__main__":
    main()