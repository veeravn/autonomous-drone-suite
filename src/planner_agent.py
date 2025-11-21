from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from .vision.semantic_vision import (
    SemanticPerception,
    SemanticFrame,
    CLIPOnnxEmbedder,
    YoloOnnxDetector,
    MeanColorEmbedder,
    NoOpDetector,
)


@dataclass
class NBVDecision:
    """
    Output of PlannerAgent.decide(...).

    heading_delta_deg:
        How much to change yaw this cycle (positive ≈ "turn right").

    novelty_score:
        Scalar [0,1] used by the main loop to gate captures.
        Higher = more novel.

    primary_label:
        Label of the primary subject, or None.

    debug:
        String for on-screen / log debugging.
    """
    heading_delta_deg: float
    novelty_score: float
    primary_label: Optional[str]
    debug: str = ""


class PlannerAgent:
    """
    Phase 2D semantic NBV planner.

    Responsibilities:
      - Compute semantic embedding of each frame.
      - Detect subjects (person/car/structure).
      - Track history of embeddings + poses.
      - Propose heading deltas that:
          * keep interesting subjects centered
          * encourage semantic novelty
          * remain smooth for SITL offboard control
    """

    def __init__(self, semantic_enabled: bool = True) -> None:
        self.semantic_enabled = semantic_enabled

        # Semantic perception stack
        try:
            embedder = CLIPOnnxEmbedder("models/clip_image.onnx")
        except Exception as e:
            print(f"[NBV] CLIP embedder init failed: {e}. Using MeanColorEmbedder.")
            embedder = MeanColorEmbedder()

        try:
            detector = YoloOnnxDetector(
                "models/yolo_nano.onnx",
                class_map={0: "person", 1: "car", 2: "truck", 3: "bus", 4: "building"},
            )
        except Exception as e:
            print(f"[NBV] YOLO detector init failed: {e}. Using NoOpDetector.")
            detector = NoOpDetector()

        # If semantic NBV is disabled, force a simple perception stack:
        if not self.semantic_enabled:
            print("[NBV] Semantic NBV disabled via flag; using simple perception.")
            embedder = MeanColorEmbedder()
            detector = NoOpDetector()

        self.perception = SemanticPerception(embedder=embedder, detector=detector)

        self._history: List[SemanticFrame] = []
        self._yaw_history: List[float] = []
        self.max_history = 100

    # ---------- Public API used by main.py ----------

    def analyze_frame(self, frame_bgr) -> SemanticFrame:
        """
        Run semantic perception on a frame.
        """
        return self.perception.analyze(frame_bgr)

    def update_history(self, sf: SemanticFrame, yaw_deg: float) -> None:
        """
        Append embedding + yaw to history.
        """
        self._history.append(sf)
        self._yaw_history.append(float(yaw_deg))

        if len(self._history) > self.max_history:
            self._history.pop(0)
        if len(self._yaw_history) > self.max_history:
            self._yaw_history.pop(0)

    def decide(self, sf: SemanticFrame, yaw_deg: float, battery_pct: float) -> NBVDecision:
        """
        Main NBV decision function.

        Arguments:
            sf: SemanticFrame (embedding + subjects) for the current frame.
            yaw_deg: current yaw/heading in degrees.
            battery_pct: current battery level (0-100).

        Returns:
            NBVDecision with heading_delta_deg + novelty_score + debug info.
        """
        emb = sf.embedding
        subjects = sf.subjects

        novelty = self._compute_semantic_novelty(emb)
        primary = self._pick_primary_subject(subjects) if self.semantic_enabled else None

        # Base yaw step (deg). This is the "sweep" amount when nothing interesting.
        base_step = 8.0

        # If low battery, be more conservative (smaller sweeps).
        if battery_pct < 30.0:
            base_step = 5.0
        if battery_pct < 15.0:
            base_step = 3.0

        # 1) If we have a primary subject, try to keep it centered.
        #    Use its horizontal center to decide rotation direction.
        heading_delta = 0.0
        dbg_parts = []

        if primary is not None:
            cx, cy = primary.center
            dbg_parts.append(f"subj={primary.label}@({cx:.2f},{cy:.2f}) a={primary.area:.2f}")

            # If subject is off-center horizontally, rotate to re-center.
            # NOTE: If the direction feels flipped in SITL, swap the signs.
            off = cx - 0.5
            if off < -0.05:
                # subject left of center → yaw left (negative delta)
                heading_delta = -base_step
                dbg_parts.append("aim:center_left")
            elif off > 0.05:
                # subject right of center → yaw right (positive delta)
                heading_delta = +base_step
                dbg_parts.append("aim:center_right")
            else:
                # already fairly centered: small sweep for semantic variety
                heading_delta = base_step * (1.0 if novelty > 0.5 else 0.5)
                dbg_parts.append("aim:center_sweep")
        else:
            # 2) No detected subject: do a gentle sweep to explore.
            heading_delta = base_step
            dbg_parts.append("no_subject_sweep")

        # Clamp the delta for smoothness
        heading_delta = float(max(-20.0, min(20.0, heading_delta)))

        dbg_parts.append(f"nov={novelty:.2f}")

        return NBVDecision(
            heading_delta_deg=heading_delta,
            novelty_score=float(novelty),
            primary_label=(primary.label if primary is not None else None),
            debug=";".join(dbg_parts),
        )

    # ---------- Internals ----------

    def _compute_semantic_novelty(self, emb: np.ndarray) -> float:
        """
        Compute semantic novelty of current embedding vs history.

        Returns:
            novelty in [0,1], where 1 ~ very novel, 0 ~ identical to past.
        """
        if emb is None or emb.size == 0:
            return 0.0
        if not self._history:
            return 1.0

        # Cosine distance to most similar past embedding
        e = emb.astype(np.float32).reshape(-1)
        e_norm = np.linalg.norm(e)
        if e_norm == 0:
            return 0.0
        e = e / e_norm

        max_sim = -1.0
        for past_sf in self._history[-30:]:  # look at last 30 shots
            p = past_sf.embedding.astype(np.float32).reshape(-1)
            p_norm = np.linalg.norm(p)
            if p_norm == 0:
                continue
            p = p / p_norm
            sim = float(np.dot(e, p))
            if sim > max_sim:
                max_sim = sim

        if max_sim < 0:
            return 1.0

        dist = 1.0 - max_sim  # cosine distance
        # map [0,2] -> [0,1], clamp
        novelty = max(0.0, min(1.0, dist / 2.0))
        return novelty

    def _pick_primary_subject(self, subjects: List) -> Optional:
        """
        Choose the most photogenically interesting subject.

        Priority:
          1. person
          2. car
          3. structure
          4. other with large area
        """
        if not subjects:
            return None

        # Split by label
        persons = [s for s in subjects if s.label == "person"]
        cars = [s for s in subjects if s.label == "car"]
        structs = [s for s in subjects if s.label == "structure"]
        others = [s for s in subjects if s.label not in ("person", "car", "structure")]

        def pick_largest(ls):
            if not ls:
                return None
            return max(ls, key=lambda s: s.area * s.score)

        for group in (persons, cars, structs, others):
            cand = pick_largest(group)
            if cand is not None and cand.area > 0.01:
                return cand

        return None
