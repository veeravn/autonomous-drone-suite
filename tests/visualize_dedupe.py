from __future__ import annotations
import argparse
from pathlib import Path
import shutil

from numpy import unique
import cv2


from src.shot_dedupe import ShotDeduper

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-dir", required=True)
    ap.add_argument("--copy-unique", default=None)
    args = ap.parse_args()


    p = Path(args.img_dir)
    imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in {".jpg", ".png", ".jpeg"}])


    dedupe = ShotDeduper()


    unique = []
    for fp in imgs:
        img = cv2.imread(str(fp))
        if img is None:
            continue
        pos = (0.0, 0.0, 0.0) # if you have GPS, replace here
        if not dedupe.is_duplicate(img, pos):
            dedupe.add(img, pos)
            unique.append(fp)
    print(f"Unique images: {len(unique)} / {len(imgs)}")
    if args.copy_unique:
        out = Path(args.copy_unique); out.mkdir(parents=True, exist_ok=True)
        for fp in unique:
            shutil.copy2(fp, out / fp.name)
        print(f"Copied unique images to {out}")

if __name__ == "__main__":
    main()