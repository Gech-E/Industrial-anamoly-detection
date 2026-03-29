"""Check dataset integrity — find broken images."""
import os
from PIL import Image

root = "Data/mvtec_ad"
broken = []
total = 0

for cat in sorted(os.listdir(root)):
    cat_path = os.path.join(root, cat)
    if not os.path.isdir(cat_path):
        continue
    for split in ["train", "test"]:
        split_path = os.path.join(cat_path, split)
        if not os.path.exists(split_path):
            continue
        for defect in sorted(os.listdir(split_path)):
            defect_path = os.path.join(split_path, defect)
            if not os.path.isdir(defect_path):
                continue
            for fname in sorted(os.listdir(defect_path)):
                if fname.lower().endswith((".png", ".jpg", ".jpeg", ".bmp")):
                    total += 1
                    fpath = os.path.join(defect_path, fname)
                    try:
                        img = Image.open(fpath)
                        img.verify()  # Verify it's not broken
                    except Exception as e:
                        broken.append((fpath, str(e)))
                        print(f"BROKEN: {fpath} -> {e}")

print(f"\nTotal images: {total}")
print(f"Broken images: {len(broken)}")
if not broken:
    print("All images OK!")
