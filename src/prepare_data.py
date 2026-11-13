from pathlib import Path
import random

import cv2


RAW_DIR = Path("data/raw")
OUTPUT_DIR = Path("data/processed")
PATCH_SIZE = 256
LR_SIZES = (32, 64)


def save_image(path, image):
    cv2.imwrite(str(path), image)


def process_split(source, split, random_crop):
    images = sorted(source.glob("*.png"))

    output = OUTPUT_DIR / split
    for folder in ("HR", "LR_32", "LR_64"):
        (output / folder).mkdir(parents=True, exist_ok=True)

    for path in images:
        image = cv2.imread(str(path))
        height, width = image.shape[:2]

        if random_crop:
            x = random.randint(0, width - PATCH_SIZE)
            y = random.randint(0, height - PATCH_SIZE)
        else:
            x = (width - PATCH_SIZE) // 2
            y = (height - PATCH_SIZE) // 2

        hr = image[y:y + PATCH_SIZE, x:x + PATCH_SIZE]
        save_image(output / "HR" / path.name, hr)

        for size in LR_SIZES:
            lr = cv2.resize(hr, (size, size), interpolation=cv2.INTER_CUBIC)
            save_image(output / f"LR_{size}" / path.name, lr)

    print(f"{split}: {len(images)} images prepared")


def main():
    random.seed(42)
    process_split(RAW_DIR / "DIV2K_train_HR", "train", random_crop=True)
    process_split(RAW_DIR / "DIV2K_valid_HR", "val", random_crop=False)


if __name__ == "__main__":
    main()
