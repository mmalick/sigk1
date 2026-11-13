from pathlib import Path

import cv2
import matplotlib.pyplot as plt


DATA_DIR = Path("data/processed/val")
IMAGE_NAME = "0801.png"


def load_rgb(path):
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():
    hr = load_rgb(DATA_DIR / "HR" / IMAGE_NAME)
    lr32 = load_rgb(DATA_DIR / "LR_32" / IMAGE_NAME)
    lr64 = load_rgb(DATA_DIR / "LR_64" / IMAGE_NAME)

    bicubic32 = cv2.resize(lr32, (256, 256), interpolation=cv2.INTER_CUBIC)
    bicubic64 = cv2.resize(lr64, (256, 256), interpolation=cv2.INTER_CUBIC)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [hr, bicubic32, bicubic64]
    titles = ["Original 256x256", "Bicubic from 32x32", "Bicubic from 64x64"]
    for i in range(3):
        axes[i].imshow(images[i])
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.subplots_adjust(top=0.85)
    Path("results").mkdir(exist_ok=True)
    plt.savefig("results/bicubic_comparison.png", dpi=150, bbox_inches="tight", pad_inches=0.25)
    plt.show()


if __name__ == "__main__":
    main()
