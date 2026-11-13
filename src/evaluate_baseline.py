from pathlib import Path
import csv

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


VAL_DIR = Path("data/processed/val")
RESULTS_DIR = Path("results")


def load_rgb(path):
    image = cv2.imread(str(path))
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def evaluate_size(size):
    images = sorted((VAL_DIR / "HR").glob("*.png"))

    psnr_values = []
    ssim_values = []

    for path in images:
        hr = load_rgb(path)
        lr = load_rgb(VAL_DIR / f"LR_{size}" / path.name)
        prediction = cv2.resize(lr, (256, 256), interpolation=cv2.INTER_CUBIC)

        psnr_values.append(peak_signal_noise_ratio(hr, prediction, data_range=255))
        ssim_values.append(structural_similarity(
            hr, prediction, channel_axis=2, data_range=255
        ))

    return {
        "lr_size": size,
        "images_evaluated": len(images),
        "mean_psnr": np.mean(psnr_values),
        "mean_ssim": np.mean(ssim_values),
    }


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for size in (32, 64):
        result = evaluate_size(size)
        results.append(result)
        print(f"LR {size}: PSNR {result['mean_psnr']:.3f}, SSIM {result['mean_ssim']:.4f}")

    with (RESULTS_DIR / "baseline_metrics.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=[
            "lr_size", "images_evaluated", "mean_psnr", "mean_ssim"
        ])
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    main()
