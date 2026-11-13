from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.restoration import denoise_bilateral
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data.dataset import DenoisingDataset
from src.evaluate_model import calculate_metrics
from src.models.train_unet import UNetModel


SIGMA = 0.03
CHECKPOINT = f"checkpoints/denoising_{SIGMA}_gpu/best.ckpt"
RESULTS_DIR = Path("results")


def save_comparison(target, noisy, bilateral, prediction):
    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    images = [target, noisy, bilateral, prediction]
    titles = ["Original", f"Noisy (sigma={SIGMA})", "Bilateral", "UNet"]
    for i in range(4):
        image = images[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(image)
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.subplots_adjust(top=0.85)
    plt.savefig(RESULTS_DIR / f"denoising_comparison_{SIGMA}.png", dpi=150,
                bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = UNetModel.load_from_checkpoint(CHECKPOINT, map_location=device)
    model = model.to(device)
    model.eval()

    lpips = LearnedPerceptualImagePatchSimilarity(net_type="vgg", normalize=True)
    lpips = lpips.to(device)
    lpips.eval()

    dataset = DenoisingDataset("data/processed", "val", SIGMA)
    results = {"Noisy": [], "Bilateral": [], "UNet": []}
    RESULTS_DIR.mkdir(exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            pair = dataset[i]
            noisy = pair["input"].unsqueeze(0).to(device)
            target = pair["target"].unsqueeze(0).to(device)
            prediction = model(noisy).clamp(0, 1)

            image = pair["input"].permute(1, 2, 0).numpy()
            filtered = denoise_bilateral(image, sigma_color=SIGMA,
                                         sigma_spatial=1, mode="reflect", channel_axis=-1)
            bilateral = torch.from_numpy(filtered.transpose(2, 0, 1).copy())
            bilateral = bilateral.unsqueeze(0).to(device).clamp(0, 1)

            results["Noisy"].append(calculate_metrics(target, noisy, lpips))
            results["Bilateral"].append(calculate_metrics(target, bilateral, lpips))
            results["UNet"].append(calculate_metrics(target, prediction, lpips))

            if i == 0:
                save_comparison(target, noisy, bilateral, prediction)

    with (RESULTS_DIR / f"denoising_metrics_{SIGMA}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "sigma", "images", "sne", "psnr", "ssim", "lpips"])
        for method in results:
            sne, psnr, ssim, perceptual = np.mean(results[method], axis=0)
            writer.writerow([method, SIGMA, len(dataset), sne, psnr, ssim, perceptual])
            print(f"{method}: SNE {sne:.3f}, PSNR {psnr:.3f}, "
                  f"SSIM {ssim:.4f}, LPIPS {perceptual:.4f}")


if __name__ == "__main__":
    main()
