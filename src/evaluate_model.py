from pathlib import Path
import csv

import matplotlib.pyplot as plt
import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio, structural_similarity
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity

from src.data.dataset import SuperResolutionDataset
from src.models.train_unet import UNetModel


LR_SIZE = 32
CHECKPOINT = f"checkpoints/sr_{LR_SIZE}_gpu/best.ckpt"
RESULTS_DIR = Path("results")


def calculate_metrics(target, prediction, lpips):
    original = target.squeeze(0).permute(1, 2, 0).cpu().numpy()
    restored = prediction.squeeze(0).permute(1, 2, 0).cpu().numpy()

    sne = ((target.double() - prediction.double()) ** 2).sum().item()
    psnr = peak_signal_noise_ratio(original, restored, data_range=1)
    ssim = structural_similarity(original, restored, channel_axis=2, data_range=1)
    perceptual = lpips(target, prediction).item()
    lpips.reset()

    return [sne, psnr, ssim, perceptual]


def save_comparison(target, bicubic, prediction):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    images = [target, bicubic, prediction]
    titles = ["Original", "Bicubic", "UNet"]
    for i in range(3):
        image = images[i].squeeze(0).permute(1, 2, 0).cpu().numpy()
        axes[i].imshow(image)
        axes[i].set_title(titles[i])
        axes[i].axis("off")

    plt.subplots_adjust(top=0.85)
    plt.savefig(RESULTS_DIR / f"comparison_{LR_SIZE}.png", dpi=150,
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

    dataset = SuperResolutionDataset("data/processed", "val", LR_SIZE)
    results = {"Bicubic": [], "UNet": []}
    RESULTS_DIR.mkdir(exist_ok=True)

    with torch.no_grad():
        for i in range(len(dataset)):
            pair = dataset[i]
            bicubic = pair["input"].unsqueeze(0).to(device)
            target = pair["target"].unsqueeze(0).to(device)
            prediction = model(bicubic).clamp(0, 1)

            results["Bicubic"].append(calculate_metrics(target, bicubic, lpips))
            results["UNet"].append(calculate_metrics(target, prediction, lpips))

            if i == 0:
                save_comparison(target, bicubic, prediction)

    with (RESULTS_DIR / f"metrics_{LR_SIZE}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["method", "lr_size", "images", "sne", "psnr", "ssim", "lpips"])
        for method in results:
            sne, psnr, ssim, perceptual = np.mean(results[method], axis=0)
            writer.writerow([method, LR_SIZE, len(dataset), sne, psnr, ssim, perceptual])
            print(f"{method}: SNE {sne:.3f}, PSNR {psnr:.3f}, "
                  f"SSIM {ssim:.4f}, LPIPS {perceptual:.4f}")


if __name__ == "__main__":
    main()
