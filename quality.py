"""Synthetic-image quality: FID, KID (unbiased, better for small classes) and nearest-neighbour SSIM."""
import numpy as np
import torch


@torch.no_grad()
def image_quality(real, fake, device):
    from torchmetrics.image.fid import FrechetInceptionDistance
    from torchmetrics.image.kid import KernelInceptionDistance
    from torchmetrics.functional.image import structural_similarity_index_measure as ssim
    to_t = lambda a: torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2)
    subset = int(max(2, min(50, len(real), len(fake))))
    fid = FrechetInceptionDistance(feature=2048, normalize=False).to(device)
    kid = KernelInceptionDistance(feature=2048, subset_size=subset, subsets=50, normalize=False).to(device)
    for arr, is_real in ((real, True), (fake, False)):
        for s in range(0, len(arr), 64):
            batch = to_t(arr[s:s + 64]).to(device)
            fid.update(batch, real=is_real)
            kid.update(batch, real=is_real)
    fid_val = float(fid.compute())
    kid_mean, kid_std = kid.compute()
    rng = np.random.default_rng(0)
    ref = real[rng.choice(len(real), size=min(100, len(real)), replace=False)]
    ref_t = to_t(ref).float().div(255).to(device)
    nn_ssim = []
    for f in fake[:50]:
        ft = to_t(f[None]).float().div(255).to(device).expand(len(ref_t), -1, -1, -1)
        nn_ssim.append(float(ssim(ft, ref_t, data_range=1.0, reduction="none").max()))
    return dict(fid=fid_val, kid=float(kid_mean), kid_std=float(kid_std), nn_ssim=float(np.mean(nn_ssim)))
