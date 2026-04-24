from __future__ import annotations

import re
import time
from dataclasses import dataclass
from pathlib import Path
import hashlib

import cv2
import numpy as np
import torch
from basicsr.archs.rrdbnet_arch import RRDBNet
from realesrgan import RealESRGANer


@dataclass
class UpscaleResult:
    image: np.ndarray
    model_used: str
    elapsed_seconds: float
    route_note: str


def _anime_score(image_rgb: np.ndarray) -> float:
    img = image_rgb.astype(np.float32) / 255.0
    small = cv2.resize(img, (0, 0), fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)

    gray = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_ratio = float(np.count_nonzero(edges)) / edges.size

    hsv = cv2.cvtColor((small * 255).astype(np.uint8), cv2.COLOR_RGB2HSV)
    sat = hsv[:, :, 1].astype(np.float32) / 255.0
    val = hsv[:, :, 2].astype(np.float32) / 255.0
    vivid_ratio = float(np.mean((sat > 0.35) & (val > 0.45)))

    blur = cv2.GaussianBlur(gray, (7, 7), 0)
    flat_regions = float(np.mean(np.abs(gray.astype(np.float32) - blur.astype(np.float32)) < 4.0))

    score = 0.45 * flat_regions + 0.35 * vivid_ratio - 0.20 * edge_ratio
    return float(np.clip(score, 0.0, 1.0))


def _unsharp(image_rgb: np.ndarray, amount: float) -> np.ndarray:
    if amount <= 0:
        return image_rgb
    sigma = 1.0 + amount * 1.5
    blur = cv2.GaussianBlur(image_rgb, (0, 0), sigmaX=sigma, sigmaY=sigma)
    sharpened = cv2.addWeighted(image_rgb, 1.0 + amount, blur, -amount, 0)
    return np.clip(sharpened, 0, 255).astype(np.uint8)


def _quality_boost(image_rgb: np.ndarray, quality: float) -> np.ndarray:
    if quality <= 0:
        return image_rgb

    h, w = image_rgb.shape[:2]
    if h > 2500 or w > 2500:
        return image_rgb

    yuv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2YUV)
    y = yuv[:, :, 0]

    clahe = cv2.createCLAHE(clipLimit=1.5 + quality * 2.0, tileGridSize=(8, 8))
    yuv[:, :, 0] = clahe.apply(y)
    boosted = cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB)

    denoise_strength = 3 + int(quality * 4)
    boosted = cv2.fastNlMeansDenoisingColored(boosted, None, denoise_strength, denoise_strength, 7, 21)
    mix = cv2.addWeighted(image_rgb, 1.0 - quality * 0.35, boosted, quality * 0.35, 0)
    return np.clip(mix, 0, 255).astype(np.uint8)


def _remap_old_rrdb_keys(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    mapped: dict[str, torch.Tensor] = {}
    for key, value in state.items():
        if key.startswith("model.0."):
            mapped[key.replace("model.0", "conv_first", 1)] = value
            continue
        if key.startswith("model.1.sub."):
            m = re.match(r"model\.1\.sub\.(\d+)\.RDB(\d)\.conv(\d)\.0\.(weight|bias)", key)
            if m:
                block_idx = int(m.group(1))
                rdb_idx = int(m.group(2))
                conv_idx = int(m.group(3))
                param = m.group(4)
                mapped[f"body.{block_idx}.rdb{rdb_idx}.conv{conv_idx}.{param}"] = value
                continue
            if key.startswith("model.1.sub.23."):
                mapped[key.replace("model.1.sub.23", "conv_body", 1)] = value
                continue
        if key.startswith("model.3."):
            mapped[key.replace("model.3", "conv_up1", 1)] = value
            continue
        if key.startswith("model.6."):
            mapped[key.replace("model.6", "conv_up2", 1)] = value
            continue
        if key.startswith("model.8."):
            mapped[key.replace("model.8", "conv_hr", 1)] = value
            continue
        if key.startswith("model.10."):
            mapped[key.replace("model.10", "conv_last", 1)] = value
            continue
        mapped[key] = value
    return mapped


class FusionUpscaler:
    def __init__(self, model_dir: str | Path):
        model_dir = Path(model_dir)
        general_model_path = model_dir / "4x-UltraSharp.pth"
        anime_model_path = model_dir / "realesrganX4plusAnime_v1.pt"

        if not general_model_path.exists() or not anime_model_path.exists():
            raise FileNotFoundError(
                "Missing model files. Expected 4x-UltraSharp.pth and realesrganX4plusAnime_v1.pt"
            )

        general_model_path = self._normalize_checkpoint(general_model_path)
        anime_model_path = self._normalize_checkpoint(anime_model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.half = self.device.type == "cuda"
        tile = 0 if self.device.type == "cuda" else 256

        self.general_upsampler = RealESRGANer(
            scale=4,
            model_path=str(general_model_path),
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4),
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=self.half,
            device=self.device,
        )

        self.anime_upsampler = RealESRGANer(
            scale=4,
            model_path=str(anime_model_path),
            model=RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=6, num_grow_ch=32, scale=4),
            tile=tile,
            tile_pad=10,
            pre_pad=0,
            half=self.half,
            device=self.device,
        )

    @staticmethod
    def _normalize_checkpoint(model_path: Path) -> Path:
        raw = torch.load(str(model_path), map_location="cpu")

        if isinstance(raw, dict):
            if "params" in raw or "params_ema" in raw:
                return model_path
            if all(isinstance(v, torch.Tensor) for v in raw.values()):
                if any(k.startswith("model.0.") for k in raw):
                    wrapped = {"params": _remap_old_rrdb_keys(raw)}
                else:
                    wrapped = {"params": raw}
            elif "state_dict" in raw and isinstance(raw["state_dict"], dict):
                wrapped = {"params": raw["state_dict"]}
            elif "model" in raw and isinstance(raw["model"], dict):
                wrapped = {"params": raw["model"]}
            else:
                return model_path

            stat = model_path.stat()
            sig = f"{model_path}:{stat.st_size}:{stat.st_mtime_ns}:v2"
            digest = hashlib.sha1(sig.encode("utf-8")).hexdigest()[:10]
            cache_dir = model_path.parent / ".normalized"
            cache_dir.mkdir(exist_ok=True)
            out_path = cache_dir / f"{model_path.stem}.{digest}.pth"
            if not out_path.exists():
                torch.save(wrapped, str(out_path))
            return out_path

        return model_path

    @staticmethod
    def _extract_scale(path_or_name: str) -> int:
        m = re.search(r"(\d+)x", path_or_name.lower())
        return int(m.group(1)) if m else 4

    def _upscale_single(self, image_rgb: np.ndarray, route: str) -> tuple[np.ndarray, str]:
        input_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        if route == "anime":
            out_bgr, _ = self.anime_upsampler.enhance(input_bgr, outscale=4)
            return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), "Anime model selected"

        out_bgr, _ = self.general_upsampler.enhance(input_bgr, outscale=4)
        return cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB), "General model selected"

    def _upscale_fusion(self, image_rgb: np.ndarray) -> tuple[np.ndarray, str]:
        input_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
        general_bgr, _ = self.general_upsampler.enhance(input_bgr, outscale=4)
        anime_bgr, _ = self.anime_upsampler.enhance(input_bgr, outscale=4)

        general = cv2.cvtColor(general_bgr, cv2.COLOR_BGR2RGB)
        anime = cv2.cvtColor(anime_bgr, cv2.COLOR_BGR2RGB)

        anime_prob = _anime_score(image_rgb)
        if anime_prob >= 0.7:
            w_anime = 0.7
            note = "Fusion route (anime-biased blend)"
        elif anime_prob <= 0.35:
            w_anime = 0.25
            note = "Fusion route (photo-biased blend)"
        else:
            w_anime = 0.45
            note = "Fusion route (balanced blend)"

        fused = cv2.addWeighted(general, 1.0 - w_anime, anime, w_anime, 0)
        return fused, note

    def upscale(
        self,
        image_rgb: np.ndarray,
        mode: str = "Auto",
        sharpness: float = 0.35,
        quality: float = 0.35,
    ) -> UpscaleResult:
        start = time.perf_counter()

        if mode == "Auto":
            anime_prob = _anime_score(image_rgb)
            route = "anime" if anime_prob > 0.58 else "general"
            upscaled, route_note = self._upscale_single(image_rgb, route)
            route_note = f"{route_note} (auto score={anime_prob:.2f})"
            model_used = "Auto"
        elif mode == "Fusion":
            upscaled, route_note = self._upscale_fusion(image_rgb)
            model_used = "Fusion (custom ensemble)"
        elif mode == "Anime":
            upscaled, route_note = self._upscale_single(image_rgb, "anime")
            model_used = "Anime"
        else:
            upscaled, route_note = self._upscale_single(image_rgb, "general")
            model_used = "Photo"

        enhanced = _quality_boost(upscaled, quality=float(np.clip(quality, 0.0, 1.0)))
        enhanced = _unsharp(enhanced, amount=float(np.clip(sharpness, 0.0, 1.0)))

        elapsed = time.perf_counter() - start
        return UpscaleResult(
            image=enhanced,
            model_used=model_used,
            elapsed_seconds=elapsed,
            route_note=route_note,
        )
