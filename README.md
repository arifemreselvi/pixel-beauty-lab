# Fusion Upscaler

Fusion Upscaler is a Windows desktop application for fast, local 4x image upscaling with automatic model routing, model fusion, and post-enhancement for clarity.
It is designed for users who want a clean native UI (not a browser app), one-click EXE startup, and high-quality results across both photographic and anime/illustration images.

## Developer

- **Name:** Arif Emre Selvi
- **Academic Status:** 2nd year MIS (Management Information Systems) student

## Highlights

- Native Windows desktop interface built with Tkinter.
- Smart `Auto` routing between specialist upscaler checkpoints.
- `Fusion` mode that blends both model outputs for hybrid detail.
- Post-processing pipeline for quality boost and sharpening.
- Local-only processing (no cloud uploads, no external API calls).
- EXE launcher support for easy daily use.

## Model Setup

Place the following checkpoints in `models/`:

- `models/4x-UltraSharp.pth` (general photo-oriented)
- `models/realesrganX4plusAnime_v1.pt` (anime/illustration-oriented)

The app expects these names by default.

## How It Works

1. Input image is loaded in RGB.
2. Content characteristics are estimated (anime/photo style cues).
3. Selected mode decides routing:
   - `Auto`: picks one model from content score
   - `Photo`: forces `4x-UltraSharp.pth`
   - `Anime`: forces `realesrganX4plusAnime_v1.pt`
   - `Fusion`: runs both and blends adaptively
4. Output is enhanced with:
   - adaptive quality/contrast refinement
   - configurable unsharp sharpening
5. Final upscaled result is shown and can be saved.

## Tech Stack

- Python 3.10 (recommended)
- PyTorch + RealESRGAN ecosystem
- OpenCV + Pillow
- Tkinter (native desktop UI)

## Project Structure

- `Fusion Upscaler.exe` - desktop executable entrypoint (main-folder launch)
- `_internal/` - runtime files required by the executable
- `desktop_app.py` - main native desktop application
- `fusion_upscaler.py` - core routing, blending, and enhancement pipeline
- `launcher.py` - launcher source used when building the EXE
- `analyze_models.py` - utility for checkpoint inspection
- `generate_icon.py` - icon asset generator
- `assets/` - icon files
- `models/` - local model checkpoints (user provided)

## Installation (Source)

Use Python 3.10 for best compatibility.

```bash
py -3.10 -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Run (Source)

```bash
py -3.10 desktop_app.py
```

## EXE Usage

Launch:

- `./Fusion Upscaler.exe`

Notes:

- Keep `Fusion Upscaler.exe` and `_internal/` in the project root together.
- First launch can be slower; subsequent launches are faster due to interpreter caching.

## Performance Notes

- GPU acceleration is used automatically when CUDA is available.
- CPU fallback is supported.
- Heavy model initialization is lazy-loaded to reduce window startup delay.

## Troubleshooting

If app startup fails, check logs:

- `%TEMP%\fusion_upscaler_launcher.log`
- `%TEMP%\fusion_upscaler_app.log`

Common fixes:

- Ensure Python 3.10 is installed.
- Reinstall dependencies from `requirements.txt`.
- Confirm model files are present in `models/` with exact expected filenames.

## Model Analysis Utility

To inspect checkpoint metadata:

```bash
py -3.10 analyze_models.py models
```

## Privacy

All image processing runs locally on your machine.

## License

This project is licensed under the MIT License. See `LICENSE`.
