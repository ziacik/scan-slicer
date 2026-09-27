# Slicer

A small desktop app for splitting a flatbed scan containing several physical photos into separate image files.

The app is intentionally conservative: automatic detection gives every detected photo a configurable safety margin, and every frame can be corrected manually before export.

## Current features

- Open PNG, JPEG and TIFF scans
- Scan directly from SANE-compatible scanners through the system `scanimage` frontend
- Detect multiple photos with OpenAI vision
- Store the user's OpenAI API key securely in the system keyring
- Configurable safety margin
- Move detected frames with the mouse
- Resize frames from their corners
- Add or remove frames manually
- Export all frames as individual PNG files
- Keeps crop pixels at the original scan resolution

## Run

```bash
cargo run --release
```

The app uses GTK4/libadwaita. Scanner acquisition uses the system SANE stack via `scanimage`.

On Arch Linux, install scanner support with:

```bash
sudo pacman -S sane
```

Scanning is optional; opening existing PNG/JPEG/TIFF files works without SANE.

Photo detection requires an OpenAI API key. Use the key button in the header to enter one, or simply run detection: if no key is saved, Scan Slicer opens the setup dialog automatically. The dialog links directly to the OpenAI API key page, and the key is stored in the system keyring rather than an environment variable or the application binary.

## Workflow

1. Open an existing scanner image, or click **Scan** to acquire one directly.
2. On first use, paste an OpenAI API key when prompted. Use **Create an API key…** in the dialog if needed.
3. Adjust **Padding** if you want extra pixels around detected photos.
4. Click **Detect Photos** after changing detection settings.
5. Drag frames or resize them from the corner handles.
6. Add/remove frames where automatic detection got it wrong.
7. Export PNGs.

## Why conservative cropping?

For archival scans, losing a few pixels from an original photograph is worse than keeping a narrow strip of scanner background. Slicer therefore prefers a safe margin over a tight automatic crop.
