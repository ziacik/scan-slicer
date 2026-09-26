# Slicer

A small desktop app for splitting a flatbed scan containing several physical photos into separate image files.

The app is intentionally conservative: automatic detection gives every detected photo a configurable safety margin, and every frame can be corrected manually before export.

## Current features

- Open PNG, JPEG and TIFF scans
- Scan directly from SANE-compatible scanners through the system `scanimage` frontend
- Detect multiple photos against the scanner background
- Configurable detection threshold and safety margin
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

## Workflow

1. Open an existing scanner image, or click **Scan** to acquire one directly.
2. Adjust **threshold** if detection misses a photo or picks up scanner noise.
3. Adjust **margin px** to keep a little space around every photo.
4. Click **Detect photos** after changing detection settings.
5. Drag frames or resize them from the yellow corner handles.
6. Add/remove frames where automatic detection got it wrong.
7. Export PNGs.

## Why conservative cropping?

For archival scans, losing a few pixels from an original photograph is worse than keeping a narrow strip of scanner background. Slicer therefore prefers a safe margin over a tight automatic crop.
