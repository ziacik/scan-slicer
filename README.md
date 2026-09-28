# Scan Slicer

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

## Install release packages

GitHub Releases provide three Linux package formats:

- **Debian/Ubuntu:** `.deb`
- **Arch Linux:** `.pkg.tar.zst`
- **Portable:** `.AppImage`

Examples:

```bash
sudo apt install ./scan-slicer_0.1.0_amd64.deb
sudo pacman -U ./scan-slicer-0.1.0-1-x86_64.pkg.tar.zst
chmod +x scan-slicer_0.1.0_x86_64.AppImage
./scan-slicer_0.1.0_x86_64.AppImage
```

The native packages install the desktop launcher, scalable app icon and AppStream metadata, so Scan Slicer appears normally in the desktop application menu.

## Creating a release

Set the version in `Cargo.toml`, commit it, then push a matching `v*` tag:

```bash
git tag v0.1.0
git push origin v0.1.0
```

The Packages workflow validates the desktop metadata and icon, builds all three package formats, creates the GitHub Release, uploads the packages, adds `SHA256SUMS`, and updates the `scan-slicer` package in the AUR.

AUR publishing needs one-time SSH setup:

1. Create a dedicated SSH key and add its public key to your AUR account.
2. Add the private key to this GitHub repository as an Actions secret named `AUR_SSH_PRIVATE_KEY`.

After that, every matching release tag updates AUR automatically; there is no separate AUR release step.

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
