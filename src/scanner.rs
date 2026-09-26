use std::{
    fs,
    io,
    path::PathBuf,
    process::Command,
    time::{SystemTime, UNIX_EPOCH},
};

use image::DynamicImage;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct ScannerDevice {
    pub(crate) id: String,
    pub(crate) label: String,
}

pub(crate) fn list_devices() -> Result<Vec<ScannerDevice>, String> {
    let output = Command::new("scanimage")
        .arg("-L")
        .output()
        .map_err(command_error)?;

    if !output.status.success() {
        return Err(command_failure("Could not list scanners", &output.stderr));
    }

    let stdout = String::from_utf8_lossy(&output.stdout);
    Ok(stdout.lines().filter_map(parse_device_line).collect())
}

pub(crate) fn scan(device_id: &str, resolution: u32) -> Result<DynamicImage, String> {
    let path = temporary_scan_path();

    let output = Command::new("scanimage")
        .arg("-d")
        .arg(device_id)
        .arg("--format=png")
        .arg("--mode")
        .arg("Color")
        .arg("--resolution")
        .arg(resolution.to_string())
        .arg("-o")
        .arg(&path)
        .output()
        .map_err(command_error)?;

    if !output.status.success() {
        let _ = fs::remove_file(&path);
        return Err(command_failure("Scanner returned an error", &output.stderr));
    }

    let image = image::open(&path)
        .map_err(|error| format!("Could not read the scanned image: {error}"));
    let _ = fs::remove_file(&path);
    image
}

fn parse_device_line(line: &str) -> Option<ScannerDevice> {
    let line = line.trim();
    let (rest, closing) = if let Some(rest) = line.strip_prefix("device `") {
        (rest, '\'')
    } else if let Some(rest) = line.strip_prefix("device '") {
        (rest, '\'')
    } else {
        return None;
    };

    let end = rest.find(closing)?;
    let id = rest[..end].trim();
    if id.is_empty() {
        return None;
    }

    let description = rest[end + closing.len_utf8()..]
        .trim()
        .strip_prefix("is a ")
        .unwrap_or("")
        .trim();

    let label = if description.is_empty() {
        id.to_owned()
    } else {
        description.to_owned()
    };

    Some(ScannerDevice {
        id: id.to_owned(),
        label,
    })
}

fn temporary_scan_path() -> PathBuf {
    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    std::env::temp_dir().join(format!(
        "scan-slicer-{}-{nanos}.png",
        std::process::id()
    ))
}

fn command_error(error: io::Error) -> String {
    if error.kind() == io::ErrorKind::NotFound {
        "SANE frontend 'scanimage' was not found. Install the 'sane' package.".into()
    } else {
        format!("Could not start scanimage: {error}")
    }
}

fn command_failure(prefix: &str, stderr: &[u8]) -> String {
    let details = String::from_utf8_lossy(stderr).trim().to_owned();
    if details.is_empty() {
        prefix.to_owned()
    } else {
        format!("{prefix}: {details}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_standard_sane_device_line() {
        let device = parse_device_line(
            "device `airscan:e0:Example Scanner' is a WSD Example Scanner ip=192.168.1.10",
        )
        .unwrap();

        assert_eq!(device.id, "airscan:e0:Example Scanner");
        assert_eq!(
            device.label,
            "WSD Example Scanner ip=192.168.1.10"
        );
    }

    #[test]
    fn ignores_unrelated_output() {
        assert!(parse_device_line("No scanners were identified.").is_none());
    }
}
