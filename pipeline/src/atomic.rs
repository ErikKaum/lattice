//! Atomic file writes via tmp + rename. Critical when the destination is a
//! FUSE-mounted bucket: a process crash or signal mid-`std::fs::write` can
//! leave a partial file on the bucket — for JSON metadata that's a hard
//! parse error on the next read, so the source's recovery story breaks.
//!
//! POSIX `rename(2)` within the same directory is atomic on every
//! filesystem we care about (ext4, APFS, HF bucket FUSE). After
//! `atomic_write(path, bytes)`, readers see *either* the pre-call content
//! at `path` *or* the new content — never a truncated/half-written file.

use std::io::Write;
use std::path::Path;

use anyhow::{Context, Result};

pub fn atomic_write(path: &Path, bytes: &[u8]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("create_dir_all {}", parent.display()))?;
    }
    // Unique-enough temp name in the same dir so the rename is on the same
    // filesystem (and therefore atomic). Including `nanos + pid` is cheap
    // and dodges races between concurrent writers of the same path.
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos())
        .unwrap_or(0);
    let pid = std::process::id();
    let tmp_name = format!(
        "{}.tmp.{nanos}.{pid}",
        path.file_name()
            .and_then(|s| s.to_str())
            .unwrap_or("atomic_write")
    );
    let tmp_path = path.with_file_name(tmp_name);

    let mut f = std::fs::File::create(&tmp_path)
        .with_context(|| format!("create {}", tmp_path.display()))?;
    f.write_all(bytes)
        .with_context(|| format!("write {}", tmp_path.display()))?;
    // fsync before rename: ensures bytes hit storage before the rename
    // makes them visible. Costs ~ms; avoids "post-rename, pre-flush"
    // crash leaving a zero-byte file at `path` on power loss / OOM-kill.
    f.sync_all().ok(); // best-effort — FUSE may not support fsync.
    drop(f);

    std::fs::rename(&tmp_path, path)
        .with_context(|| format!("rename {} -> {}", tmp_path.display(), path.display()))?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn round_trip() {
        let dir = std::env::temp_dir().join(format!(
            "lattice-atomic-test-{}",
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("foo.json");
        atomic_write(&p, b"{\"a\": 1}").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"{\"a\": 1}");
        // Overwrite — readers see the new content as a single atomic flip.
        atomic_write(&p, b"{\"a\": 2}").unwrap();
        assert_eq!(std::fs::read(&p).unwrap(), b"{\"a\": 2}");
        // No leftover tmp files.
        let leftover: Vec<_> = std::fs::read_dir(&dir)
            .unwrap()
            .filter_map(|e| e.ok())
            .map(|e| e.file_name().to_string_lossy().to_string())
            .filter(|n| n.contains(".tmp."))
            .collect();
        assert!(leftover.is_empty(), "unexpected leftovers: {leftover:?}");
        std::fs::remove_dir_all(&dir).ok();
    }
}
