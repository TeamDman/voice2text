use directories_next::ProjectDirs;
use eyre::bail;
use std::path::Path;
use std::path::PathBuf;
use std::sync::LazyLock;
use tracing::warn;

/// The cache home directory for API responses.
pub static CACHE_DIR: LazyLock<CacheHome> = LazyLock::new(|| match CacheHome::resolve() {
    Ok(c) => c,
    Err(e) => {
        warn!("Failed to resolve cache home: {}", e);
        CacheHome(std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")))
    }
});

/// Helper that resolves the application cache directory.
#[derive(Clone, Debug)]
pub struct CacheHome(pub PathBuf);

impl CacheHome {
    /// Resolve the `CacheHome` according to:
    /// * If [`super::APP_CACHE_ENV_VAR`] env var is set, use that directory
    /// * Otherwise use the platform `ProjectDirs::cache_dir()`
    ///
    /// # Errors
    ///
    /// This function will return an error if the cache directory cannot be determined.
    pub fn resolve() -> eyre::Result<CacheHome> {
        if let Ok(override_dir) = std::env::var(super::APP_CACHE_ENV_VAR) {
            return Ok(CacheHome(PathBuf::from(override_dir)));
        }
        if let Some(project_dirs) = ProjectDirs::from("", "teamdman", super::APP_CACHE_DIR_NAME) {
            Ok(CacheHome(project_dirs.cache_dir().to_path_buf()))
        } else {
            bail!("Could not determine cache directory")
        }
    }
}

impl std::ops::Deref for CacheHome {
    type Target = Path;

    fn deref(&self) -> &Self::Target {
        self.0.as_path()
    }
}

/// Clean the entire API response cache directory.
///
/// # Errors
///
/// This function will return an error if removing files or directories fails.
pub fn clean_cache() -> eyre::Result<CleanResult> {
    let cache_dir = CACHE_DIR.0.as_path();
    let mut result = CleanResult::default();

    if !cache_dir.exists() {
        return Ok(result);
    }

    for entry in std::fs::read_dir(cache_dir)? {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            std::fs::remove_dir_all(&path)?;
            result.entries_removed += 1;
        }
    }

    // Remove the cache directory itself if empty
    if std::fs::read_dir(cache_dir)?.next().is_none() {
        std::fs::remove_dir(cache_dir)?;
    }

    Ok(result)
}

/// Result of a cache clean operation.
#[derive(Debug, Default)]
pub struct CleanResult {
    /// Number of cache entries removed.
    pub entries_removed: usize,
}
