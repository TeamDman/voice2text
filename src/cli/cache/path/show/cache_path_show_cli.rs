use crate::cli::ToArgs;
use arbitrary::Arbitrary;
use clap::Args;
use eyre::Result;

/// Show the cache path.
#[derive(Args, Debug, Clone, Arbitrary, PartialEq)]
pub struct CachePathShowArgs;

impl CachePathShowArgs {
    /// # Errors
    ///
    /// This function does not return any errors.
    #[expect(clippy::unused_async)]
    pub async fn invoke(self) -> Result<()> {
        println!("{}", crate::paths::CACHE_DIR.display());
        Ok(())
    }
}

impl ToArgs for CachePathShowArgs {
    fn to_args(&self) -> Vec<std::ffi::OsString> {
        Vec::new()
    }
}
