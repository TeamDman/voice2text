use crate::cli::ToArgs;
use arbitrary::Arbitrary;
use clap::Args;
use eyre::Result;

/// Delete the cache files.
#[derive(Args, Debug, Clone, Arbitrary, PartialEq)]
pub struct CacheCleanArgs;

impl CacheCleanArgs {
    /// # Errors
    ///
    /// This function does not return any errors.
    #[expect(clippy::unused_async)]
    pub async fn invoke(self) -> Result<()> {
        Ok(())
    }
}

impl ToArgs for CacheCleanArgs {
    fn to_args(&self) -> Vec<std::ffi::OsString> {
        Vec::new()
    }
}
