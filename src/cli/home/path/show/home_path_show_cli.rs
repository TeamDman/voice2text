use crate::cli::ToArgs;
use arbitrary::Arbitrary;
use clap::Args;
use eyre::Result;

/// Show the home path.
#[derive(Args, Debug, Clone, Arbitrary, PartialEq)]
pub struct HomePathShowArgs;

impl HomePathShowArgs {
    /// # Errors
    ///
    /// This function does not return any errors.
    #[expect(clippy::unused_async)]
    pub async fn invoke(self) -> Result<()> {
        println!("{}", crate::paths::APP_HOME.display());
        Ok(())
    }
}

impl ToArgs for HomePathShowArgs {
    fn to_args(&self) -> Vec<std::ffi::OsString> {
        Vec::new()
    }
}
