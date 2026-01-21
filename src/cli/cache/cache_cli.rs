use crate::cli::ToArgs;
use crate::cli::cache::clean::CacheCleanArgs;
use crate::cli::cache::path::CachePathArgs;
use arbitrary::Arbitrary;
use clap::Args;
use clap::Subcommand;
use eyre::Result;

/// Cache-related commands.
#[derive(Args, Debug, Clone, Arbitrary, PartialEq)]
pub struct CacheArgs {
    #[command(subcommand)]
    pub command: CacheCommand,
}

#[derive(Subcommand, Debug, Clone, Arbitrary, PartialEq)]
pub enum CacheCommand {
    Clean(CacheCleanArgs),
    Path(CachePathArgs),
}

impl CacheArgs {
    /// # Errors
    ///
    /// This function will return an error if the subcommand fails.
    pub async fn invoke(self) -> Result<()> {
        match self.command {
            CacheCommand::Clean(args) => args.invoke().await?,
            CacheCommand::Path(args) => args.invoke().await?,
        }

        Ok(())
    }
}

impl ToArgs for CacheArgs {
    fn to_args(&self) -> Vec<std::ffi::OsString> {
        let mut args = Vec::new();
        match &self.command {
            CacheCommand::Clean(clean_args) => {
                args.push("clean".into());
                args.extend(clean_args.to_args());
            }
            CacheCommand::Path(path_args) => {
                args.push("path".into());
                args.extend(path_args.to_args());
            }
        }
        args
    }
}
