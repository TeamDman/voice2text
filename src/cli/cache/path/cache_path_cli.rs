use crate::cli::ToArgs;
use crate::cli::cache::path::show::CachePathShowArgs;
use arbitrary::Arbitrary;
use clap::Args;
use clap::Subcommand;
use eyre::Result;

/// Cache path commands.
#[derive(Args, Debug, Clone, Arbitrary, PartialEq)]
pub struct CachePathArgs {
    #[command(subcommand)]
    pub command: CachePathCommand,
}

#[derive(Subcommand, Debug, Clone, Arbitrary, PartialEq)]
pub enum CachePathCommand {
    Show(CachePathShowArgs),
}

impl CachePathArgs {
    /// # Errors
    ///
    /// This function will return an error if the subcommand fails.
    pub async fn invoke(self) -> Result<()> {
        match self.command {
            CachePathCommand::Show(args) => args.invoke().await?,
        }

        Ok(())
    }
}

impl ToArgs for CachePathArgs {
    fn to_args(&self) -> Vec<std::ffi::OsString> {
        let mut args = Vec::new();
        match &self.command {
            CachePathCommand::Show(show_args) => {
                args.push("show".into());
                args.extend(show_args.to_args());
            }
        }
        args
    }
}
