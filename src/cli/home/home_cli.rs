use crate::cli::ToArgs;
use crate::cli::home::path::HomePathArgs;
use arbitrary::Arbitrary;
use clap::Args;
use clap::Subcommand;
use eyre::Result;

/// Home-related commands.
#[derive(Args, Debug, Clone, Arbitrary, PartialEq)]
pub struct HomeArgs {
    #[command(subcommand)]
    pub command: HomeCommand,
}

#[derive(Subcommand, Debug, Clone, Arbitrary, PartialEq)]
pub enum HomeCommand {
    Path(HomePathArgs),
}

impl HomeArgs {
    /// # Errors
    ///
    /// This function will return an error if the subcommand fails.
    pub async fn invoke(self) -> Result<()> {
        match self.command {
            HomeCommand::Path(args) => args.invoke().await?,
        }

        Ok(())
    }
}

impl ToArgs for HomeArgs {
    fn to_args(&self) -> Vec<std::ffi::OsString> {
        let mut args = Vec::new();
        match &self.command {
            HomeCommand::Path(path_args) => {
                args.push("path".into());
                args.extend(path_args.to_args());
            }
        }
        args
    }
}
