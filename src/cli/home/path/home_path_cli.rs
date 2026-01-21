use crate::cli::ToArgs;
use crate::cli::home::path::show::HomePathShowArgs;
use arbitrary::Arbitrary;
use clap::Args;
use clap::Subcommand;
use eyre::Result;

/// Home path commands.
#[derive(Args, Debug, Clone, Arbitrary, PartialEq)]
pub struct HomePathArgs {
    #[command(subcommand)]
    pub command: HomePathCommand,
}

#[derive(Subcommand, Debug, Clone, Arbitrary, PartialEq)]
pub enum HomePathCommand {
    Show(HomePathShowArgs),
}

impl HomePathArgs {
    /// # Errors
    ///
    /// This function will return an error if the subcommand fails.
    pub async fn invoke(self) -> Result<()> {
        match self.command {
            HomePathCommand::Show(args) => args.invoke().await?,
        }

        Ok(())
    }
}

impl ToArgs for HomePathArgs {
    fn to_args(&self) -> Vec<std::ffi::OsString> {
        let mut args = Vec::new();
        match &self.command {
            HomePathCommand::Show(show_args) => {
                args.push("show".into());
                args.extend(show_args.to_args());
            }
        }
        args
    }
}
