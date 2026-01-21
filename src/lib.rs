#![deny(clippy::disallowed_methods)]
#![deny(clippy::disallowed_macros)]

pub mod cli;
pub mod logging;
pub mod paths;

use crate::cli::Cli;
use clap::CommandFactory;
use clap::FromArgMatches;

// Entrypoint for the program to reduce coupling to the name of this crate.
///
/// # Errors
///
/// This function will return an error if `color_eyre` installation, CLI parsing, logging initialization, or command execution fails.
pub fn main() -> eyre::Result<()> {
    // Install color_eyre for better error reports
    color_eyre::install()?;

    // Parse command line arguments
    let cli = Cli::command();
    let cli = Cli::from_arg_matches(&cli.get_matches())?;

    // Initialize logging
    logging::init_logging(&cli.global_args.logging_config()?)?;

    #[cfg(windows)]
    {
        // Enable ANSI support on Windows
        // This fails in a pipe scenario, so we ignore the error
        let _ = teamy_windows::console::enable_ansi_support();

        // Warn if UTF-8 is not enabled on Windows
        #[cfg(windows)]
        teamy_windows::string::warn_if_utf8_not_enabled();
    };

    // Invoke whatever command was requested
    cli.invoke()?;
    Ok(())
}
