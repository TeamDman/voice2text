use crate::cli::ToArgs;
use crate::logging::LoggingConfig;
use arbitrary::Arbitrary;
use chrono::Local;
use clap::Args;
use std::ffi::OsString;
use std::path::PathBuf;
use std::str::FromStr;
use tracing::level_filters::LevelFilter;

#[derive(Args, Default, Arbitrary, PartialEq, Debug)]
pub struct GlobalArgs {
    /// Enable debug logging, including backtraces on panics.
    #[arg(long, global = true, default_value_t = false)]
    pub debug: bool,

    /// Log level filter directive.
    #[arg(long, global = true, value_name = "DIRECTIVE")]
    pub log_filter: Option<String>,

    /// Write structured ndjson logs to this file or directory. If a directory is provided,
    /// a filename will be generated there. If omitted, no JSON log file will be written.
    #[arg(long, global = true, value_name = "FILE|DIR")]
    pub log_file: Option<PathBuf>,
}

impl GlobalArgs {
    /// # Errors
    ///
    /// This function will return an error if the log filter string is invalid.
    pub fn logging_config(&self) -> eyre::Result<LoggingConfig> {
        Ok(LoggingConfig {
            default_directive: match (self.debug, &self.log_filter) {
                (true, _) => LevelFilter::DEBUG,
                (false, Some(filter)) => LevelFilter::from_str(filter)?,
                (false, None) => LevelFilter::INFO,
            }
            .into(),
            json_log_path: match &self.log_file {
                None => None,
                Some(path) if path.is_dir() => {
                    let timestamp = Local::now().format("%Y-%m-%d_%H-%M-%S");
                    let filename = format!("log_{timestamp}.ndjson");
                    Some(path.join(filename))
                }
                Some(path) => Some(path.clone()),
            },
        })
    }
}

impl ToArgs for GlobalArgs {
    fn to_args(&self) -> Vec<OsString> {
        let mut args = Vec::new();
        if self.debug {
            args.push("--debug".into());
        }
        if let Some(filter) = &self.log_filter {
            args.push("--log-filter".into());
            args.push(filter.into());
        }
        if let Some(path) = &self.log_file {
            args.push("--log-file".into());
            args.push(path.as_os_str().into());
        }
        args
    }
}
