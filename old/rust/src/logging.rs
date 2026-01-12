// logging.rs

use std::fs::{create_dir_all, File};

use anyhow::Result;
use itertools::Itertools;
use tracing::level_filters::LevelFilter;
use tracing_error::ErrorLayer;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt, EnvFilter, Layer};

use crate::get_project_dirs;

pub fn get_logs_path() -> Result<std::path::PathBuf> {
    let project_dirs = get_project_dirs()?;
    let dir = project_dirs.data_dir();
    create_dir_all(dir)?;
    Ok(dir.join("mic.log"))
}

pub fn initialize_logging() -> Result<()> {
    let log_path = get_logs_path()?;
    let log_file = File::create(log_path)?;

    let env_filter = EnvFilter::builder()
        .with_default_directive(LevelFilter::INFO.into())
        .from_env_lossy()
        .add_directive(
            format!(
                "
                    {}=debug
                ",
                env!("CARGO_PKG_NAME")
            )
            .lines()
            .map(|line| line.trim())
            .filter(|line| !line.starts_with("//"))
            .filter(|line| !line.is_empty())
            .join(",")
            .trim()
            .parse()
            .unwrap(),
        );

    let file_subscriber = tracing_subscriber::fmt::layer()
        .with_file(true)
        .with_line_number(true)
        .with_writer(log_file)
        .with_target(false)
        .with_ansi(true)
        .with_filter(env_filter);
    tracing_subscriber::registry()
        .with(file_subscriber)
        .with(ErrorLayer::default())
        .init();

    Ok(())
}
