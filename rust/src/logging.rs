// logging.rs

use std::fs::{create_dir_all, File};

use anyhow::Result;
use itertools::Itertools;
use tracing::level_filters::LevelFilter;
use tracing_subscriber::Layer;
use std::io::{self, Write};
use std::sync::Mutex;
use tracing_error::ErrorLayer;
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

use crate::get_project_dirs;

pub fn get_logs_path() -> Result<std::path::PathBuf> {
    let project_dirs = get_project_dirs()?;
    let dir = project_dirs.data_dir();
    create_dir_all(dir)?;
    Ok(dir.join("mic.log"))
}

// Define a custom writer that flushes after every write
struct FlushingWriter {
    file: Mutex<File>, // Wrap the file in a mutex to safely access it across threads
}

impl Write for FlushingWriter {
    fn write(&mut self, buf: &[u8]) -> io::Result<usize> {
        let mut file = self.file.lock().unwrap();
        let result = file.write(buf);
        file.flush()?; // Flush the file after each write
        result
    }

    fn flush(&mut self) -> io::Result<()> {
        let mut file = self.file.lock().unwrap();
        file.flush()
    }
}

pub fn initialize_logging() -> Result<()> {
    let log_path = get_logs_path()?;
    let log_file = File::create(log_path)?;

    // Wrap the log file in the flushing writer
    let flushing_writer = Mutex::new(FlushingWriter {
        file: Mutex::new(log_file),
    });

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

    let file_subscriber = fmt::layer()
        .with_file(true)
        .with_line_number(true)
        .with_writer(flushing_writer) // Use the flushing writer
        .with_target(false)
        .with_ansi(false)
        .with_filter(env_filter);

    tracing_subscriber::registry()
        .with(file_subscriber)
        .with(ErrorLayer::default())
        .init();

    Ok(())
}
