// main.rs

mod config;
mod logging;
mod microphone;
mod transcription;
mod ui;

use anyhow::{Context, Result};
use clap::{Parser, Subcommand};
use config::AppConfig;
use directories::ProjectDirs;
use microphone::list_microphones;
use tracing_subscriber::field::debug;
use std::path::PathBuf;
use tracing::{debug, error, info};
use transcription::{list_transcript_paths, show_latest_transcript};

#[derive(Parser)]
#[command(name = "mic", version, about)]
struct Cli {
    #[command(subcommand)]
    command: Option<Commands>,
    #[arg(long)]
    config_path: Option<String>,
}

#[derive(Subcommand, Debug)]
enum Commands {
    /// List available microphones
    List,
    /// Manage transcriptions
    Transcript {
        #[command(subcommand)]
        action: TranscriptAction,
    },
    /// Get the config path
    Config,
}

#[derive(Subcommand, Debug)]
enum TranscriptAction {
    /// List paths to transcription files
    PathsList,
    /// Show the latest transcription
    ShowLatest,
}

pub fn get_project_dirs() -> Result<ProjectDirs> {
    ProjectDirs::from("ca", "teamdman", "mic").context("Unable to determine config directory")
}

pub fn get_config_path() -> Result<PathBuf> {
    let project_dirs = get_project_dirs()?;
    Ok(project_dirs.config_dir().join("config.json"))
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging
    logging::initialize_logging()?;

    info!("Hello, world!");

    let cli = Cli::parse();

    // Determine config path
    let config_path = match cli.config_path.map(PathBuf::from) {
        Some(path) => path,
        None => get_config_path()?,
    };

    debug!("Using config path: {:?}", config_path);

    // Load configuration
    let mut config = AppConfig::load(&config_path)?;

    info!("Running app with command {:?}", cli.command);

    match cli.command {
        Some(Commands::List) => {
            list_microphones_command()?;
        }
        Some(Commands::Transcript { action }) => match action {
            TranscriptAction::PathsList => {
                list_transcript_paths(&config)?;
            }
            TranscriptAction::ShowLatest => {
                show_latest_transcript(&config)?;
            }
        },
        Some(Commands::Config) => {
            println!("{}", config_path.display());
        }
        None => {
            // Launch interactive application
            ui::run_app(&mut config).await?;
        }
    }

    info!("Goodbye, world!");
    Ok(())
}

fn list_microphones_command() -> Result<()> {
    let microphones = list_microphones();
    for mic in microphones {
        println!("{}", mic);
    }
    Ok(())
}
