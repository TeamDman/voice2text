// config.rs

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::io::Write;
use std::path::PathBuf;

use crate::get_project_dirs;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct AppConfig {
    pub config_editor: String,
    pub big_config_editor: String,
    pub logs_editor: String,
    pub transcription_api_url: String,
    pub transcription_results_dir: PathBuf,
    pub key_config: KeyConfig,
    pub microphones: HashMap<String, MicrophoneConfig>,
}

impl Default for AppConfig {
    fn default() -> Self {
        AppConfig {
            config_editor: "hx".to_string(),
            big_config_editor: "code.cmd".to_string(),
            logs_editor: "code.cmd".to_string(),
            transcription_api_url: "https://127.0.0.1:8383/transcribe".to_string(),
            transcription_results_dir: get_project_dirs()
                .map(|x| x.data_dir().join("transcripts"))
                .ok()
                .unwrap_or_else(|| PathBuf::from("./transcripts")),
            key_config: KeyConfig::default(),
            microphones: HashMap::new(),
        }
    }
}

impl AppConfig {
    pub fn load(path: &PathBuf) -> anyhow::Result<Self> {
        if !path.exists() {
            let config = AppConfig::default();
            fs::create_dir_all(path.parent().unwrap())?;
            let config_data = serde_json::to_string_pretty(&config)?;
            fs::write(path, config_data)?;
            return Ok(config);
        }
        match AppConfig::load_inner(path) {
            Ok(config) => Ok(config),
            Err(e) => {
                // prompt the user to panic or overwrite with defualt
                eprintln!("Error loading config: {:?}", e);
                eprintln!("Overwrite with default config? (y/n)");
                std::io::stderr().flush()?;
                let mut input = String::new();
                std::io::stdin().read_line(&mut input)?;
                if input.trim().to_lowercase() == "y" {
                    let config = AppConfig::default();
                    let config_data = serde_json::to_string_pretty(&config)?;
                    fs::write(path, config_data)?;
                    Ok(config)
                } else {
                    Err(e)
                }
            }
        }
    }
    fn load_inner(path: &PathBuf) -> anyhow::Result<Self> {
        let config_data = fs::read_to_string(path)?;
        let config: AppConfig = serde_json::from_str(&config_data)?;
        Ok(config)
    }

    pub fn save(&self, path: &PathBuf) -> anyhow::Result<()> {
        let parent_dir = path.parent().unwrap();
        fs::create_dir_all(parent_dir)?;
        let config_data = serde_json::to_string_pretty(self)?;
        fs::write(path, config_data)?;
        Ok(())
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct MicrophoneConfig {
    pub samples_until_idle: u32,
    pub activity_threshold_amplitude: f32,
    pub enabled: bool,
}

impl Default for MicrophoneConfig {
    fn default() -> Self {
        MicrophoneConfig {
            samples_until_idle: 44100, // 1 second at 44.1kHz
            activity_threshold_amplitude: 0.01,
            enabled: true,
        }
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct KeyConfig {
    pub quit: char,
    pub help: char,
    pub mic_toggle_disabled: char,
    pub mic_cycle_mode: char,
    pub callback_toggle_write: char,
    pub callback_toggle_typewriter: char,
    pub edit_config: char,
    pub open_config: char,
    pub open_logs: char,
}

impl Default for KeyConfig {
    fn default() -> Self {
        KeyConfig {
            quit: 'q',
            help: 'h',
            mic_toggle_disabled: 'd',
            mic_cycle_mode: 'm',
            callback_toggle_write: 'w',
            callback_toggle_typewriter: 't',
            edit_config: 'e',
            open_config: 'b',
            open_logs: 'b',
        }
    }
}
