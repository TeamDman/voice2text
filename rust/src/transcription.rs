// transcription.rs

use crate::config::AppConfig;
use crate::microphone::{AudioChunk, SAMPLE_RATE};
use anyhow::{bail, Context, Result};
use chrono::{DateTime, Local};
use cpal::SampleRate;
use reqwest::blocking::Client;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tracing::debug;
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::PathBuf;
use uuid::Uuid;

#[derive(Serialize, Deserialize, Debug)]
pub struct TranscriptionResult {
    pub segments: Vec<TranscriptionResultSegment>,
    pub language: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct TranscriptionResultSegment {
    pub text: String,
    pub start: f32,
    pub end: f32,
}

pub fn send_audio_for_transcription(
    api_url: &str,
    audio: AudioChunk,
) -> Result<TranscriptionResult> {
    let audio = audio.downmix();
    if !matches!(audio, AudioChunk {
        channels: 1,
        sample_rate: SAMPLE_RATE,
        ..
    }) {
        bail!("Audio must be mono and 16kHz");
    }
    debug!("Sending {} samples", audio.data.len());

    let body = audio.to_byte_slice();
    debug!("Total body length: {}", body.len());

    // Send the request
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client
        .post(api_url)
        .header("Content-Type", "audio/f32le")
        .body(body)
        .send()
        .context("Failed to send transcription request")?;

    if response.status().is_success() {
        let result: Value = response.json()?;
        debug!("Transcription response: {:?}", result);
        let result = serde_json::from_value::<TranscriptionResult>(result)
            .context("Failed to parse transcription response")?;
        Ok(result)
    } else {
        Err(anyhow::anyhow!(
            "Transcription API error: {}",
            response.status()
        ))
    }
}


pub fn list_transcript_paths(config: &AppConfig) -> Result<()> {
    let dir = &config.transcription_results_dir;
    let entries = fs::read_dir(dir).context("Failed to read transcription directory")?;
    for entry in entries {
        let entry = entry?;
        let path = entry.path();
        if path.is_file() {
            println!("{}", path.display());
        }
    }
    Ok(())
}

pub fn show_latest_transcript(config: &AppConfig) -> Result<()> {
    let dir = &config.transcription_results_dir;
    let mut entries = fs::read_dir(dir)
        .context("Failed to read transcription directory")?
        .collect::<Result<Vec<_>, std::io::Error>>()?;

    entries.sort_by_key(|dir| dir.metadata().unwrap().modified().unwrap());

    if let Some(latest_entry) = entries.last() {
        let path = latest_entry.path();
        let contents = fs::read_to_string(&path)?;
        println!("{}", contents);
    } else {
        println!("No transcription files found.");
    }
    Ok(())
}

pub fn save_transcription_result(
    config: &AppConfig,
    result: &TranscriptionResult,
    timestamp: DateTime<Local>,
) -> Result<()> {
    let year = timestamp.format("%Y").to_string();
    let month = timestamp.format("%m").to_string();
    let day = timestamp.format("%d").to_string();

    let dir = config.transcription_results_dir.join(year).join(month);
    fs::create_dir_all(&dir)?;

    let file_path = dir.join(format!("{}.jsonl", day));
    let mut file = OpenOptions::new()
        .create(true)
        .append(true)
        .open(file_path)?;

    let json_line = serde_json::to_string(result)?;
    writeln!(file, "{}", json_line)?;

    Ok(())
}
