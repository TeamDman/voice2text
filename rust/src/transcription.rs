// transcription.rs

use crate::config::AppConfig;
use crate::microphone::AudioChunk;
use anyhow::{Context, Result};
use chrono::{DateTime, Local};
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
    audio_data: &AudioChunk,
) -> Result<TranscriptionResult> {
    // Convert audio data to WAV format
    let wav_data = generate_wav_data(&audio_data.data)?;

    // Convert sample rate and channel count to bytes
    let sample_rate_bytes = audio_data.sample_rate.0.to_le_bytes().to_vec();
    let channels_bytes = audio_data.channels.to_le_bytes().to_vec();

    // Log the values being sent
    debug!("Sending audio with sample rate: {}", audio_data.sample_rate.0);
    debug!("Sending audio with channel count: {}", audio_data.channels);
    debug!("Audio data length: {}", wav_data.len());

    // Create the body to send: sample rate + channel count + audio data
    let mut body = sample_rate_bytes;
    body.extend_from_slice(&channels_bytes);
    body.extend_from_slice(&wav_data);

    // Log the total byte length
    debug!("Total body length: {}", body.len());

    // Send the request
    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client
        .post(api_url)
        .header("Content-Type", "audio/wav")
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


fn generate_wav_data(audio_data: &[f32]) -> Result<Vec<u8>> {
    let spec = hound::WavSpec {
        channels: 1,
        sample_rate: 44100,
        bits_per_sample: 16,
        sample_format: hound::SampleFormat::Int,
    };

    let mut cursor = std::io::Cursor::new(Vec::new());
    let mut writer = hound::WavWriter::new(&mut cursor, spec)?;

    for sample in audio_data {
        let amplitude = (sample * i16::MAX as f32) as i16;
        writer.write_sample(amplitude)?;
    }

    writer.finalize()?;
    Ok(cursor.into_inner())
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
