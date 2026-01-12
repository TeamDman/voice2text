use crate::config::AppConfig;
use crate::get_config_path;
use crate::hue::HueAuthState;
use crate::logging::get_logs_path;
use crate::microphone::{
    hook_microphones, process_raw_audio, AudioChunk, Microphone, MicrophoneState,
};
use crate::transcription::{
    save_transcription_result, send_audio_for_transcription, TranscriptionResult,
};
use anyhow::Context;
use crossterm::event::{Event as CEvent, EventStream, KeyCode, KeyEvent, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use futures::{FutureExt, StreamExt};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Tabs};
use ratatui::{DefaultTerminal, Frame, Terminal};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tracing::{debug, error, info, warn};

pub struct AppState {
    pub config: AppConfig,
    pub config_path: PathBuf,
    pub terminal: Option<DefaultTerminal>,
    pub transcription_callbacks: Vec<TranscriptionCallback>,
    pub activity_log: Vec<String>,
    pub hue_auth_state: HueAuthState,
    log_sender: UnboundedSender<String>,
    pub log_receiver: UnboundedReceiver<String>,
    pub raw_audio_sender: UnboundedSender<AudioChunk>,
    pub raw_audio_receiver: UnboundedReceiver<AudioChunk>,
    pub batch_audio_sender: UnboundedSender<AudioChunk>,
    pub batch_audio_receiver: UnboundedReceiver<AudioChunk>,
    pub transcription_sender: UnboundedSender<TranscriptionResult>,
    pub transcription_receiver: UnboundedReceiver<TranscriptionResult>,
    pub microphones: HashMap<String, Microphone>,
}
impl AppState {
    pub fn new(config: AppConfig, config_path: PathBuf, terminal: DefaultTerminal) -> AppState {
        let (log_sender, log_receiver) = unbounded_channel::<String>();
        let (transcription_sender, transcription_receiver) =
            unbounded_channel::<TranscriptionResult>();
        let (raw_audio_sender, raw_audio_receiver) = unbounded_channel::<AudioChunk>();
        let (batch_audio_sender, batch_audio_receiver) = unbounded_channel::<AudioChunk>();

        AppState {
            config,
            config_path,
            terminal: Some(terminal),
            transcription_callbacks: vec![TranscriptionCallback::WriteJsonLine],
            activity_log: Vec::new(),
            hue_auth_state: HueAuthState::Unauthenticated,
            log_sender,
            log_receiver,
            raw_audio_sender,
            raw_audio_receiver,
            batch_audio_sender,
            batch_audio_receiver,
            transcription_sender,
            transcription_receiver,
            microphones: HashMap::default(),
        }
    }
    pub async fn light_list(&self) -> String {
        match fetch_lights(&self.config).await {
            Ok(lights) => lights
                .iter()
                .map(|(id, name)| format!("{}: {}", id, name))
                .collect::<Vec<String>>()
                .join("\n"),
            Err(e) => {
                self.push_activity_log(format!("Error fetching lights: {}", e));
                "".to_string()
            }
        }
    }
    pub fn push_activity_log(&self, entry: impl AsRef<str>) {
        if let Err(e) = self.log_sender.send(entry.as_ref().to_owned()) {
            error!("Error sending log entry: {}", e);
        };
    }
    pub fn add_microphone(&mut self, mic: Microphone) {
        self.microphones.insert(mic.name.clone(), mic);
    }
}
