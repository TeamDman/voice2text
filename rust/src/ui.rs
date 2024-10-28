// ui.rs

use crate::config::AppConfig;
use crate::get_config_path;
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
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tracing::{debug, error, info, warn};

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
enum TranscriptionCallback {
    WriteJsonLine,
    ChatLights,
}

enum HueAuthState {
    Unauthenticated,
    AwaitingButtonPress,
    Authenticated,
}

pub struct AppState {
    pub config: AppConfig,
    terminal: Option<DefaultTerminal>,
    transcription_callbacks: Vec<TranscriptionCallback>,
    activity_log: Vec<String>,
    hue_auth_state: HueAuthState,
    log_sender: UnboundedSender<String>,
    log_receiver: UnboundedReceiver<String>,
    pub raw_audio_sender: UnboundedSender<AudioChunk>,
    raw_audio_receiver: UnboundedReceiver<AudioChunk>,
    pub batch_audio_sender: UnboundedSender<AudioChunk>,
    batch_audio_receiver: UnboundedReceiver<AudioChunk>,
    pub transcription_sender: UnboundedSender<TranscriptionResult>,
    transcription_receiver: UnboundedReceiver<TranscriptionResult>,
    microphones: HashMap<String, Microphone>,
}
impl AppState {
    fn new(config: AppConfig, terminal: DefaultTerminal) -> AppState {
        let (log_sender, log_receiver) = unbounded_channel::<String>();
        let (transcription_sender, transcription_receiver) =
            unbounded_channel::<TranscriptionResult>();
        let (raw_audio_sender, raw_audio_receiver) = unbounded_channel::<AudioChunk>();
        let (batch_audio_sender, batch_audio_receiver) = unbounded_channel::<AudioChunk>();

        AppState {
            config,
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
    async fn light_list(&self) -> String {
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

pub async fn run_app(config: AppConfig) -> anyhow::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let terminal = Terminal::new(backend)?;

    // Create app state
    let mut app_state = AppState::new(config, terminal);

    // Restore hue authentication
    if app_state.config.hue_username.is_some() {
        app_state.hue_auth_state = HueAuthState::Authenticated;
    };

    // Start microphones
    hook_microphones(&mut app_state)?;

    // Main loop
    let res = run_ui(&mut app_state).await;

    let mut terminal = app_state.terminal.take().unwrap();
    // Restore terminal
    disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    res
}

async fn process_batch_audio(
    app_state: &mut AppState,
    audio_data: AudioChunk,
) -> anyhow::Result<()> {
    debug!(
        "Received audio data for transcription, got {} samples",
        audio_data.data.len()
    );
    match send_audio_for_transcription(&app_state.config.transcription_api_url, audio_data).await {
        Ok(result) => {
            let timestamp = chrono::Local::now();
            if let Err(e) = save_transcription_result(&app_state.config, &result, timestamp) {
                error!("Failed to save transcription: {}", e);
            }

            for segment in &result.segments {
                info!("Heard \"{}\"", segment.text);
                app_state.push_activity_log(format!("Heard \"{}\"", segment.text));
            }
            app_state.transcription_sender.send(result)?;
        }
        Err(e) => {
            error!("Transcription failed: {}", e);
            app_state.push_activity_log(format!("Transcription failed: {}", e));
        }
    }
    Ok(())
}

async fn run_ui(mut app_state: &mut AppState) -> anyhow::Result<()> {
    let tick_rate = Duration::from_millis(200);
    let mut interval = tokio::time::interval(tick_rate);
    let mut crossterm_event_stream = EventStream::new();
    loop {
        let mut terminal = app_state.terminal.take().unwrap();
        terminal.draw(|f| ui(f, &app_state))?;
        app_state.terminal = Some(terminal);

        let delay = interval.tick();
        let crossterm_event = crossterm_event_stream.next().fuse();
        let raw_audio_chunk = app_state.raw_audio_receiver.recv();
        let batch_audio_chunk = app_state.batch_audio_receiver.recv();
        tokio::select! {
            _ = delay => {
                // this branch ensures the UI redraws frequently
            }
            maybe_event = crossterm_event => {
                match maybe_event {
                    Some(Ok(event)) => {
                        if let CEvent::Key(key) = event {
                            if key.kind != KeyEventKind::Press {
                                continue;
                            }
                            match handle_key_event(key, &mut app_state).await? {
                                KeyHandlerResult::Break => {
                                    return Ok(());
                                }
                                KeyHandlerResult::Continue => {}
                            }
                        }
                    }
                    Some(Err(e)) => {
                        error!("Error reading event: {}", e);
                        app_state.push_activity_log(format!("Error reading event: {}", e));
                    }
                    None => {}
                }
            }
            Some(log) = app_state.log_receiver.recv() => {
                app_state.activity_log.push(log);
            }
            Some(transcription) = app_state.transcription_receiver.recv() => {
                handle_transcription_result(&mut app_state, transcription).await?;
            }
            Some(chunk) = raw_audio_chunk => {
                if let Some(mic) = app_state.microphones.get_mut(&chunk.mic_name) {
                    process_raw_audio(chunk, &mut mic.state, &app_state.batch_audio_sender);
                } else {
                    warn!("Received audio chunk for unknown mic: {}", chunk.mic_name);
                }
            }
            Some(chunk) = batch_audio_chunk => {
                if let Err(e) = process_batch_audio(&mut app_state, chunk).await {
                    error!("Error handling audio data: {}", e);
                    app_state.push_activity_log(format!("Error handling audio data: {}", e));
                };
            }
        }
    }
}

async fn handle_transcription_result(
    app_state: &mut AppState,
    transcription: TranscriptionResult,
) -> anyhow::Result<()> {
    let chat_light = app_state
        .transcription_callbacks
        .contains(&TranscriptionCallback::ChatLights);
    for segment in &transcription.segments {
        if chat_light {
            if let Err(e) = handle_hue_llm_voice_commands(&app_state, &segment.text).await {
                app_state.push_activity_log(format!("Error processing ChatLights: {}", e));
            }
        }
    }
    Ok(())
}

enum KeyHandlerResult {
    Continue,
    Break,
}
async fn handle_key_event(
    key: KeyEvent,
    app_state: &mut AppState,
) -> anyhow::Result<KeyHandlerResult> {
    match key.code {
        KeyCode::Char(x) if x == app_state.config.key_config.quit => {
            return Ok(KeyHandlerResult::Break);
        }
        KeyCode::Char(x) if x == app_state.config.key_config.help => {
            app_state.push_activity_log("Help requested, TODO: implement this lol".to_string());
        }
        KeyCode::Char(x) if x == app_state.config.key_config.edit_config => {
            edit_config(
                &app_state.config.config_editor,
                &mut app_state.terminal.as_mut().unwrap(),
            )?;
        }
        KeyCode::Char(x) if x == app_state.config.key_config.open_config => {
            open_config(&app_state.config.big_config_editor)?;
        }
        KeyCode::Char(x) if x == app_state.config.key_config.open_logs => {
            open_logs(&app_state.config)?;
        }
        KeyCode::Char(x) if x == app_state.config.key_config.authenticate_lights => {
            authenticate_lights(app_state).await?;
        }
        KeyCode::Char(x) if x == app_state.config.key_config.toggle_chat_lights => {
            toggle_chat_lights_callback(app_state);
        }
        _ => {}
    }
    Ok(KeyHandlerResult::Continue)
}

fn ui(f: &mut Frame, app_state: &AppState) {
    let size = f.area();

    // Divide layout
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .margin(1)
        .constraints(
            [
                Constraint::Percentage(30), // Microphones
                Constraint::Percentage(20), // Lights
                Constraint::Min(1),         // Activity log
                Constraint::Length(3),      // Callbacks
            ]
            .as_ref(),
        )
        .split(size);

    // Microphone list
    let mic_items: Vec<ListItem> = app_state
        .microphones
        .values()
        .map(|mic| {
            let status = match &mic.state {
                MicrophoneState::Disabled => "DISABLED".to_string(),
                MicrophoneState::WaitingForPushToTalk => "IDLE - WaitingForPushToTalk".to_string(),
                MicrophoneState::WaitingForVoiceActivity => {
                    "IDLE - WaitingForVoiceActivity".to_string()
                }
                MicrophoneState::VoiceActivated(active_state) => {
                    format!("LISTENING - Samples: {}", active_state.data_so_far.len())
                }
                _ => "UNKNOWN".to_string(),
            };

            ListItem::new(Span::raw(format!("{} | {}", mic.name, status)))
        })
        .collect();

    let mic_list =
        List::new(mic_items).block(Block::default().borders(Borders::ALL).title("Microphones"));

    f.render_widget(mic_list, chunks[0]);

    // Lights block
    let lights_block = match &app_state.hue_auth_state {
        HueAuthState::Unauthenticated => {
            let text = format!(
                "Not authenticated. Press '{}' to authenticate.",
                app_state.config.key_config.authenticate_lights
            );
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Lights"))
        }
        HueAuthState::AwaitingButtonPress => {
            let text = "Please press the link button on the Hue bridge.";
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Lights"))
        }
        HueAuthState::Authenticated => {
            let text = format!("Authenticated");
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Lights"))
        }
    };

    f.render_widget(lights_block, chunks[1]);

    // Activity log
    let log_items: Vec<ListItem> = app_state
        .activity_log
        .iter()
        .rev()
        .map(|entry| ListItem::new(entry.clone()))
        .collect();

    let log_list =
        List::new(log_items).block(Block::default().borders(Borders::ALL).title("Activity Log"));

    f.render_widget(log_list, chunks[2]);

    // Callbacks
    let callback_titles: Vec<&str> = app_state
        .transcription_callbacks
        .iter()
        .map(|callback| match callback {
            TranscriptionCallback::WriteJsonLine => "WriteJsonLine",
            TranscriptionCallback::ChatLights => "ChatLights",
        })
        .collect();

    let tabs = Tabs::new(callback_titles.iter().cloned().map(Span::from))
        .block(Block::default().borders(Borders::ALL).title("Callbacks"))
        .style(Style::default().fg(Color::Cyan));

    f.render_widget(tabs, chunks[3]);
}

async fn authenticate_lights(app_state: &mut AppState) -> anyhow::Result<()> {
    // only proceed if not already authenticated
    if let HueAuthState::Authenticated { .. } = app_state.hue_auth_state {
        app_state.push_activity_log("Already authenticated with Hue bridge.");
        return Ok(());
    }

    let bridge_ip = &app_state.config.hue_bridge_ip;
    if bridge_ip.is_empty() {
        app_state.push_activity_log("Hue bridge IP not set in config.");
        return Ok(());
    }

    let url = format!("https://{}/api", bridge_ip);

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client
        .post(&url)
        .json(&serde_json::json!({"devicetype": "mic_app#rust"}))
        .send()
        .await?;

    let response_json: serde_json::Value = response.json().await?;

    let result = response_json
        .as_array()
        .ok_or_else(|| anyhow::anyhow!("Unexpected response"))?;

    if result.is_empty() {
        anyhow::bail!("Empty response from Hue bridge");
    }

    let first_item = &result[0];

    if let Some(success) = first_item.get("success") {
        let username = success
            .get("username")
            .and_then(|v| v.as_str())
            .ok_or_else(|| anyhow::anyhow!("No username in success response"))?
            .to_string();

        app_state.config.hue_username = Some(username.clone());
        app_state.hue_auth_state = HueAuthState::Authenticated;

        // Save the config with the new username
        let config_path = get_config_path()?;
        app_state.config.save(&config_path)?;

        app_state.push_activity_log("Successfully authenticated with Hue bridge.");
    } else if let Some(error) = first_item.get("error") {
        let error_type = error.get("type").and_then(|v| v.as_i64()).unwrap_or(0);
        let description = error
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if error_type == 101 {
            // link button not pressed
            app_state.hue_auth_state = HueAuthState::AwaitingButtonPress;

            app_state.push_activity_log("Please press the link button on the Hue bridge.");
        } else {
            app_state.push_activity_log(format!(
                "Error authenticating with Hue bridge: {}",
                description
            ));
        }
    } else {
        app_state.push_activity_log("Unknown response from Hue bridge.");
    }

    Ok(())
}

fn edit_config(editor: &str, terminal: &mut DefaultTerminal) -> anyhow::Result<()> {
    use std::process::Command;

    // Get config path
    let config_path = get_config_path()?;

    // Restore terminal
    disable_raw_mode()?;
    crossterm::execute!(io::stdout(), crossterm::terminal::LeaveAlternateScreen)?;
    let status = Command::new(editor).arg(&config_path).status()?;

    // Re-enter alternate screen
    crossterm::execute!(io::stdout(), crossterm::terminal::EnterAlternateScreen)?;
    enable_raw_mode()?;
    terminal.clear()?;

    if status.success() {
        info!("Config edited successfully");
        warn!("TODO: reload config here");
    } else {
        error!("Failed to edit config");
    }
    Ok(())
}

fn open_config(editor: &str) -> anyhow::Result<()> {
    use std::process::Command;
    let config_path = get_config_path()?;
    let status = Command::new(editor).arg(&config_path).status()?;
    if status.success() {
        info!("Config opened successfully");
    } else {
        error!("Failed to edit config");
    }
    Ok(())
}

fn open_logs(config: &AppConfig) -> anyhow::Result<()> {
    use std::process::Command;

    // Get log path
    let log_path = get_logs_path()?;

    // Open logs
    let status = Command::new(&config.logs_editor).arg(&log_path).status()?;

    if status.success() {
        info!("Logs opened successfully");
    } else {
        error!("Failed to open logs");
    }

    Ok(())
}

fn toggle_chat_lights_callback(app_state: &mut AppState) {
    info!("Toggling ChatLights callback");
    if app_state
        .transcription_callbacks
        .contains(&TranscriptionCallback::ChatLights)
    {
        app_state
            .transcription_callbacks
            .retain(|c| c != &TranscriptionCallback::ChatLights);

        app_state.push_activity_log("Disabled ChatLights callback.".to_string());
    } else {
        app_state
            .transcription_callbacks
            .push(TranscriptionCallback::ChatLights);

        app_state.push_activity_log("Enabled ChatLights callback.".to_string());
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct LightUpdateResponse {
    light_updates: Vec<LightUpdate>,
}

#[derive(Serialize, Deserialize, Debug)]
struct LightUpdate {
    light_id: u32,
    red: Option<u8>,    // Red value between 0-255
    green: Option<u8>,  // Green value between 0-255
    blue: Option<u8>,   // Blue value between 0-255
    brightness: Option<u8>, // Brightness between 1-254
    on: Option<bool>,
}
fn rgb_to_hsv(r: u8, g: u8, b: u8) -> (u16, u8, u8) {
    let r = r as f32 / 255.0;
    let g = g as f32 / 255.0;
    let b = b as f32 / 255.0;

    let max = r.max(g.max(b));
    let min = r.min(g.min(b));
    let delta = max - min;

    // Hue calculation
    let mut h = 0.0;
    if delta != 0.0 {
        if max == r {
            h = 60.0 * (((g - b) / delta) % 6.0);
        } else if max == g {
            h = 60.0 * (((b - r) / delta) + 2.0);
        } else if max == b {
            h = 60.0 * (((r - g) / delta) + 4.0);
        }
    }
    if h < 0.0 {
        h += 360.0;
    }

    // Saturation calculation
    let s = if max == 0.0 { 0.0 } else { delta / max };

    // Value calculation
    let v = max;

    // Map h from [0,360) to [0,65535]
    let hue = (h / 360.0 * 65535.0).round() as u16;

    // Map s from [0,1] to [0,254]
    let sat = (s * 254.0).round() as u8;

    // Map v from [0,1] to [1,254]
    let bri = (v * 253.0 + 1.0).round() as u8;

    (hue, sat, bri)
}

async fn handle_hue_llm_voice_commands(app_state: &AppState, transcript: &str) -> anyhow::Result<()> {
    info!("Processing ChatLights for \"{}\"", transcript);
    let client = Client::new();
    let model_api_url = "http://localhost:11434/api/generate"; // TODO: make config variable
    let model = "x/llama3.2-vision"; // TODO: make config variable

    // Build the prompt
    let prompt = format!(
        r#"
You are a light controlling robot.
Your job is to detect when a user is instructing you to change the lights.

{}


Your output should have the following structure.
{{
    "light_updates": [ {{
        "light_id": number,
        "red": number (0-255),
        "green": number (0-255),
        "blue": number (0-255),
        "brightness": number (1-254),
        "on": bool
    }} ]
}}

If it seems like the user is not talking to the robot, then an empty array should be returned for the "light_updates" property.

Transcript:
"{}"

Respond only with the JSON output.
"#,
        app_state.light_list().await, // We'll implement this method
        transcript
    );

    // Send the request
    let response = client
        .post(model_api_url)
        .json(&serde_json::json!({
            "model": model,
            "prompt": prompt,
            "stream": false,
        }))
        .send().await?;

    let response_json: serde_json::Value = response.json().await?;
    let generated_text = response_json
        .get("response")
        .and_then(|v| v.as_str())
        .ok_or_else(|| anyhow::anyhow!("No response from model"))?;

    // Parse the generated text as JSON
    let light_updates: LightUpdateResponse = serde_json::from_str(generated_text.trim())
        .context("Failed to parse model response as JSON")?;

    // Process the light updates
    if !light_updates.light_updates.is_empty() {
        // Send commands to the Hue bridge
        for update in light_updates.light_updates {
            send_hue_command(app_state, update).await?;
        }
    }

    Ok(())
}

async fn send_hue_command(app_state: &AppState, update: LightUpdate) -> anyhow::Result<()> {
    let bridge_ip = &app_state.config.hue_bridge_ip;
    let username = match &app_state.config.hue_username {
        Some(u) => u,
        None => {
            app_state.push_activity_log("Not authenticated with Hue bridge.".to_string());
            return Ok(());
        }
    };

    let url = format!(
        "https://{}/api/{}/lights/{}/state",
        bridge_ip, username, update.light_id
    );

    let mut body = serde_json::Map::new();

    if let Some(on) = update.on {
        body.insert("on".to_string(), serde_json::Value::Bool(on));
    }

    if let (Some(red), Some(green), Some(blue)) = (update.red, update.green, update.blue) {
        let (hue, sat, bri) = rgb_to_hsv(red, green, blue);
        body.insert("hue".to_string(), serde_json::Value::Number(hue.into()));
        body.insert("sat".to_string(), serde_json::Value::Number(sat.into()));
        body.insert("bri".to_string(), serde_json::Value::Number(bri.into()));
    } else if let Some(bri) = update.brightness {
        body.insert("bri".to_string(), serde_json::Value::Number(bri.into()));
    }

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client.put(&url).json(&body).send().await?;

    let response_json: serde_json::Value = response.json().await?;

    app_state.push_activity_log(format!(
        "Sent light command to light {}: {:?}",
        update.light_id, response_json
    ));

    Ok(())
}

async fn fetch_lights(config: &AppConfig) -> anyhow::Result<HashMap<u32, String>> {
    let bridge_ip = &config.hue_bridge_ip;
    let username = match &config.hue_username {
        Some(u) => u,
        None => anyhow::bail!("Not authenticated with Hue bridge"),
    };

    let url = format!("https://{}/api/{}/lights", bridge_ip, username);

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client.get(&url).send().await?;

    let response_json: serde_json::Value = response.json().await?;

    let lights = response_json
        .as_object()
        .ok_or_else(|| anyhow::anyhow!("Invalid lights response"))?;

    let mut light_map = HashMap::new();

    for (id_str, light_info) in lights {
        if let Ok(id) = id_str.parse::<u32>() {
            if let Some(name) = light_info.get("name").and_then(|n| n.as_str()) {
                light_map.insert(id, name.to_string());
            }
        }
    }

    Ok(light_map)
}
