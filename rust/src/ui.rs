// ui.rs

use crate::config::{AppConfig, KeyConfig};
use crate::logging::get_logs_path;
use crate::microphone::{initialize_microphones, start_microphone_streams, AudioChunk, MicrophoneState};
use crate::transcription::{
    save_transcription_result, send_audio_for_transcription, TranscriptionResult,
};
use crate::{get_config_path, get_project_dirs};
use anyhow::Context;
use cpal::SampleRate;
use crossterm::event::{self, Event as CEvent, KeyCode, KeyEventKind};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline, Tabs};
use ratatui::{DefaultTerminal, Frame, Terminal};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crossbeam_channel::{unbounded, Receiver, Sender};

#[derive(Eq, PartialEq, Debug, Clone, Copy)]
enum TranscriptionCallback {
    WriteJsonLine,
    SendTypewriterKeystrokes,
    ChatLights,
}

enum LightAuthState {
    Unauthenticated,
    AwaitingButtonPress,
    Authenticated { username: String },
}

struct AppState {
    config: AppConfig,
    transcription_callbacks: Vec<TranscriptionCallback>,
    activity_log: Arc<Mutex<Vec<String>>>,
    light_auth_state: LightAuthState,
}
impl AppState {
    fn light_list(&self) -> String {
        match fetch_lights(&self.config) {
            Ok(lights) => {
                lights
                    .iter()
                    .map(|(id, name)| format!("{}: {}", id, name))
                    .collect::<Vec<String>>()
                    .join("\n")
            }
            Err(e) => {
                let mut log = self.activity_log.lock().unwrap();
                log.push(format!("Error fetching lights: {}", e));
                "".to_string()
            }
        }
    }
}


pub async fn run_app(config: &mut AppConfig) -> anyhow::Result<()> {
    // Setup terminal
    enable_raw_mode()?;
    let mut stdout = io::stdout();
    crossterm::execute!(stdout, crossterm::terminal::EnterAlternateScreen)?;
    let backend = CrosstermBackend::new(stdout);
    let mut terminal = Terminal::new(backend)?;

    // Channels for inter-thread communication
    let (mic_audio_sender, mic_audio_receiver) = unbounded::<AudioChunk>();

    // Initialize microphones
    let mut microphones = initialize_microphones(&config);

    // Start microphone streams
    start_microphone_streams(&mut microphones, mic_audio_sender.clone());

    // Shared state
    let activity_log = Arc::new(Mutex::new(vec!["Config loaded".to_string()]));

    let light_auth_state = if let Some(username) = config.hue_username.clone() {
        LightAuthState::Authenticated { username }
    } else {
        LightAuthState::Unauthenticated
    };

    let mut app_state = AppState {
        config: config.clone(),
        transcription_callbacks: vec![TranscriptionCallback::WriteJsonLine],
        activity_log: activity_log.clone(),
        light_auth_state,
    };
    let app_state = Arc::new(Mutex::new(app_state));

    // Spawn a thread to handle transcription
    let app_config_clone = app_state.lock().unwrap().config.clone();
    let transcription_receiver_clone = mic_audio_receiver.clone();
    let activity_log_clone = activity_log.clone();
    let app_state_clone = app_state.clone();
    thread::spawn(move || {
        while let Ok(audio_data) = transcription_receiver_clone.recv() {
            debug!(
                "Received audio data for transcription, got {} samples",
                audio_data.data.len()
            );
            match send_audio_for_transcription(&app_config_clone.transcription_api_url, audio_data)
            {
                Ok(result) => {
                    let timestamp = chrono::Local::now();
                    save_transcription_result(&app_config_clone, &result, timestamp)
                        .unwrap_or_else(|e| error!("Failed to save transcription: {}", e));

                    let mut log = activity_log_clone.lock().unwrap();
                    for segment in &result.segments {
                        info!("Heard \"{}\"", segment.text);
                        log.push(format!("Heard \"{}\"", segment.text));

                        // Check if ChatLights callback is enabled
                        let app_state = app_state_clone.lock().unwrap();
                        if app_state
                            .transcription_callbacks
                            .contains(&TranscriptionCallback::ChatLights)
                        {
                            if let Err(e) = process_chat_lights(&app_state, &segment.text) {
                                log.push(format!("Error processing ChatLights: {}", e));
                            }
                        }
                    }
                }
                Err(e) => {
                    error!("Transcription failed: {}", e);
                }
            }
        }
    });

    // Main loop
    let res = run_ui(&mut terminal, &mut microphones, app_state).await;

    // Restore terminal
    disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    res
}

async fn run_ui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    microphones: &mut std::collections::HashMap<String, crate::microphone::Microphone>,
    app_state: Arc<Mutex<AppState>>,
) -> anyhow::Result<()> {
    let tick_rate = Duration::from_millis(200);
    let mut last_tick = Instant::now();

    let app_state = &mut app_state.lock().unwrap();
    loop {
        terminal.draw(|f| ui(f, microphones, app_state))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let CEvent::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }
                match key.code {
                    KeyCode::Char(x) if x == app_state.config.key_config.quit => {
                        break;
                    }
                    KeyCode::Char(x) if x == app_state.config.key_config.help => {}
                    KeyCode::Char(x) if x == app_state.config.key_config.edit_config => {
                        edit_config(&app_state.config.config_editor, terminal)?;
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
            }
        }

        if last_tick.elapsed() >= tick_rate {
            // Update application state
            last_tick = Instant::now();
        }
    }

    Ok(())
}

fn ui(
    f: &mut Frame,
    microphones: &std::collections::HashMap<String, crate::microphone::Microphone>,
    app_state: &AppState,
) {
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
    let mic_items: Vec<ListItem> = microphones
        .values()
        .map(|mic| {
            let state = mic.state.lock().unwrap();
            let status = match &*state {
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
    let lights_block = match &app_state.light_auth_state {
        LightAuthState::Unauthenticated => {
            let text = format!(
                "Not authenticated. Press '{}' to authenticate.",
                app_state.config.key_config.authenticate_lights
            );
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Lights"))
        }
        LightAuthState::AwaitingButtonPress => {
            let text = "Please press the link button on the Hue bridge.";
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Lights"))
        }
        LightAuthState::Authenticated { username } => {
            // For now, just display that we're authenticated
            // let text = format!("Authenticated with username: {}", username);
            let text = format!("Authenticated with username: <redacted>");
            Paragraph::new(text).block(Block::default().borders(Borders::ALL).title("Lights"))
        }
    };

    f.render_widget(lights_block, chunks[1]);

    // Activity log
    let log_items: Vec<ListItem> = app_state
        .activity_log
        .lock()
        .unwrap()
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
            TranscriptionCallback::SendTypewriterKeystrokes => "SendTypewriterKeystrokes",
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
    if let LightAuthState::Authenticated { .. } = app_state.light_auth_state {
        let mut log = app_state.activity_log.lock().unwrap();
        log.push("Already authenticated with Hue bridge.".to_string());
        return Ok(());
    }

    let bridge_ip = &app_state.config.hue_bridge_ip;
    if bridge_ip.is_empty() {
        let mut log = app_state.activity_log.lock().unwrap();
        log.push("Hue bridge IP not set in config.".to_string());
        return Ok(());
    }

    let url = format!("https://{}/api", bridge_ip);

    let client = Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client
        .post(&url)
        .json(&serde_json::json!({"devicetype": "mic_app#rust"}))
        .send().await?;

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
        app_state.light_auth_state = LightAuthState::Authenticated {
            username: username.clone(),
        };

        // Save the config with the new username
        let config_path = get_config_path()?;
        app_state.config.save(&config_path)?;

        let mut log = app_state.activity_log.lock().unwrap();
        log.push("Successfully authenticated with Hue bridge.".to_string());
    } else if let Some(error) = first_item.get("error") {
        let error_type = error.get("type").and_then(|v| v.as_i64()).unwrap_or(0);
        let description = error
            .get("description")
            .and_then(|v| v.as_str())
            .unwrap_or("");
        if error_type == 101 {
            // link button not pressed
            app_state.light_auth_state = LightAuthState::AwaitingButtonPress;
            let mut log = app_state.activity_log.lock().unwrap();
            log.push("Please press the link button on the Hue bridge.".to_string());
        } else {
            let mut log = app_state.activity_log.lock().unwrap();
            log.push(format!("Error authenticating with Hue bridge: {}", description));
        }
    } else {
        let mut log = app_state.activity_log.lock().unwrap();
        log.push("Unknown response from Hue bridge.".to_string());
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
        let mut log = app_state.activity_log.lock().unwrap();
        log.push("Disabled ChatLights callback.".to_string());
    } else {
        app_state
            .transcription_callbacks
            .push(TranscriptionCallback::ChatLights);
        let mut log = app_state.activity_log.lock().unwrap();
        log.push("Enabled ChatLights callback.".to_string());
    }
}

#[derive(Serialize, Deserialize, Debug)]
struct LightUpdateResponse {
    light_updates: Vec<LightUpdate>,
}

#[derive(Serialize, Deserialize, Debug)]
struct LightUpdate {
    light_id: u32,
    hue: Option<u16>,        // Hue value between 0-65535
    saturation: Option<u8>,  // Saturation between 0-254
    brightness: Option<u8>,  // Brightness between 1-254
    on: Option<bool>,
}

fn process_chat_lights(app_state: &AppState, transcript: &str) -> anyhow::Result<()> {
    info!("Processing ChatLights for \"{}\"", transcript);
    let client = reqwest::blocking::Client::new();
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
        "hue": number (0-65535),
        "saturation": number (0-254),
        "brightness": number (1-254),
        "on": bool
    }} ]
}}

If it seems like the user is not talking to the robot, then an empty array should be returned for the "light_updates" property.

Transcript:
"{}"

Respond only with the JSON output.
"#,
        app_state.light_list(), // We'll implement this method
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
        .send()?;

    let response_json: serde_json::Value = response.json()?;
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
            send_light_command(app_state, update)?;
        }
    }

    Ok(())
}

fn send_light_command(app_state: &AppState, update: LightUpdate) -> anyhow::Result<()> {
    let bridge_ip = &app_state.config.hue_bridge_ip;
    let username = match &app_state.config.hue_username {
        Some(u) => u,
        None => {
            let mut log = app_state.activity_log.lock().unwrap();
            log.push("Not authenticated with Hue bridge.".to_string());
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
    if let Some(bri) = update.brightness {
        body.insert("bri".to_string(), serde_json::Value::Number(bri.into()));
    }
    if let Some(hue) = update.hue {
        body.insert("hue".to_string(), serde_json::Value::Number(hue.into()));
    }
    if let Some(sat) = update.saturation {
        body.insert("sat".to_string(), serde_json::Value::Number(sat.into()));
    }

    let client = reqwest::blocking::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client
        .put(&url)
        .json(&body)
        .send()?;

    let response_json: serde_json::Value = response.json()?;

    let mut log = app_state.activity_log.lock().unwrap();
    log.push(format!(
        "Sent light command to light {}: {:?}",
        update.light_id, response_json
    ));

    Ok(())
}


fn fetch_lights(config: &AppConfig) -> anyhow::Result<HashMap<u32, String>> {
    let bridge_ip = &config.hue_bridge_ip;
    let username = match &config.hue_username {
        Some(u) => u,
        None => anyhow::bail!("Not authenticated with Hue bridge"),
    };

    let url = format!("https://{}/api/{}/lights", bridge_ip, username);

    let client = reqwest::blocking::Client::builder()
        .danger_accept_invalid_certs(true)
        .build()?;

    let response = client.get(&url).send()?;

    let response_json: serde_json::Value = response.json()?;

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