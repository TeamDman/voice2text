// ui.rs

use crate::config::{AppConfig, KeyConfig};
use crate::logging::get_logs_path;
use crate::microphone::{initialize_microphones, start_microphone_streams, AudioChunk, MicrophoneState};
use crate::transcription::{
    save_transcription_result, send_audio_for_transcription, TranscriptionResult,
};
use crate::{get_config_path, get_project_dirs};
use cpal::SampleRate;
use crossterm::event::{self, Event as CEvent, KeyCode};
use crossterm::terminal::{disable_raw_mode, enable_raw_mode};
use ratatui::backend::CrosstermBackend;
use ratatui::layout::{Constraint, Direction, Layout};
use ratatui::style::{Color, Style};
use ratatui::text::Span;
use ratatui::widgets::{Block, Borders, List, ListItem, Paragraph, Sparkline, Tabs};
use ratatui::{DefaultTerminal, Frame, Terminal};
use std::io;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};
use tracing::{debug, error, info, warn};

use crossbeam_channel::{unbounded, Receiver, Sender};

enum TranscriptionCallback {
    WriteJsonLine,
    SendTypewriterKeystrokes,
}

struct AppState {
    config: AppConfig,
    transcription_callbacks: Vec<TranscriptionCallback>,
    activity_log: Arc<Mutex<Vec<String>>>,
}


pub fn run_app(config: &mut AppConfig) -> anyhow::Result<()> {
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
    let app_state = AppState {
        config: config.clone(),
        transcription_callbacks: vec![TranscriptionCallback::WriteJsonLine],
        activity_log: activity_log.clone(),
    };

    // Spawn a thread to handle transcription
    let app_config_clone = app_state.config.clone();
    let transcription_receiver_clone = mic_audio_receiver.clone();
    let activity_log_clone = activity_log.clone();
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
                        log.push(format!("Heard \"{}\"", segment.text));
                    }
                }
                Err(e) => {
                    error!("Transcription failed: {}", e);
                }
            }
        }
    });

    // Main loop
    let res = run_ui(&mut terminal, &mut microphones, &app_state);

    // Restore terminal
    disable_raw_mode()?;
    crossterm::execute!(
        terminal.backend_mut(),
        crossterm::terminal::LeaveAlternateScreen
    )?;
    terminal.show_cursor()?;

    res
}

fn run_ui(
    terminal: &mut Terminal<CrosstermBackend<io::Stdout>>,
    microphones: &mut std::collections::HashMap<String, crate::microphone::Microphone>,
    app_state: &AppState,
) -> anyhow::Result<()> {
    let tick_rate = Duration::from_millis(200);
    let mut last_tick = Instant::now();

    loop {
        terminal.draw(|f| ui(f, microphones, app_state))?;

        let timeout = tick_rate
            .checked_sub(last_tick.elapsed())
            .unwrap_or_else(|| Duration::from_secs(0));

        if crossterm::event::poll(timeout)? {
            if let CEvent::Key(key) = event::read()? {
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
                Constraint::Percentage(50),
                Constraint::Length(3),
                Constraint::Min(1),
                Constraint::Length(3),
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
        })
        .collect();

    let tabs = Tabs::new(callback_titles.iter().cloned().map(Span::from))
        .block(Block::default().borders(Borders::ALL).title("Callbacks"))
        .style(Style::default().fg(Color::Cyan));

    f.render_widget(tabs, chunks[3]);
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
