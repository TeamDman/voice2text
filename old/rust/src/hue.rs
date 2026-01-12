
use crate::app_state::AppState;
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
use std::path::PathBuf;
use std::time::Duration;
use tokio::sync::mpsc::{unbounded_channel, UnboundedReceiver, UnboundedSender};
use tracing::{debug, error, info, warn};

pub enum HueAuthState {
    Unauthenticated,
    AwaitingButtonPress,
    Authenticated,
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



