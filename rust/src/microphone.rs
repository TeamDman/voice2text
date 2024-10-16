// microphone.rs

use crate::config::{AppConfig, MicrophoneConfig};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ChannelCount, SampleFormat, SampleRate, Stream};
use std::collections::HashMap;
use std::sync::{Arc, Mutex};
use tracing::{error, info};

pub struct Microphone {
    pub id: String,
    pub name: String,
    pub config: MicrophoneConfig,
    pub state: Arc<Mutex<MicrophoneState>>,
    pub audio_levels: Arc<Mutex<Vec<f32>>>,
    pub stream: Option<Stream>,
}

#[derive(Clone, Debug)]
pub enum MicrophoneState {
    Disabled,
    WaitingForPushToTalk,
    PushToTalkActivated(ActiveMicrophoneState),
    WaitingForVoiceActivity,
    VoiceActivated(ActiveMicrophoneState),
}

#[derive(Clone, Debug)]
pub struct ActiveMicrophoneState {
    pub activity_started: std::time::Instant,
    pub last_activity: std::time::Instant,
    pub data_so_far: Vec<f32>, // Assuming f32 samples
    pub sample_rate: SampleRate,
}

pub fn list_microphones() -> Vec<String> {
    let host = cpal::default_host();
    host.input_devices()
        .unwrap()
        .map(|device| device.name().unwrap_or_default())
        .collect()
}

pub fn initialize_microphones(config: &AppConfig) -> HashMap<String, Microphone> {
    let host = cpal::default_host();
    let devices = host.input_devices().unwrap();

    let mut microphones = HashMap::new();

    for device in devices {
        let name = device.name().unwrap_or_else(|_| "Unknown".to_string());
        let id = name.clone();

        let mic_config = config
            .microphones
            .get(&name)
            .cloned()
            .unwrap_or_else(MicrophoneConfig::default);

        let state = if mic_config.enabled {
            MicrophoneState::WaitingForVoiceActivity
        } else {
            MicrophoneState::Disabled
        };

        let microphone = Microphone {
            id: id.clone(),
            name: name.clone(),
            config: mic_config,
            state: Arc::new(Mutex::new(state)),
            audio_levels: Arc::new(Mutex::new(vec![0.0; 100])),
            stream: None,
        };

        microphones.insert(id, microphone);
    }

    microphones
}

pub struct AudioChunk {
    pub data: Vec<f32>,
    pub sample_rate: SampleRate,
    pub channels: ChannelCount,
}

pub fn start_microphone_streams(
    microphones: &mut HashMap<String, Microphone>,
    transcription_sender: crossbeam_channel::Sender<AudioChunk>,
) {
    for mic in microphones.values_mut() {
        if mic.config.enabled {
            let device = cpal::default_host()
                .input_devices()
                .unwrap()
                .find(|d| d.name().unwrap_or_default() == mic.name)
                .expect("Microphone not found");

            let config = device.default_input_config().unwrap();
            let sample_rate = config.sample_rate();
            let channel_count = config.channels();

            let sample_format = config.sample_format();
            let config = config.into();

            let state_clone = mic.state.clone();
            let audio_levels_clone = mic.audio_levels.clone();
            let transcription_sender_clone = transcription_sender.clone();

            let err_fn = |err| error!("An error occurred on the input stream: {}", err);

            let mic_name = mic.name.clone();

            info!("Starting stream for mic {mic_name} with format {sample_format:?}");
            let stream = match sample_format {
                SampleFormat::F32 => device.build_input_stream(
                    &config,
                    move |data: &[f32], _| {
                        process_input_data_f32(
                            channel_count,
                            sample_rate,
                            &mic_name,
                            data,
                            &state_clone,
                            &audio_levels_clone,
                            &transcription_sender_clone,
                        )
                    },
                    err_fn,
                ),
                SampleFormat::I16 => device.build_input_stream(
                    &config,
                    move |data: &[i16], _| {
                        let data_f32: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                        process_input_data_f32(
                            channel_count,
                            sample_rate,
                            &mic_name,
                            &data_f32,
                            &state_clone,
                            &audio_levels_clone,
                            &transcription_sender_clone,
                        )
                    },
                    err_fn,
                ),
                SampleFormat::U16 => device.build_input_stream(
                    &config,
                    move |data: &[u16], _| {
                        let data_f32: Vec<f32> =
                            data.iter().map(|&s| s as f32 / 65536.0 - 0.5).collect();
                        process_input_data_f32(
                            channel_count,
                            sample_rate,
                            &mic_name,
                            &data_f32,
                            &state_clone,
                            &audio_levels_clone,
                            &transcription_sender_clone,
                        )
                    },
                    err_fn,
                ),
                _ => panic!("Unsupported sample format"),
            }
            .expect("Failed to build input stream");

            stream.play().expect("Failed to start input stream");
            mic.stream = Some(stream);
        }
    }
}

fn process_input_data_f32(
    channel_count: ChannelCount,
    sample_rate: SampleRate,
    mic_name: &str,
    data: &[f32],
    state: &Arc<Mutex<MicrophoneState>>,
    audio_levels: &Arc<Mutex<Vec<f32>>>,
    transcription_sender: &crossbeam_channel::Sender<AudioChunk>,
) {
    let amplitude = data.iter().map(|&s| s.abs()).sum::<f32>() / data.len() as f32;

    // Update audio levels for sparkline
    {
        let mut levels = audio_levels.lock().unwrap();
        levels.push(amplitude);
        if levels.len() > 100 {
            levels.remove(0);
        }
    }

    // Process state transitions
    let mut state = state.lock().unwrap();
    match &mut *state {
        MicrophoneState::Disabled => {
            // Do nothing
        }
        MicrophoneState::WaitingForVoiceActivity => {
            if amplitude > 0.01 {
                // Voice activity detected
                info!("Voice activity detected from mic {mic_name}");
                *state = MicrophoneState::VoiceActivated(ActiveMicrophoneState {
                    activity_started: std::time::Instant::now(),
                    last_activity: std::time::Instant::now(),
                    data_so_far: data.to_vec(),
                    sample_rate,
                });
            }
        }
        MicrophoneState::VoiceActivated(active_state) => {
            if amplitude > 0.01 {
                // Continue recording
                active_state.last_activity = std::time::Instant::now();
                active_state.data_so_far.extend_from_slice(data);
            } else {
                // Check for pause
                let elapsed = active_state.last_activity.elapsed();
                if elapsed > std::time::Duration::from_secs(1) {
                    // Pause detected
                    info!("Pause detected for mic {mic_name}, sending data for transcription");
                    let audio_data = active_state.data_so_far.clone();
                    transcription_sender
                        .send(AudioChunk {
                            data: audio_data,
                            sample_rate: active_state.sample_rate,
                            channels: channel_count,
                        })
                        .unwrap();
                    *state = MicrophoneState::WaitingForVoiceActivity;
                } else {
                    // Continue recording
                    active_state.data_so_far.extend_from_slice(data);
                }
            }
        }
        _ => {
            // Handle other states if necessary
        }
    }
}
