// microphone.rs
use crate::ui::AppState;
use anyhow::Context;
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{ChannelCount, SampleFormat, SampleRate, Stream};
use rubato::Resampler;
use tokio::sync::mpsc::UnboundedSender;
use tracing::{debug, error, info};

pub const SAMPLE_RATE: SampleRate = SampleRate(16_000);

pub struct Microphone {
    pub name: String,
    pub state: MicrophoneState,
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

pub fn hook_microphones(state: &mut AppState) -> anyhow::Result<()> {
    let host = cpal::default_host();
    let devices = host.input_devices().unwrap();
    for (i, device) in devices.enumerate() {
        let name = device.name().unwrap_or_else(|_| format!("Unknown-{i}"));

        let enabled = state
            .config
            .microphones
            .get(&name)
            .map(|config| config.enabled)
            .unwrap_or(true);

        let microphone = Microphone {
            name: name.clone(),
            state: if enabled {
                MicrophoneState::WaitingForVoiceActivity
            } else {
                MicrophoneState::Disabled
            },
            stream: None,
        };
        if enabled {
            info!("Hooking microphone {}", name);
            hook_microphone(state, microphone)?;
        } else {
            info!("Skipping microphone {}", name);
        }
    }
    Ok(())
}

pub struct AudioChunk {
    pub mic_name: String,
    pub channels: ChannelCount,
    pub sample_rate: SampleRate,
    pub data: Vec<f32>,
}

impl AudioChunk {
    pub fn downmix(mut self) -> Self {
        // convert to mono
        if self.channels > 1 {
            debug!("Downmixing audio from {} channels to 1", self.channels);
            self.data = self
                .data
                .chunks(self.channels as usize)
                .map(|chunk| chunk.iter().sum::<f32>() / self.channels as f32)
                .collect();
            self.channels = 1;
        }
        // resample to 16khz
        if self.sample_rate != SAMPLE_RATE {
            debug!(
                "Resampling {} audio samples from {} to {}",
                self.data.len(),
                self.sample_rate.0,
                SAMPLE_RATE.0
            );
            let chunk_size = 441; // frames per buffer
            let mut resampler = rubato::FftFixedInOut::<f32>::new(
                self.sample_rate.0 as usize, // 48,000
                SAMPLE_RATE.0 as usize,      // 16,000
                chunk_size,
                self.channels as usize,
            )
            .expect("Failed to create resampler");

            let mut resampled_data = Vec::new();
            for chunk in self.data.chunks(chunk_size) {
                // If the last chunk is smaller than chunk_size, pad it with zeros
                let mut input_chunk = chunk.to_vec();
                if input_chunk.len() < chunk_size {
                    input_chunk.resize(chunk_size, 0.0);
                }

                let output = resampler
                    .process(&[input_chunk], None)
                    .expect("Resampling failed");

                for channel_data in output {
                    resampled_data.extend(channel_data);
                }
            }

            let before = self.data.len();
            self.data = resampled_data;
            let after = self.data.len();
            let ratio = self.sample_rate.0 as f32 / SAMPLE_RATE.0 as f32;
            debug!(
                "Resampling complete: {} samples -> {} samples (ratio: {})",
                before, after, ratio
            );

            let observed_ratio = before as f32 / after as f32;
            if (observed_ratio - ratio).abs() > 1.0 {
                error!(
                    "Resampling failed: {} samples -> {} samples, expected ratio {} observed ratio {}",
                    before, after, ratio, observed_ratio
                );
            }

            self.sample_rate = SAMPLE_RATE;
        }
        self
    }
    pub fn to_byte_slice<'a>(&self) -> &'a [u8] {
        unsafe {
            std::slice::from_raw_parts(
                self.data.as_ptr() as *const u8,
                self.data.len() * std::mem::size_of::<f32>(),
            )
        }
    }
}

pub fn hook_microphone(app_state: &mut AppState, mut mic: Microphone) -> anyhow::Result<()> {
    let device = cpal::default_host()
        .input_devices()?
        .find(|d| d.name().unwrap_or_default() == mic.name)
        .context("Microphone not found")?;
    let config = device.default_input_config().unwrap();
    let sample_rate = config.sample_rate();
    let channels = config.channels();
    let sample_format = config.sample_format();
    let config = config.into();
    let err_fn = |err| error!("An error occurred on the input stream: {}", err);
    let mic_name = mic.name.clone();
    info!("Starting stream for mic {mic_name} with format {sample_format:?}");
    let chunk_sender = app_state.raw_audio_sender.clone();
    let stream = match sample_format {
        SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _| {
                chunk_sender
                    .send(AudioChunk {
                        mic_name: mic_name.to_string(),
                        channels,
                        sample_rate,
                        data: data.to_vec(),
                    })
                    .expect("Failed to send audio data");
            },
            err_fn,
        ),
        SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _| {
                let data_f32: Vec<f32> = data.iter().map(|&s| s as f32 / 32768.0).collect();
                chunk_sender
                    .send(AudioChunk {
                        mic_name: mic_name.to_string(),
                        channels,
                        sample_rate,
                        data: data_f32,
                    })
                    .expect("Failed to send audio data");
            },
            err_fn,
        ),
        SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _| {
                let data_f32: Vec<f32> = data.iter().map(|&s| s as f32 / 65536.0 - 0.5).collect();
                chunk_sender
                    .send(AudioChunk {
                        mic_name: mic_name.to_string(),
                        channels,
                        sample_rate,
                        data: data_f32,
                    })
                    .expect("Failed to send audio data");
            },
            err_fn,
        ),
    }
    .context("Failed to build input stream")?;

    stream.play().context("Failed to start input stream")?;
    mic.stream = Some(stream);
    app_state.add_microphone(mic);
    Ok(())
}

pub fn process_raw_audio(
    chunk: AudioChunk,
    state: &mut MicrophoneState,
    batch_audio_sender: &UnboundedSender<AudioChunk>,
) {
    let amplitude = chunk.data.iter().map(|&s| s.abs()).sum::<f32>() / chunk.data.len() as f32;
    // Process state transitions
    match state {
        MicrophoneState::Disabled => {
            // Do nothing
        }
        MicrophoneState::WaitingForVoiceActivity => {
            if amplitude > 0.01 {
                // Voice activity detected
                info!("Voice activity detected from mic {}", chunk.mic_name);
                *state = MicrophoneState::VoiceActivated(ActiveMicrophoneState {
                    activity_started: std::time::Instant::now(),
                    last_activity: std::time::Instant::now(),
                    data_so_far: chunk.data,
                    sample_rate: chunk.sample_rate,
                });
            }
        }
        MicrophoneState::VoiceActivated(active_state) => {
            if amplitude > 0.01 {
                // Continue recording
                active_state.last_activity = std::time::Instant::now();
                active_state.data_so_far.extend_from_slice(&chunk.data);
            } else {
                // Check for pause
                let elapsed = active_state.last_activity.elapsed();
                if elapsed > std::time::Duration::from_secs(1) {
                    // Pause detected
                    info!(
                        "Pause detected for mic {}, sending data for transcription",
                        chunk.mic_name
                    );
                    let audio_data = std::mem::take(&mut active_state.data_so_far);
                    if let Err(e) = batch_audio_sender.send(AudioChunk {
                        mic_name: chunk.mic_name,
                        data: audio_data,
                        sample_rate: active_state.sample_rate,
                        channels: chunk.channels,
                    }) {
                        error!("Failed to send audio data for transcription: {:?}", e);
                        panic!("Failed to send audio data for transcription");
                    }
                    *state = MicrophoneState::WaitingForVoiceActivity;
                } else {
                    // Continue recording
                    active_state.data_so_far.extend_from_slice(&chunk.data);
                }
            }
        }
        MicrophoneState::WaitingForPushToTalk => todo!(),
        MicrophoneState::PushToTalkActivated(_active_microphone_state) => todo!(),
    }
}
