use anyhow::{anyhow, bail, Context, Result};
use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use cpal::{Device, SampleFormat, StreamConfig};
use hound::{self, WavWriter};
use rubato::FftFixedInOut;
use rubato::Resampler;
use std::fs::{self, File};
use std::io::{self, BufWriter, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn get_device() -> Result<Device> {
    // Get the default host for audio devices
    let host = cpal::default_host();

    // Collect all available input devices (microphones)
    let devices: Vec<_> = host
        .input_devices()
        .context("No input devices available")?
        .collect();

    if devices.is_empty() {
        bail!("No input devices available");
    }

    let mut device_index: usize;
    loop {
        // List the available input devices
        println!("Available input devices:");
        for (i, device) in devices.iter().enumerate() {
            println!("{}: {}", i, device.name().unwrap_or("Unknown".to_string()));
        }

        // Prompt the user to select a device
        print!("Please select an input device by number: ");
        io::stdout().flush()?; // Ensure the prompt is displayed

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        device_index = input
            .trim()
            .parse()
            .context("Please enter a valid number")?;

        if device_index >= devices.len() {
            println!("Invalid device index");
            continue;
        } else {
            break;
        }
    }

    let device = devices
        .into_iter()
        .nth(device_index)
        .ok_or_else(|| anyhow!("device somehow not present???"))?;

    Ok(device)
}

fn get_recording_duration() -> Result<Duration> {
    // Prompt the user to enter the recording duration in seconds
    print!("Please enter the recording duration in seconds: ");
    io::stdout().flush()?;

    let mut duration_input = String::new();
    io::stdin().read_line(&mut duration_input)?;
    let duration_secs: u64 = duration_input
        .trim()
        .parse()
        .context("Please enter a valid number")?;
    let duration = Duration::from_secs(duration_secs);
    Ok(duration)
}

fn get_device_stream_config(device: &Device) -> Result<StreamConfig> {
    let device_input_config = device.default_input_config()?;
    let sample_format = device_input_config.sample_format();
    if sample_format != SampleFormat::F32 {
        bail!("Unsupported sample format: {:?}", sample_format);
    }
    let device_stream_config: StreamConfig = device_input_config.into();
    Ok(device_stream_config)
}

fn get_muted_amplitude(device: &Device) -> Result<f64> {
    let device_name = device.name().unwrap_or("Unknown".to_string());
    let device_stream_config = get_device_stream_config(device)?;

    // Check for existing muted microphone sample
    let muted_samples_dir = Path::new("muted_microphone_samples");
    let sanitized_device_name = device_name.replace("/", "_").replace("\\", "_");
    let muted_sample_filename = format!("muted {}.wav", sanitized_device_name);
    let muted_sample_path = muted_samples_dir.join(muted_sample_filename);

    if !muted_sample_path.exists() {
        // Create directory if it doesn't exist
        if !muted_samples_dir.exists() {
            fs::create_dir(&muted_samples_dir)
                .context("Failed to create directory for muted samples")?;
        }

        // Prompt user to mute microphone
        println!(
            "No muted sample found for '{}'. Please mute your microphone and press Enter to record a 5-second muted sample.",
            device_name
        );
        let mut dummy_input = String::new();
        io::stdin().read_line(&mut dummy_input)?;

        fn record_sample(
            device: &cpal::Device,
            config: &cpal::StreamConfig,
            output_path: &Path,
            duration: Duration,
        ) -> Result<()> {
            let spec = hound::WavSpec {
                channels: config.channels,
                sample_rate: config.sample_rate.0,
                bits_per_sample: 32, // f32
                sample_format: hound::SampleFormat::Float,
            };

            let wav_writer = WavWriter::create(output_path, spec)?;
            let writer = Arc::new(Mutex::new(Some(wav_writer))); // Wrap in Option

            let err_fn = |err| eprintln!("An error occurred on the input audio stream: {}", err);

            let writer_clone = writer.clone();
            let stream = device.build_input_stream(
                config,
                move |data: &[f32], _: &_| {
                    write_input_data_f32(data, &writer_clone).unwrap();
                },
                err_fn,
            )?;
            stream.play().context("Failed to start input stream")?;
            thread::sleep(duration);

            // Finalize the WAV file
            let wav_writer = writer.lock().unwrap().take();
            if let Some(writer) = wav_writer {
                writer.finalize()?;
            }
            Ok(())
        }

        // Record 5-second muted sample
        println!("Recording muted sample for 5 seconds...");
        record_sample(
            &device,
            &device_stream_config,
            &muted_sample_path,
            Duration::from_secs(5),
        )?;

        println!(
            "Muted sample recorded and saved to '{}'.",
            muted_sample_path.display()
        );
    }

    // Function to calculate the amplitude of the muted sample
    fn calculate_muted_sample_amplitude(muted_sample_path: &Path) -> Result<f64> {
        let mut reader =
            hound::WavReader::open(muted_sample_path).context("Failed to open muted sample")?;
        let spec = reader.spec();

        let samples: Vec<f64> = match spec.sample_format {
            hound::SampleFormat::Int => {
                reader.samples::<i16>().map(|s| s.unwrap() as f64).collect()
            }
            hound::SampleFormat::Float => {
                reader.samples::<f32>().map(|s| s.unwrap() as f64).collect()
            }
        };

        let sum_squares: f64 = samples.iter().map(|&sample| sample * sample).sum();
        let rms = (sum_squares / samples.len() as f64).sqrt();
        Ok(rms)
    }
    // Load muted sample and calculate its amplitude
    let muted_sample_amplitude = calculate_muted_sample_amplitude(&muted_sample_path)?;
    Ok(muted_sample_amplitude)
}

// Shared state structure
struct SharedState {
    activity_level_history: Vec<f64>,
    is_muted: bool,
}

const SAMPLE_RATE: usize = 16_000;

fn main() -> Result<()> {
    let device = get_device()?;
    let device_name = device.name().unwrap_or("Unknown".to_string());
    println!("Selected device: {}", device_name);

    let recording_duration = get_recording_duration()?;
    println!("Recording for {} seconds...", recording_duration.as_secs());

    let muted_sample_amplitude = get_muted_amplitude(&device)?;
    println!("Muted sample amplitude: {:.6}", muted_sample_amplitude);

    // Create a WAV writer to write the audio data
    let writer = WavWriter::create(
        "output.wav",
        hound::WavSpec {
            channels: 1,                               // Mono
            sample_rate: SAMPLE_RATE as u32,           // 16kHz
            bits_per_sample: 32,                       // 32-bit float
            sample_format: hound::SampleFormat::Float, // f32le
        },
    )?;
    let writer = Arc::new(Mutex::new(Some(writer)));

    // Set up recording parameters
    let device_stream_config = get_device_stream_config(&device)?;

    // Shared state for activity level and mute detection
    let shared_state = Arc::new(Mutex::new(SharedState {
        activity_level_history: Vec::new(),
        is_muted: false,
    }));

    // Define constants
    const INPUT_SAMPLE_RATE: usize = 48000;
    const OUTPUT_SAMPLE_RATE: usize = SAMPLE_RATE;
    const CHANNELS: usize = 1;
    const FRAMES_PER_BUFFER: usize = 441; // https://github.com/HEnquist/rubato/issues/76#issuecomment-1966452981

    // Initialize the resampler
    let resampler = FftFixedInOut::<f32>::new(
        INPUT_SAMPLE_RATE,
        OUTPUT_SAMPLE_RATE,
        FRAMES_PER_BUFFER,
        CHANNELS,
    )
    .context("Failed to create resampler")?;
    let resampler = Arc::new(Mutex::new(resampler));
    let stream = {
        let writer_clone = writer.clone();
        let shared_state_clone = shared_state.clone();
        let resampler_clone = resampler.clone(); // To be defined
        device.build_input_stream(
            &device_stream_config,
            move |data: &[f32], _: &_| {
                // Downmix to mono
                fn downmix_to_mono_f32(stereo_samples: &[f32]) -> Vec<f32> {
                    stereo_samples
                        .chunks(2)
                        .map(|chunk| (chunk[0] + chunk[1]) / 2.0)
                        .collect()
                }
                let mono_data = downmix_to_mono_f32(data);

                // Lock and process resampling
                let mut resampler_guard = resampler_clone.lock().unwrap();
                let resampled_data = resampler_guard
                    .process(&[mono_data], None)
                    .expect("Resampling failed")
                    .into_iter()
                    .flatten()
                    .collect::<Vec<f32>>();
                drop(resampler_guard);

                // Write resampled data
                write_input_data_f32(&resampled_data, &writer_clone).unwrap();
                // Analyze data to detect mute/unmute

                fn calculate_rms_amplitude_f32(data: &[f32]) -> f32 {
                    let sum_squares: f32 = data.iter().map(|&sample| sample * sample).sum();
                    let rms = (sum_squares / data.len() as f32).sqrt();
                    rms
                }
                let amplitude = calculate_rms_amplitude_f32(data);

                // Update shared state
                let mut state = shared_state_clone.lock().unwrap();
                state.activity_level_history.push(amplitude as f64);
                // Keep history for last 5 seconds
                let max_history_length = (SAMPLE_RATE as usize / data.len()) * 5;
                if state.activity_level_history.len() > max_history_length {
                    state.activity_level_history.remove(0);
                }

                // Calculate moving average
                let sum: f64 = state.activity_level_history.iter().sum();
                let moving_average = sum / state.activity_level_history.len() as f64;

                // Determine mute state
                let threshold_margin = 0.001;
                state.is_muted = moving_average <= muted_sample_amplitude + threshold_margin;

                // For debugging: print current moving average and mute state
                // println!(
                //     "Moving Average: {:.6}, Muted: {}",
                //     moving_average, state.is_muted
                // );
            },
            |err| eprintln!("An error occurred on the input audio stream: {}", err),
        )
    }?;

    // Start the input stream
    stream.play().context("Failed to start input stream")?;

    // Record audio for the specified duration and monitor mute state
    let start_time = Instant::now();
    let print_interval = Duration::from_secs(1);
    let mut last_print_time = Instant::now();
    let mut prev_mute_state = None;

    while Instant::now().duration_since(start_time) < recording_duration {
        // Only check every print_interval
        if Instant::now().duration_since(last_print_time) >= print_interval {
            let state = shared_state.lock().unwrap();
            if prev_mute_state != Some(state.is_muted) {
                if state.is_muted {
                    println!("Microphone has been muted");
                } else {
                    println!("Microphone has been unmuted");
                }
                prev_mute_state = Some(state.is_muted);
            }
            last_print_time = Instant::now();
        }
        thread::sleep(Duration::from_millis(100));
    }

    // Stop the stream to finish recording
    drop(stream);

    // Finalize the WAV file
    let wav_writer = writer.lock().unwrap().take(); // Take ownership
    if let Some(writer) = wav_writer {
        writer.finalize()?;
    }

    println!("Recording saved to output.wav");
    Ok(())
}

fn write_input_data_f32(
    input: &[f32],
    writer: &Arc<Mutex<Option<WavWriter<BufWriter<File>>>>>,
) -> Result<()> {
    let mut guard = writer.lock().unwrap();
    if let Some(ref mut writer) = *guard {
        for &sample in input.iter() {
            writer.write_sample(sample)?;
        }
    }
    Ok(())
}
