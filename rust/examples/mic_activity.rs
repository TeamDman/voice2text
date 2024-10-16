use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use hound;
use std::fs;
use std::io::{self, Write};
use std::path::Path;
use std::sync::{Arc, Mutex};
use std::thread;
use std::time::{Duration, Instant};

fn main() {
    // Get the default host for audio devices
    let host = cpal::default_host();

    // Collect all available input devices (microphones)
    let devices: Vec<_> = host
        .input_devices()
        .expect("No input devices available")
        .collect();

    if devices.is_empty() {
        println!("No input devices available");
        return;
    }

    // List the available input devices
    println!("Available input devices:");
    for (i, device) in devices.iter().enumerate() {
        println!("{}: {}", i, device.name().unwrap_or("Unknown".to_string()));
    }

    // Prompt the user to select a device
    print!("Please select an input device by number: ");
    io::stdout().flush().unwrap(); // Ensure the prompt is displayed

    let mut input = String::new();
    io::stdin().read_line(&mut input).unwrap();
    let device_index: usize = input.trim().parse().expect("Please enter a valid number");

    if device_index >= devices.len() {
        println!("Invalid device index");
        return;
    }

    let device = &devices[device_index];
    let device_name = device.name().unwrap_or("Unknown".to_string());

    // Prompt the user to enter the recording duration in seconds
    print!("Please enter the recording duration in seconds: ");
    io::stdout().flush().unwrap();

    let mut duration_input = String::new();
    io::stdin().read_line(&mut duration_input).unwrap();
    let duration_secs: u64 = duration_input
        .trim()
        .parse()
        .expect("Please enter a valid number");

    // Get the default input configuration for the selected device
    let config = device.default_input_config().unwrap();
    println!("Selected device: {}", device_name);
    println!("Recording for {} seconds...", duration_secs);

    // Set up recording parameters
    let sample_format = config.sample_format();
    let config: cpal::StreamConfig = config.into();

    let bits_per_sample = match sample_format {
        cpal::SampleFormat::I16 => 16,
        cpal::SampleFormat::U16 => 16,
        cpal::SampleFormat::F32 => 32,
        _ => 16, // Default to 16 bits per sample
    };

    let hound_sample_format = match sample_format {
        cpal::SampleFormat::I16 | cpal::SampleFormat::U16 => hound::SampleFormat::Int,
        cpal::SampleFormat::F32 => hound::SampleFormat::Float,
        _ => hound::SampleFormat::Int,
    };

    let spec = hound::WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate.0,
        bits_per_sample,
        sample_format: hound_sample_format,
    };

    // Check for existing muted microphone sample
    let muted_samples_dir = Path::new("muted_microphone_samples");
    let sanitized_device_name = device_name.replace("/", "_").replace("\\", "_");
    let muted_sample_filename = format!("muted {}.wav", sanitized_device_name);
    let muted_sample_path = muted_samples_dir.join(muted_sample_filename);

    if !muted_sample_path.exists() {
        // Create directory if it doesn't exist
        if !muted_samples_dir.exists() {
            fs::create_dir(&muted_samples_dir)
                .expect("Failed to create directory for muted samples");
        }

        // Prompt user to mute microphone
        println!(
            "No muted sample found for '{}'. Please mute your microphone and press Enter to record a 5-second muted sample.",
            device_name
        );
        let mut dummy_input = String::new();
        io::stdin().read_line(&mut dummy_input).unwrap();

        // Record 5-second muted sample
        println!("Recording muted sample for 5 seconds...");

        record_sample(
            device,
            &config,
            sample_format,
            &muted_sample_path,
            Duration::from_secs(5),
        );

        println!(
            "Muted sample recorded and saved to '{}'.",
            muted_sample_path.display()
        );
    }

    // Load muted sample and calculate its amplitude
    let muted_sample_amplitude = calculate_muted_sample_amplitude(&muted_sample_path);

    println!(
        "Muted sample amplitude calculated: {:.6}",
        muted_sample_amplitude
    );

    // Create a WAV writer to write the audio data
    let wav_writer = hound::WavWriter::create("output.wav", spec).unwrap();
    let writer = Arc::new(Mutex::new(Some(wav_writer))); // Wrap in Option

    // Shared state for activity level and mute detection
    let shared_state = Arc::new(Mutex::new(SharedState {
        activity_level_history: Vec::new(),
        is_muted: false,
        sample_rate: config.sample_rate.0,
    }));

    let err_fn = |err| eprintln!("An error occurred on the input audio stream: {}", err);

    // Build the input stream based on the sample format
    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let writer_clone = writer.clone();
            let shared_state_clone = shared_state.clone();
            device.build_input_stream(
                &config,
                move |data: &[f32], _: &_| {
                    write_input_data_f32(data, &writer_clone);

                    // Analyze data to detect mute/unmute
                    let amplitude = calculate_rms_amplitude_f32(data);

                    // Update shared state
                    let mut state = shared_state_clone.lock().unwrap();
                    state.activity_level_history.push(amplitude as f64);
                    // Keep history for last 5 seconds
                    let max_history_length = (state.sample_rate as usize / data.len()) * 5;
                    if state.activity_level_history.len() > max_history_length {
                        state.activity_level_history.remove(0);
                    }

                    // Calculate moving average
                    let sum: f64 = state.activity_level_history.iter().sum();
                    let moving_average = sum / state.activity_level_history.len() as f64;

                    // Determine mute state
                    state.is_muted =
                        moving_average <= muted_sample_amplitude + get_threshold_margin_f32();

                    // For debugging: print current moving average and mute state
                    // println!(
                    //     "Moving Average: {:.6}, Muted: {}",
                    //     moving_average, state.is_muted
                    // );
                },
                err_fn,
            )
        }
        cpal::SampleFormat::I16 => {
            let writer_clone = writer.clone();
            let shared_state_clone = shared_state.clone();
            device.build_input_stream(
                &config,
                move |data: &[i16], _: &_| {
                    write_input_data_i16(data, &writer_clone);

                    // Analyze data to detect mute/unmute
                    let amplitude = calculate_rms_amplitude_i16(data);

                    // Update shared state
                    let mut state = shared_state_clone.lock().unwrap();
                    state.activity_level_history.push(amplitude);
                    // Keep history for last 5 seconds
                    let max_history_length = (state.sample_rate as usize / data.len()) * 5;
                    if state.activity_level_history.len() > max_history_length {
                        state.activity_level_history.remove(0);
                    }

                    // Calculate moving average
                    let sum: f64 = state.activity_level_history.iter().sum();
                    let moving_average = sum / state.activity_level_history.len() as f64;

                    // Determine mute state
                    state.is_muted =
                        moving_average <= muted_sample_amplitude + get_threshold_margin_i16();

                    // For debugging: print current moving average and mute state
                    // println!(
                    //     "Moving Average: {:.6}, Muted: {}",
                    //     moving_average, state.is_muted
                    // );
                },
                err_fn,
            )
        }
        cpal::SampleFormat::U16 => {
            let writer_clone = writer.clone();
            let shared_state_clone = shared_state.clone();
            device.build_input_stream(
                &config,
                move |data: &[u16], _: &_| {
                    write_input_data_u16(data, &writer_clone);

                    // Analyze data to detect mute/unmute
                    let amplitude = calculate_rms_amplitude_u16(data);

                    // Update shared state
                    let mut state = shared_state_clone.lock().unwrap();
                    state.activity_level_history.push(amplitude);
                    // Keep history for last 5 seconds
                    let max_history_length = (state.sample_rate as usize / data.len()) * 5;
                    if state.activity_level_history.len() > max_history_length {
                        state.activity_level_history.remove(0);
                    }

                    // Calculate moving average
                    let sum: f64 = state.activity_level_history.iter().sum();
                    let moving_average = sum / state.activity_level_history.len() as f64;

                    // Determine mute state
                    state.is_muted =
                        moving_average <= muted_sample_amplitude + get_threshold_margin_i16();

                    // For debugging: print current moving average and mute state
                    // println!(
                    //     "Moving Average: {:.6}, Muted: {}",
                    //     moving_average, state.is_muted
                    // );
                },
                err_fn,
            )
        }
        _ => panic!("Unsupported sample format"),
    }
    .expect("Failed to build input stream");

    // Start the input stream
    stream.play().expect("Failed to start input stream");

    // Record audio for the specified duration and monitor mute state
    let start_time = Instant::now();
    let print_interval = Duration::from_secs(1);
    let mut last_print_time = Instant::now();
    let mut prev_mute_state = None;

    while Instant::now().duration_since(start_time) < Duration::from_secs(duration_secs) {
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
        writer.finalize().unwrap();
    }

    println!("Recording saved to output.wav");
}

// Function to record a sample for a specified duration
fn record_sample(
    device: &cpal::Device,
    config: &cpal::StreamConfig,
    sample_format: cpal::SampleFormat,
    output_path: &Path,
    duration: Duration,
) {
    let spec = hound::WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate.0,
        bits_per_sample: match sample_format {
            cpal::SampleFormat::I16 => 16,
            cpal::SampleFormat::U16 => 16,
            cpal::SampleFormat::F32 => 32,
            _ => 16,
        },
        sample_format: match sample_format {
            cpal::SampleFormat::I16 | cpal::SampleFormat::U16 => hound::SampleFormat::Int,
            cpal::SampleFormat::F32 => hound::SampleFormat::Float,
            _ => hound::SampleFormat::Int,
        },
    };

    let wav_writer = hound::WavWriter::create(output_path, spec).unwrap();
    let writer = Arc::new(Mutex::new(Some(wav_writer))); // Wrap in Option

    let err_fn = |err| eprintln!("An error occurred on the input audio stream: {}", err);

    let stream = match sample_format {
        cpal::SampleFormat::F32 => {
            let writer_clone = writer.clone();
            device.build_input_stream(
                config,
                move |data: &[f32], _: &_| {
                    write_input_data_f32(data, &writer_clone);
                },
                err_fn,
            )
        }
        cpal::SampleFormat::I16 => {
            let writer_clone = writer.clone();
            device.build_input_stream(
                config,
                move |data: &[i16], _: &_| {
                    write_input_data_i16(data, &writer_clone);
                },
                err_fn,
            )
        }
        cpal::SampleFormat::U16 => {
            let writer_clone = writer.clone();
            device.build_input_stream(
                config,
                move |data: &[u16], _: &_| {
                    write_input_data_u16(data, &writer_clone);
                },
                err_fn,
            )
        }
        _ => panic!("Unsupported sample format"),
    }
    .expect("Failed to build input stream");

    stream.play().expect("Failed to start input stream");

    thread::sleep(duration);

    drop(stream);

    // Finalize the WAV file
    let wav_writer = writer.lock().unwrap().take();
    if let Some(writer) = wav_writer {
        writer.finalize().unwrap();
    }
}

// Function to calculate the amplitude of the muted sample
fn calculate_muted_sample_amplitude(muted_sample_path: &Path) -> f64 {
    let mut reader =
        hound::WavReader::open(muted_sample_path).expect("Failed to open muted sample");
    let spec = reader.spec();

    let samples: Vec<f64> = match spec.sample_format {
        hound::SampleFormat::Int => reader.samples::<i16>().map(|s| s.unwrap() as f64).collect(),
        hound::SampleFormat::Float => reader.samples::<f32>().map(|s| s.unwrap() as f64).collect(),
    };

    let sum_squares: f64 = samples.iter().map(|&sample| sample * sample).sum();
    let rms = (sum_squares / samples.len() as f64).sqrt();
    rms
}

// Shared state structure
struct SharedState {
    activity_level_history: Vec<f64>,
    is_muted: bool,
    sample_rate: u32,
}

// Function to write f32 samples to the WAV file
fn write_input_data_f32(
    input: &[f32],
    writer: &Arc<Mutex<Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>>>,
) {
    let mut guard = writer.lock().unwrap();
    if let Some(ref mut writer) = *guard {
        for &sample in input.iter() {
            writer.write_sample(sample).unwrap();
        }
    }
}

// Function to write i16 samples to the WAV file
fn write_input_data_i16(
    input: &[i16],
    writer: &Arc<Mutex<Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>>>,
) {
    let mut guard = writer.lock().unwrap();
    if let Some(ref mut writer) = *guard {
        for &sample in input.iter() {
            writer.write_sample(sample).unwrap();
        }
    }
}

// Function to write u16 samples to the WAV file (converted to i16)
fn write_input_data_u16(
    input: &[u16],
    writer: &Arc<Mutex<Option<hound::WavWriter<std::io::BufWriter<std::fs::File>>>>>,
) {
    let mut guard = writer.lock().unwrap();
    if let Some(ref mut writer) = *guard {
        for &sample in input.iter() {
            // Convert u16 to i16 by subtracting 32768 using i32 to avoid overflow
            let sample_i16 = (sample as i32 - 32768) as i16;
            writer.write_sample(sample_i16).unwrap();
        }
    }
}

// Function to calculate RMS amplitude for f32 samples
fn calculate_rms_amplitude_f32(data: &[f32]) -> f32 {
    let sum_squares: f32 = data.iter().map(|&sample| sample * sample).sum();
    let rms = (sum_squares / data.len() as f32).sqrt();
    rms
}

// Function to calculate RMS amplitude for i16 samples
fn calculate_rms_amplitude_i16(data: &[i16]) -> f64 {
    let sum_squares: f64 = data.iter().map(|&sample| (sample as f64).powi(2)).sum();
    let rms = (sum_squares / data.len() as f64).sqrt();
    rms
}

// Function to calculate RMS amplitude for u16 samples
fn calculate_rms_amplitude_u16(data: &[u16]) -> f64 {
    let sum_squares: f64 = data
        .iter()
        .map(|&sample| {
            let sample_i32 = sample as i32 - 32768;
            (sample_i32 as f64).powi(2)
        })
        .sum();
    let rms = (sum_squares / data.len() as f64).sqrt();
    rms
}

// Functions to get threshold margins
fn get_threshold_margin_f32() -> f64 {
    // Adjust this margin based on experimentation
    0.001
}

fn get_threshold_margin_i16() -> f64 {
    // Adjust this margin based on experimentation
    50.0
}
