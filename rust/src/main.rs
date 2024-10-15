use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};
use std::io::{self, Write};
use std::sync::{Arc, Mutex};
use std::time::Duration;
use hound;
use std::thread;

fn main() {
    // Get the default host for audio devices
    let host = cpal::default_host();

    // Collect all available input devices (microphones)
    let devices: Vec<_> = host.input_devices().expect("No input devices available").collect();

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

    // Get the default input configuration for the selected device
    let config = device.default_input_config().unwrap();
    println!(
        "Selected device: {}",
        device.name().unwrap_or("Unknown".to_string())
    );
    println!("Recording for 10 seconds...");

    // Set up recording parameters
    let sample_format = config.sample_format();
    let config: cpal::StreamConfig = config.into();

    let bits_per_sample = match sample_format {
        cpal::SampleFormat::I16 => 16,
        cpal::SampleFormat::U16 => 16,
        cpal::SampleFormat::F32 => 32,
    };

    let hound_sample_format = match sample_format {
        cpal::SampleFormat::I16 | cpal::SampleFormat::U16 => hound::SampleFormat::Int,
        cpal::SampleFormat::F32 => hound::SampleFormat::Float,
    };

    let spec = hound::WavSpec {
        channels: config.channels,
        sample_rate: config.sample_rate.0,
        bits_per_sample,
        sample_format: hound_sample_format,
    };

    // Create a WAV writer to write the audio data
    let wav_writer = hound::WavWriter::create("output.wav", spec).unwrap();
    let writer = Arc::new(Mutex::new(Some(wav_writer))); // Wrap in Option

    let err_fn = |err| eprintln!("An error occurred on the input audio stream: {}", err);

    // Clone the writer to move into the stream closure
    let writer_clone = writer.clone();

    // Build the input stream based on the sample format
    let stream = match sample_format {
        cpal::SampleFormat::F32 => device.build_input_stream(
            &config,
            move |data: &[f32], _: &_| write_input_data_f32(data, &writer_clone),
            err_fn,
        ),
        cpal::SampleFormat::I16 => device.build_input_stream(
            &config,
            move |data: &[i16], _: &_| write_input_data_i16(data, &writer_clone),
            err_fn,
        ),
        cpal::SampleFormat::U16 => device.build_input_stream(
            &config,
            move |data: &[u16], _: &_| write_input_data_u16(data, &writer_clone),
            err_fn,
        ),
        _ => panic!("Unsupported sample format"),
    }
    .expect("Failed to build input stream");

    // Start the input stream
    stream.play().expect("Failed to start input stream");

    // Record audio for 10 seconds
    thread::sleep(Duration::from_secs(10));

    // Stop the stream to finish recording
    drop(stream);

    // Finalize the WAV file
    let wav_writer = writer.lock().unwrap().take(); // Take ownership
    if let Some(writer) = wav_writer {
        writer.finalize().unwrap();
    }

    println!("Recording saved to output.wav");
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
