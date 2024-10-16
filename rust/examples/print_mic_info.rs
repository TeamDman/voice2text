use cpal::traits::{DeviceTrait, HostTrait};
use itertools::Itertools;

fn main() {
    for device in cpal::default_host().input_devices().unwrap() {
        println!("Device \"{}\"", device.name().unwrap_or_default());
        println!("Supported input configs:");
        for (i, config) in device.supported_input_configs().unwrap().enumerate() {
            println!("  {}: {:?}", i, config);
        }
    }
}
