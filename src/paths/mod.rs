mod app_home;
mod cache;

pub use app_home::*;
pub use cache::*;

pub const APP_HOME_ENV_VAR: &str = "V2T_HOME_DIR";
pub const APP_HOME_DIR_NAME: &str = "voice2text";

pub const APP_CACHE_ENV_VAR: &str = "V2T_CACHE_DIR";
pub const APP_CACHE_DIR_NAME: &str = "voice2text";
