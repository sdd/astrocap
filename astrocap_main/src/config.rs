use clap::Parser;
use std::path::PathBuf;

#[derive(Parser, Debug)]
#[command(name = "astrocap_main")]
#[command(about = "Astrocap astronomical video processing pipeline")]
#[command(version)]
pub struct Config {
    /// Path to the configuration TOML file
    #[arg(short, long, default_value = "astrocap.toml")]
    pub config: PathBuf,
}

impl Config {
    pub fn parse_args() -> Self {
        Self::parse()
    }
}
