use std::io;

use clap::Parser;

/// Parse and display metadata from an MKV file
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
    /// path to the MKV file
    path: std::path::PathBuf,

    #[clap(flatten)]
    verbose: clap_verbosity_flag::Verbosity,
}

fn main() -> io::Result<()> {
    let args = Args::parse();

    let matroska =
        matroska::open(&args.path).expect(&format!("Could not open file '{:?}'", &args.path));

    println!("{:?}", &matroska);

    Ok(())
}
