use clap::Parser;
use rand::{rngs::StdRng, Rng, SeedableRng};
use unicode_box_drawing_wfc::{TILE_SET, WFC};

#[derive(Debug, Default, Clone, Copy, Parser)]
struct Args {
    #[arg(long)]
    no_captue: bool,
    #[arg(short, long)]
    seed: Option<u64>,

}

fn main() {
    let mut args = Args::parse();

    args.seed = args.seed.or(StdRng::from_entropy().gen());

    let wfc_builder = WFC::builder().dimensions([80, 40]).tile_set(TILE_SET);

    let wfc_builder = match args.seed {
        Some(seed) => wfc_builder.seed_from_u64(seed),
        None => wfc_builder,
    };

    let mut wfc = wfc_builder.build();

    let solution = wfc.collapse().is_ok();

    if !args.no_captue {
        println!("Seed: {}", args.seed.expect("seed to be always set"));
        println!("Solution: {:?}", solution);
    }
}
