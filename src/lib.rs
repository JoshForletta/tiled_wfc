pub use nd_matrix;
pub use nd_matrix::*;

pub mod collapser;
pub mod state;
pub mod tile;
pub mod validation;
pub mod wfc;

#[cfg(test)]
mod test_utils;

pub use collapser::{Collapser, UnweightedCollapser, WeightedCollapser};
pub use state::{State, StateError};
pub use tile::{Tile, Weighted};
pub use wfc::WFC;
