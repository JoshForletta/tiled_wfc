use std::{error::Error, fmt::Display};

use rand::Rng;

use crate::Collapser;

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StateError {
    NoViableState,
    StateIndexOutOfBounds,
}

impl Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::NoViableState => write!(f, "superimposed state contains no viable state"),
            Self::StateIndexOutOfBounds => write!(
                f,
                "superimposed state contains state outside of tile set bounds"
            ),
        }
    }
}

impl Error for StateError {}

pub struct State;
impl State {
    pub fn collapse<C, R>(&self, collapser: &C, rng: &mut R) -> Result<(), ()>
    where
        C: Collapser,
        R: Rng,
    {
        todo!()
    }
}
