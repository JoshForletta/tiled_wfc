use std::{error::Error, fmt::Display};

use bitvec::vec::BitVec;
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

pub struct State {
    bitmask: BitVec,
}

impl State {
    pub fn fill(bit: bool, len: usize) -> Self {
        Self {
            bitmask: BitVec::repeat(bit, len),
        }
    }

    pub fn with_index(index: usize, len: usize) -> Self {
        let mut bitmask = BitVec::repeat(false, len);

        bitmask.set(index, true);

        Self { bitmask }
    }

    pub fn collapse<C, R>(&self, collapser: &C, rng: &mut R) -> Result<(), ()>
    where
        C: Collapser,
        R: Rng,
    {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill() {
        let state = State::fill(true, 4);

        assert_eq!(state.bitmask[0], true);
        assert_eq!(state.bitmask[1], true);
        assert_eq!(state.bitmask[2], true);
        assert_eq!(state.bitmask[3], true);
    }

    #[test]
    fn with_index() {
        let state = State::with_index(2, 4);

        assert_eq!(state.bitmask[0], false);
        assert_eq!(state.bitmask[1], false);
        assert_eq!(state.bitmask[2], true);
        assert_eq!(state.bitmask[3], false);
    }
}
