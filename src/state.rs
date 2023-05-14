use std::{
    error::Error,
    fmt::Display,
    ops::{BitAnd, BitAndAssign, BitOr, BitOrAssign},
};

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

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct State {
    collapsed: bool,
    bitmask: BitVec,
}

impl State {
    pub fn new_collapsed(index: usize, len: usize) -> Self {
        let mut bitmask = BitVec::repeat(false, len);

        bitmask.set(index, true);

        Self {
            collapsed: true,
            bitmask,
        }
    }

    pub fn fill(bit: bool, len: usize) -> Self {
        Self {
            collapsed: false,
            bitmask: BitVec::repeat(bit, len),
        }
    }

    pub fn with_index(index: usize, len: usize) -> Self {
        let mut bitmask = BitVec::repeat(false, len);

        bitmask.set(index, true);

        Self {
            collapsed: false,
            bitmask,
        }
    }

    pub fn with_indexes<const N: usize>(indexes: [usize; N], len: usize) -> Self {
        let mut bitmask = BitVec::repeat(false, len);

        for index in indexes.into_iter() {
            bitmask.set(index, true);
        }

        Self {
            collapsed: false,
            bitmask,
        }
    }

    #[inline(always)]
    pub fn is_collapsed(&self) -> bool {
        self.collapsed
    }

    #[inline(always)]
    pub fn count(&self) -> usize {
        self.bitmask.count_ones()
    }

    #[inline(always)]
    pub fn state_index(&self) -> Option<usize> {
        self.is_collapsed().then_some(self.bitmask.first_one()?)
    }

    #[inline(always)]
    pub fn state_indexes(&'_ self) -> impl Iterator<Item = usize> + '_ {
        self.bitmask.iter_ones()
    }

    #[inline(always)]
    pub fn contains(&self, other: &Self) -> bool {
        other.bitmask.clone() & &self.bitmask == other.bitmask
    }

    #[inline(always)]
    pub fn set(&mut self, index: usize, value: bool) {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask.set(index, value);
    }

    #[inline(always)]
    pub fn constrain(&mut self, other: &Self) -> bool {
        assert!(!self.collapsed, "NOT COLLAPSED");
        let changed = !other.contains(self);

        self.bitmask &= &other.bitmask;

        changed
    }

    pub fn collapse<C, R>(&mut self, collapser: &C, rng: &mut R) -> Result<usize, StateError>
    where
        C: Collapser,
        R: Rng,
    {
        let index = collapser.collapse(self.bitmask.iter_ones(), rng)?;

        self.bitmask.set_elements(0);
        self.set(index, true);

        self.collapsed = true;

        Ok(index)
    }
}

impl BitAnd<State> for State {
    type Output = State;

    #[inline(always)]
    fn bitand(mut self, rhs: State) -> Self::Output {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask &= rhs.bitmask;
        self
    }
}

impl BitAnd<&State> for State {
    type Output = State;

    #[inline(always)]
    fn bitand(mut self, rhs: &State) -> Self::Output {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask &= &rhs.bitmask;
        self
    }
}

impl BitAndAssign<State> for State {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: State) {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask &= rhs.bitmask;
    }
}

impl BitAndAssign<&State> for State {
    #[inline(always)]
    fn bitand_assign(&mut self, rhs: &State) {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask &= &rhs.bitmask;
    }
}

impl BitOr<State> for State {
    type Output = State;

    #[inline(always)]
    fn bitor(mut self, rhs: State) -> Self::Output {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask |= rhs.bitmask;
        self
    }
}

impl BitOr<&State> for State {
    type Output = State;

    #[inline(always)]
    fn bitor(mut self, rhs: &State) -> Self::Output {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask |= &rhs.bitmask;
        self
    }
}

impl BitOrAssign<State> for State {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: State) {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask |= rhs.bitmask;
    }
}

impl BitOrAssign<&State> for State {
    #[inline(always)]
    fn bitor_assign(&mut self, rhs: &State) {
        assert!(!self.collapsed, "NOT COLLAPSED");
        self.bitmask |= &rhs.bitmask;
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use crate::UnweightedCollapser;

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

    #[test]
    fn with_indexes() {
        let state = State::with_indexes([1, 3], 4);

        assert_eq!(state.bitmask[0], false);
        assert_eq!(state.bitmask[1], true);
        assert_eq!(state.bitmask[2], false);
        assert_eq!(state.bitmask[3], true);
    }

    #[test]
    fn is_collapsed_collapsed() {
        let state = State {
            collapsed: true,
            bitmask: BitVec::from_iter([false, true, false, false]),
        };

        assert!(state.is_collapsed());
    }

    #[test]
    fn is_collapsed_no_state() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, false, false, false]),
        };

        assert!(!state.is_collapsed());
    }

    #[test]
    fn is_collapsed_superimposed() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        assert!(!state.is_collapsed());
    }

    #[test]
    fn state_index_collapsed() {
        let state = State {
            collapsed: true,
            bitmask: BitVec::from_iter([false, true, false, false]),
        };

        assert_eq!(state.state_index(), Some(1));
    }

    #[test]
    fn state_index_no_state() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, false, false, false]),
        };

        assert_eq!(state.state_index(), None);
    }

    #[test]
    fn state_index_superimposed() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        assert_eq!(state.state_index(), None);
    }

    #[test]
    fn state_indexes() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        let mut iter = state.state_indexes();

        assert_eq!(iter.next(), Some(0));
        assert_eq!(iter.next(), Some(1));
        assert_eq!(iter.next(), Some(3));
        assert_eq!(iter.next(), None);
    }

    #[test]
    fn contains() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        assert!(state.contains(&State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false])
        }))
    }

    #[test]
    fn set() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::repeat(false, 4),
        };

        state.set(2, true);

        assert_eq!(state.bitmask[2], true);
    }

    #[test]
    fn constrain() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, true, false]),
        };

        let other = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let changed = state.constrain(&other);

        assert_eq!(state, output);
        assert!(changed);
    }

    #[test]
    fn constrain_unchanged() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, true, false]),
        };

        let other = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, true, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, true, false]),
        };

        let changed = state.constrain(&other);

        assert_eq!(state, output);
        assert!(!changed);
    }

    #[test]
    fn collapse() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        let collapser = UnweightedCollapser;

        let mut rng = StepRng::new(0, 0);

        state.collapse(&collapser, &mut rng).expect("valid state");

        assert_eq!(
            state,
            State {
                collapsed: true,
                bitmask: BitVec::from_iter([true, false, false, false])
            }
        );
    }

    #[test]
    fn bitand() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, false]),
        };

        assert_eq!(state & rhs, output);
    }

    #[test]
    fn bitand_ref() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, false]),
        };

        assert_eq!(state & &rhs, output);
    }

    #[test]
    fn bitand_assign() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, false]),
        };

        state &= rhs;

        assert_eq!(state, output);
    }

    #[test]
    fn bitand_assign_ref() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, false]),
        };

        state &= &rhs;

        assert_eq!(state, output);
    }

    #[test]
    fn bitor() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            bitmask: BitVec::from_iter([true, true, false, false]),
            collapsed: false,
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        assert_eq!(state | rhs, output);
    }

    #[test]
    fn bitor_ref() {
        let state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        assert_eq!(state | &rhs, output);
    }

    #[test]
    fn bitor_assign() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        state |= rhs;

        assert_eq!(state, output);
    }

    #[test]
    fn bitor_assign_ref() {
        let mut state = State {
            collapsed: false,
            bitmask: BitVec::from_iter([false, true, false, true]),
        };

        let rhs = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, false]),
        };

        let output = State {
            collapsed: false,
            bitmask: BitVec::from_iter([true, true, false, true]),
        };

        state |= &rhs;

        assert_eq!(state, output);
    }
}
