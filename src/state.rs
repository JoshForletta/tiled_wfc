use std::{
    error::Error,
    fmt::Display,
    iter::{once, repeat, Once},
};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateError {
    Collapsed,
    Superimposed,
    NoViableState,
    OutOfTileSetBounds,
}

impl Display for StateError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}",
            match self {
                Self::Collapsed => "state is collapsed",
                Self::Superimposed => "state is superimposed",
                Self::NoViableState => "no viable state",
                Self::OutOfTileSetBounds => "state out of tile set bounds",
            }
        )
    }
}

impl Error for StateError {}

/// [`State`] represents quantum state with either [`Collapsed`] or [`Superimposed`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum State {
    Collapsed(usize),
    Superimposed(Superposition),
}

impl State {
    /// Return `Superimposed` containing states up to `len` noninclusive.
    ///
    /// # Panics
    ///
    /// if `len` exceeds `Superposition::MAX`
    pub fn fill(len: usize) -> Self {
        Self::Superimposed(Superposition::fill(len))
    }

    /// returns `Superimposed` containing states yielded `true` from `states`
    ///
    /// # Panics
    ///
    /// if number of states yielded from `states` exceeds `Superpositon::MAX`
    pub fn from_iter<I>(states: I) -> Self
    where
        I: IntoIterator,
        <I as IntoIterator>::IntoIter: Iterator<Item = bool>,
    {
        Self::Superimposed(Superposition::from_iter(states))
    }

    /// Returns number of states contained within `self`.
    pub fn count(&self) -> usize {
        match self {
            State::Collapsed(_) => 1,
            State::Superimposed(superposition) => superposition.count(),
        }
    }

    /// Returns `true` if `self` is [`Collapsed`]
    pub fn is_collapsed(&self) -> bool {
        match self {
            State::Collapsed(_) => true,
            State::Superimposed(_) => false,
        }
    }

    /// Returns a reference to state if `self` is [`Collapsed`], otherwise
    /// returns [`StateError::Superimposed`].
    pub fn collapsed(&self) -> Result<&usize, StateError> {
        match self {
            State::Collapsed(state_index) => Ok(state_index),
            State::Superimposed(_) => Err(StateError::Superimposed),
        }
    }

    /// Returns a reference to superposition if `self` is [`Superimposed`],
    /// otherwise returns [`StateError::Collapsed`].
    pub fn superimposed(&self) -> Result<&Superposition, StateError> {
        match self {
            State::Collapsed(_) => Err(StateError::Collapsed),
            State::Superimposed(superpoition) => Ok(superpoition),
        }
    }

    /// Returns a mutable reference to state if `self` is [`Collapsed`],
    /// otherwise returns [`StateError::Superimposed`].
    pub fn collapsed_mut(&mut self) -> Result<&mut usize, StateError> {
        match self {
            State::Collapsed(state_index) => Ok(state_index),
            State::Superimposed(_) => Err(StateError::Superimposed),
        }
    }

    /// Returns a mutable reference to superposition if `self` is
    /// [`Superimposed`], otherwise returns [`StateError::Collapsed`].
    pub fn superimposed_mut(&mut self) -> Result<&mut Superposition, StateError> {
        match self {
            State::Collapsed(_) => Err(StateError::Collapsed),
            State::Superimposed(superpoition) => Ok(superpoition),
        }
    }

    // /// Collapses `self` using [`collapser`](super::Collapser) with [`rng`].
    // ///
    // /// # Errors
    // ///
    // /// - if `self` is [`Collapsed`].
    // /// - if `collapser` errors. see [`Collaper`](super::Collapser).
    // pub fn collapse<C, R>(&mut self, collapser: &mut C, rng: R) -> Result<usize, StateError>
    // where
    //     C: Collapser,
    //     R: Rng,
    // {
    //     todo!()
    // }
    //
    // /// Sets `self` to `superposition`.
    // ///
    // /// # Errors
    // ///
    // /// if `self` is [`Superimposed`].
    // pub fn uncollapse(&mut self, superposition: Superposition) -> Result<(), StateError> {
    //     todo!()
    // }
}

impl Default for State {
    /// Return [`Superimposed`] with no viable state.
    fn default() -> Self {
        Self::Superimposed(Superposition::default())
    }
}

impl<'a> IntoIterator for &'a State {
    type Item = usize;
    type IntoIter = StateIter<'a>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            State::Collapsed(state) => StateIter::Collapsed(once(*state)),
            State::Superimposed(superposition) => {
                StateIter::Superimposed(superposition.into_iter())
            }
        }
    }
}

/// An iterator over [`&State`].
///
/// This `struct` is created by the [`into_iter`] method on [`&State`](State).
/// Yields viable states.
pub enum StateIter<'a> {
    Collapsed(Once<usize>),
    Superimposed(SuperpositionIter<'a>),
}

impl<'a> Iterator for StateIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        match self {
            StateIter::Collapsed(iter) => iter.next(),
            StateIter::Superimposed(iter) => iter.next(),
        }
    }
}

/// `Superposition` represents a superimposed state.
#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct Superposition {
    bitmask: u128,
}

impl Superposition {
    pub const MAX: usize = u128::BITS as usize;

    /// Returns `Superposition` containing states up to `len` noninclusive.
    ///
    /// # Panics
    ///
    /// if `len` exceeds `Superposition::MAX`
    pub fn fill(len: usize) -> Self {
        Self::from_iter(repeat(true).take(len))
    }

    /// returns `Superpostition` containing states yielded `true` from `states`
    ///
    /// # Panics
    ///
    /// if number of states yielded from `states` exceeds `Superpositon::MAX`
    pub fn from_iter<I>(states: I) -> Self
    where
        I: IntoIterator,
        <I as IntoIterator>::IntoIter: Iterator<Item = bool>,
    {
        let mut bitmask = 0;

        for (state, include) in states.into_iter().enumerate() {
            if include {
                bitmask |= Superposition::state_to_bitmask(state);
            }
        }

        Self { bitmask }
    }

    /// Returns the number of states contained within `self`.
    pub fn count(&self) -> usize {
        self.into_iter().count()
    }

    /// returns a bitmask containing `state`
    ///
    /// # Panics
    ///
    /// if `state` exceeds `Superpositon::MAX`
    #[inline(always)]
    fn state_to_bitmask(state: usize) -> u128 {
        1u128
            .checked_shl(state as u32)
            .expect("`state` exceeds `Superpositon::MAX`")
    }

    /// return `true` if `state` is contained with self
    ///
    /// # Panics
    ///
    /// if `state` exceeds `Superpositon::MAX`
    #[inline(always)]
    pub fn contains_state(&self, state: usize) -> bool {
        self.bitmask & Superposition::state_to_bitmask(state) != 0
    }

    /// returns `true` if all viable states within `other` are contained within `self`.
    #[inline(always)]
    pub fn contains_superposition(&self, superposition: &Superposition) -> bool {
        superposition.bitmask == superposition.bitmask & self.bitmask
    }

    /// inserts `state` into `self`.
    ///
    /// # Panics
    ///
    /// if `state` exeeds `Superpositon::MAX`
    #[inline(always)]
    pub fn insert_state(&mut self, state: usize) {
        self.bitmask |= Superposition::state_to_bitmask(state);
    }

    /// inserts all viable states in `superposition` into `self`.
    #[inline(always)]
    pub fn insert_superposition(&mut self, other: &Superposition) {
        self.bitmask |= other.bitmask;
    }

    /// removes `state` from `self`.
    #[inline(always)]
    pub fn remove_state(&mut self, state: usize) {
        self.bitmask &= !Superposition::state_to_bitmask(state);
    }

    /// removes all viable states in `superposition` from `self`.
    #[inline(always)]
    pub fn remove_superposition(&mut self, other: &Superposition) {
        self.bitmask &= !other.bitmask;
    }

    /// removes all viable states from `self` that aren't contained in `other`.
    #[inline(always)]
    pub fn constrain(&mut self, other: &Superposition) {
        self.bitmask &= other.bitmask;
    }
}

impl<'a> IntoIterator for &'a Superposition {
    type Item = usize;
    type IntoIter = SuperpositionIter<'a>;

    /// returns an iterator of all states contained within `self`.
    fn into_iter(self) -> Self::IntoIter {
        SuperpositionIter {
            superposition: self,
            next_state: 0,
        }
    }
}

/// An iterator over [`&Superposition`].
///
/// This `struct` is created by the [`into_iter`] method on [`&Superposition`](Superpositon).
/// Yields viable states.
#[derive(Debug, Clone)]
pub struct SuperpositionIter<'a> {
    superposition: &'a Superposition,
    next_state: usize,
}

impl<'a> Iterator for SuperpositionIter<'a> {
    type Item = usize;

    fn next(&mut self) -> Option<Self::Item> {
        for state in self.next_state..Superposition::MAX {
            if self.superposition.contains_state(state) {
                self.next_state = state + 1;
                return Some(state);
            }
        }

        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fill() {
        let s = State::fill(3);

        assert_eq!(
            s,
            State::Superimposed(Superposition::from_iter([true, true, true]))
        );
    }

    #[test]
    #[should_panic]
    fn fill_exceeds_max() {
        State::fill(Superposition::MAX + 1);
    }

    #[test]
    fn from_iter() {
        let s = State::from_iter([true, false, false, true]);

        match s {
            State::Collapsed(_) => unreachable!(),
            State::Superimposed(s) => {
                assert!(s.contains_state(0));
                assert!(!s.contains_state(1));
                assert!(!s.contains_state(2));
                assert!(s.contains_state(3));
            }
        }
    }

    #[test]
    #[should_panic]
    fn from_iter_exceeds_max() {
        use std::iter::repeat;

        State::from_iter(repeat(true).take(Superposition::MAX + 1));
    }

    #[test]
    fn count_collapsed() {
        let s = State::Collapsed(0);

        assert_eq!(s.count(), 1);
    }

    #[test]
    fn count_superimposed() {
        let s = State::from_iter([true, false, false, true]);

        assert_eq!(s.count(), 2);
    }

    #[test]
    fn is_collapsed() {
        let s1 = State::Collapsed(1);
        let s2 = State::Superimposed(Superposition::from_iter([true, false, false]));

        assert!(s1.is_collapsed());
        assert!(!s2.is_collapsed());
    }

    #[test]
    fn collapsed() {
        let s1 = State::Collapsed(0);
        let s2 = State::Superimposed(Superposition::from_iter([true, false, false]));

        assert_eq!(s1.collapsed(), Ok(&0));
        assert_eq!(s2.collapsed(), Err(StateError::Superimposed));
    }

    #[test]
    fn collapsed_mut() {
        let mut s1 = State::Collapsed(0);
        let mut s2 = State::Superimposed(Superposition::from_iter([true, false, false]));

        assert_eq!(s1.collapsed_mut(), Ok(&mut 0));
        assert_eq!(s2.collapsed_mut(), Err(StateError::Superimposed));
    }

    #[test]
    fn superimposed() {
        let s1 = State::Superimposed(Superposition::from_iter([true, false, false]));
        let s2 = State::Collapsed(0);

        assert_eq!(
            s1.superimposed(),
            Ok(&Superposition::from_iter([true, false, false]))
        );
        assert_eq!(s2.superimposed(), Err(StateError::Collapsed));
    }

    #[test]
    fn superimposed_mut() {
        let mut s1 = State::Superimposed(Superposition::from_iter([true, false, false]));
        let mut s2 = State::Collapsed(0);

        assert_eq!(
            s1.superimposed_mut(),
            Ok(&mut Superposition::from_iter([true, false, false]))
        );
        assert_eq!(s2.superimposed_mut(), Err(StateError::Collapsed));
    }

    #[test]
    fn into_iter_collapsed() {
        let s = State::Collapsed(1);
        let mut s_iter = s.into_iter();

        assert_eq!(s_iter.next(), Some(1));
        assert_eq!(s_iter.next(), None);
    }

    #[test]
    fn into_iter_superimposed() {
        let s = State::Superimposed(Superposition::from_iter([false, true, false, true, false]));
        let mut s_iter = s.into_iter();

        assert_eq!(s_iter.next(), Some(1));
        assert_eq!(s_iter.next(), Some(3));
        assert_eq!(s_iter.next(), None)
    }

    mod sueprposition {
        use super::Superposition;

        #[test]
        fn fill() {
            let s = Superposition::fill(3);

            assert_eq!(s, Superposition::from_iter([true, true, true]));
        }

        #[test]
        #[should_panic]
        fn fill_exceeds_max() {
            Superposition::fill(Superposition::MAX + 1);
        }

        #[test]
        fn from_iter() {
            let s = Superposition::from_iter([true, false, false, true]);

            assert!(s.contains_state(0));
            assert!(!s.contains_state(1));
            assert!(!s.contains_state(2));
            assert!(s.contains_state(3));
        }

        #[test]
        #[should_panic]
        fn from_iter_exceeds_max() {
            use std::iter::repeat;

            Superposition::from_iter(repeat(true).take(Superposition::MAX + 1));
        }

        #[test]
        fn count() {
            let s = Superposition::from_iter([true, false, false, true]);

            assert_eq!(s.count(), 2);
        }

        #[test]
        fn count_empty() {
            let s = Superposition::default();

            assert_eq!(s.count(), 0);
        }

        #[test]
        fn count_full() {
            let s = Superposition::fill(Superposition::MAX);

            assert_eq!(s.count(), Superposition::MAX);
        }

        #[test]
        fn contains_state() {
            let s = Superposition::from_iter([true, false, true]);

            assert!(s.contains_state(0));
            assert!(!s.contains_state(1));
            assert!(s.contains_state(2));
        }

        #[test]
        #[should_panic]
        fn contains_state_exceeds_max() {
            let s = Superposition::from_iter([true, false, true]);

            s.contains_state(Superposition::MAX + 1);
        }

        #[test]
        fn contains_superposition() {
            let s1 = Superposition::from_iter([true, true, false, true]);
            let s2 = Superposition::from_iter([true, false, false, true]);

            assert!(s1.contains_superposition(&s2));
            assert!(!s2.contains_superposition(&s1));
        }

        #[test]
        fn insert_state() {
            let mut s = Superposition::default();

            s.insert_state(2);

            assert!(s.contains_state(2));
        }

        #[test]
        #[should_panic]
        fn insert_state_exceeds_max() {
            let mut s = Superposition::default();

            s.insert_state(Superposition::MAX + 1)
        }

        #[test]
        fn insert_superposition() {
            let mut s1 = Superposition::from_iter([false, false, true, false]);
            let s2 = Superposition::from_iter([true, true, false, true]);

            s1.insert_superposition(&s2);

            assert!(s1.contains_superposition(&s2));
            assert!(!s2.contains_superposition(&s1));
        }

        #[test]
        fn remove_state() {
            let mut s = Superposition::from_iter([true, false, true, false]);

            s.remove_state(1);

            assert!(!s.contains_state(1));
        }

        #[test]
        #[should_panic]
        fn remove_state_exceeds_max() {
            let mut s = Superposition::from_iter([true, false, true, false]);

            s.remove_state(Superposition::MAX);
        }

        #[test]
        fn remove_superposition() {
            let mut s1 = Superposition::from_iter([true, false, true, false]);
            let s2 = Superposition::from_iter([true, true, false, true]);

            s1.remove_superposition(&s2);

            assert_eq!(s1, Superposition::from_iter([false, false, true, false]));
        }

        #[test]
        fn constrain() {
            let mut s1 = Superposition::from_iter([false, false, true, true]);
            let s2 = Superposition::from_iter([true, false, false, true]);

            s1.constrain(&s2);

            assert_eq!(s1, Superposition::from_iter([false, false, false, true]));
        }

        #[test]
        fn into_iter() {
            let s = Superposition::from_iter([false, true, false, true, false]);
            let mut s_iter = s.into_iter();

            assert_eq!(s_iter.next(), Some(1));
            assert_eq!(s_iter.next(), Some(3));
            assert_eq!(s_iter.next(), None)
        }
    }
}
