// TODO: runtime decided weighted/unweighted collapser

use rand::{distributions::WeightedIndex, prelude::Distribution, seq::IteratorRandom, Rng};

use crate::{State, StateError, Weighted};

pub trait Collapser {
    /// Collapses `state`.
    fn collapse(&mut self, state: &mut State) -> Result<(), StateError>;
}

#[derive(Debug, Default, Clone, PartialEq)]
pub struct UnweightedCollapser<R> {
    rng: R,
}

impl<R> UnweightedCollapser<R>
where
    R: Rng,
{
    /// Returns an [`UnweightedCollapser`] with `rng`.
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R> Collapser for UnweightedCollapser<R>
where
    R: Rng,
{
    /// Randomly collapses `state`.
    ///
    /// # Errors
    ///
    /// if `state` contains no viable state.
    fn collapse(&mut self, state: &mut State) -> Result<(), StateError> {
        let state_index = state
            .into_iter()
            .choose(&mut self.rng)
            .ok_or(StateError::NoViableState)?;

        *state = State::Collapsed(state_index);

        Ok(())
    }
}

#[derive(Debug, Clone)]
pub struct WeightedCollapser<R> {
    weights: Vec<u32>,
    rng: R,
}

impl<R> WeightedCollapser<R> {
    /// Returns a [`WeightedCollapser`] with `rng` and a weighted distribution
    /// of tiles from `tile_set`.
    pub fn with_tile_set<T>(rng: R, tile_set: &[T]) -> Self
    where
        T: Weighted,
    {
        Self {
            weights: tile_set.into_iter().map(|tile| tile.weight()).collect(),
            rng,
        }
    }
}

impl<R> Collapser for WeightedCollapser<R>
where
    R: Rng,
{
    /// Randomly collapses `state` with a weighted distribution.
    ///
    /// # Errors
    ///
    /// - if `state` contains no viable state.
    /// - if `state` contains a state out side of tile set bounds.
    fn collapse(&mut self, state: &mut State) -> Result<(), StateError> {
        let states: Vec<usize> = state.into_iter().collect();
        let mut weights = Vec::with_capacity(states.len());

        for index in &states {
            weights.push(
                self.weights
                    .get(*index)
                    .ok_or(StateError::OutOfTileSetBounds)?,
            );
        }

        let distribution = WeightedIndex::new(weights).map_err(|_| StateError::NoViableState)?;

        let state_index = states
            .get(distribution.sample(&mut self.rng))
            .map(|index| *index)
            .ok_or(StateError::NoViableState)?;

        *state = State::Collapsed(state_index);

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use crate::state::Superposition;

    use super::*;

    const TILE_SET: &[TestWeightedTile] = &[
        TestWeightedTile(0),
        TestWeightedTile(3),
        TestWeightedTile(7),
    ];

    struct TestWeightedTile(u32);

    impl Weighted for TestWeightedTile {
        fn weight(&self) -> u32 {
            self.0
        }
    }

    fn assert_collapse<C>(collapser: &mut C, state: &State)
    where
        C: Collapser,
    {
        let mut collapsed_state = state.clone();

        assert!(collapser.collapse(&mut collapsed_state).is_ok());

        let collapsed_state_index = collapsed_state.collapsed().expect("state is collapsed");

        match state {
            State::Collapsed(index) => assert_eq!(collapsed_state_index, index),
            State::Superimposed(superposition) => {
                assert!(superposition.contains_state(*collapsed_state_index))
            }
        }
    }

    fn assert_collapse_no_viable_state<C>(collapser: &mut C)
    where
        C: Collapser,
    {
        let mut state = State::Superimposed(Superposition::fill(0));

        assert_eq!(
            collapser.collapse(&mut state),
            Err(StateError::NoViableState)
        );
    }

    #[test]
    fn unweighted_collapser() {
        let mut collapser = UnweightedCollapser::new(StepRng::new(0, 0));

        let state = State::Superimposed(Superposition::fill(3));

        assert_collapse(&mut collapser, &state);
    }

    #[test]
    fn unweighted_collapser_no_viable_state() {
        let mut collapser = UnweightedCollapser::new(StepRng::new(0, 0));

        assert_collapse_no_viable_state(&mut collapser);
    }

    #[test]
    fn weighted_collapser() {
        let mut collapser = WeightedCollapser::with_tile_set(StepRng::new(0, 0), TILE_SET);

        let state = State::Superimposed(Superposition::fill(3));

        assert_collapse(&mut collapser, &state);
    }

    #[test]
    fn weighted_collapser_no_viable_state() {
        let mut collapser = WeightedCollapser::with_tile_set(StepRng::new(0, 0), TILE_SET);

        assert_collapse_no_viable_state(&mut collapser);
    }

    #[test]
    fn weighted_collapser_state_out_of_tile_set_bounds() {
        let mut collapser =
            WeightedCollapser::with_tile_set::<TestWeightedTile>(StepRng::new(0, 0), &[]);

        let mut state = State::Superimposed(Superposition::fill(3));

        assert_eq!(
            collapser.collapse(&mut state),
            Err(StateError::OutOfTileSetBounds)
        );
    }
}
