// TODO: runtime decided weighted/unweighted collapser
use rand::{distributions::WeightedIndex, prelude::Distribution, seq::IteratorRandom, Rng};

use crate::{StateError, Weighted};

pub trait Collapser {
    fn collapse<I, R>(&self, states: I, rng: &mut R) -> Result<usize, StateError>
    where
        I: IntoIterator<Item = usize>,
        R: Rng;
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct UnweightedCollapser;

impl Collapser for UnweightedCollapser {
    fn collapse<I, R>(&self, states: I, rng: &mut R) -> Result<usize, StateError>
    where
        I: IntoIterator<Item = usize>,
        R: Rng,
    {
        states
            .into_iter()
            .choose(rng)
            .ok_or(StateError::NoViableState)
    }
}

#[derive(Debug, Clone)]
pub struct WeightedCollapser {
    weights: Vec<u32>,
}

impl<I> From<I> for WeightedCollapser
where
    I: IntoIterator,
    <I as IntoIterator>::Item: Weighted,
{
    fn from(tile_set: I) -> Self {
        Self {
            weights: tile_set.into_iter().map(|tile| tile.weight()).collect(),
        }
    }
}

impl Collapser for WeightedCollapser {
    fn collapse<I, R>(&self, states: I, rng: &mut R) -> Result<usize, StateError>
    where
        I: IntoIterator<Item = usize>,
        R: Rng,
    {
        let states: Vec<usize> = states.into_iter().collect();
        let mut weights = Vec::with_capacity(states.len());

        for index in &states {
            weights.push(
                self.weights
                    .get(*index)
                    .ok_or(StateError::StateOutOfDomainBounds)?,
            );
        }

        let distribution = WeightedIndex::new(weights).map_err(|_| StateError::NoViableState)?;

        states
            .get(distribution.sample(rng))
            .map(|index| *index)
            .ok_or(StateError::NoViableState)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use super::*;

    struct TestWeightedTile(u32);

    impl Weighted for TestWeightedTile {
        fn weight(&self) -> u32 {
            self.0
        }
    }

    #[test]
    fn weighted_collapser_from() {
        let collapser = WeightedCollapser::from([
            TestWeightedTile(0),
            TestWeightedTile(3),
            TestWeightedTile(7),
        ]);

        assert_eq!(collapser.weights, Vec::from([0, 3, 7]));
    }

    #[test]
    fn unweighted_collapser() {
        let collapser = UnweightedCollapser;

        let states = [0, 1, 2];

        let mut rng = StepRng::new(0, u64::MAX / 3);

        assert_eq!(collapser.collapse(states, &mut rng), Ok(0));

        // Assertions pass but StepRng makes this test slow I don't know why.
        // assert_eq!(collapser.collapse(states, &mut rng), 1);
        // assert_eq!(collapser.collapse(states, &mut rng), 2);
    }

    #[test]
    fn unweighted_collapser_empty_states() {
        let collapser = UnweightedCollapser;

        let states = [];

        let mut rng = StepRng::new(0, 0);

        assert_eq!(
            collapser.collapse(states, &mut rng),
            Err(StateError::NoViableState)
        );
    }

    #[test]
    fn weighted_collapser() {
        let collapser = WeightedCollapser {
            weights: Vec::from([0, 3, 7]),
        };

        let states = [0, 1, 2];

        let mut rng = StepRng::new(0, u64::MAX / 4);

        assert_eq!(collapser.collapse(states, &mut rng), Ok(1));
        assert_eq!(collapser.collapse(states, &mut rng), Ok(2));
        assert_eq!(collapser.collapse(states, &mut rng), Ok(2));
    }

    #[test]
    fn weighted_collapser_empty_states() {
        let collapser = WeightedCollapser {
            weights: Vec::new(),
        };

        let states = [];

        let mut rng = StepRng::new(0, 0);

        assert_eq!(
            collapser.collapse(states, &mut rng),
            Err(StateError::NoViableState)
        );
    }

    #[test]
    fn weighted_collapser_empty_weights() {
        let collapser = WeightedCollapser {
            weights: Vec::new(),
        };

        let states = [0, 1, 2];

        let mut rng = StepRng::new(0, 0);

        assert_eq!(
            collapser.collapse(states, &mut rng),
            Err(StateError::StateOutOfDomainBounds)
        );
    }
}
