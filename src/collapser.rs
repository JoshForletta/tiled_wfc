use rand::{distributions::WeightedIndex, prelude::Distribution, seq::IteratorRandom, Rng};

use crate::Weighted;

pub trait Collapser {
    fn collapse<I, R>(&self, states: I, rng: &mut R) -> usize
    where
        I: IntoIterator<Item = usize>,
        R: Rng;
}

#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct UnweightedCollapser;

impl Collapser for UnweightedCollapser {
    fn collapse<I, R>(&self, states: I, rng: &mut R) -> usize
    where
        I: IntoIterator<Item = usize>,
        R: Rng,
    {
        states
            .into_iter()
            .choose(rng)
            .expect("`states` to contain at least one state")
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
    fn collapse<I, R>(&self, states: I, rng: &mut R) -> usize
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
                    .expect("`states` to yeild valid states"),
            );
        }

        let distribution =
            WeightedIndex::new(weights).expect("`states` to contain at least one viable state");

        *states
            .get(distribution.sample(rng))
            .expect("`distribution.sample(rng)` should return a valid index because `weights` is same length as `states`")
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

        assert_eq!(collapser.collapse(states, &mut rng), 0);

        // Assertions pass but StepRng makes this test slow I don't know why.
        // assert_eq!(collapser.collapse(states, &mut rng), 1);
        // assert_eq!(collapser.collapse(states, &mut rng), 2);
    }

    #[test]
    #[should_panic]
    fn unweighted_collapser_empty_states() {
        let collapser = UnweightedCollapser;

        let states = [];

        let mut rng = StepRng::new(0, 0);

        collapser.collapse(states, &mut rng);
    }

    #[test]
    fn weighted_collapser() {
        let collapser = WeightedCollapser {
            weights: Vec::from([0, 3, 7]),
        };

        let states = [0, 1, 2];

        let mut rng = StepRng::new(0, u64::MAX / 4);

        assert_eq!(collapser.collapse(states, &mut rng), 1);
        assert_eq!(collapser.collapse(states, &mut rng), 2);
        assert_eq!(collapser.collapse(states, &mut rng), 2);
    }

    #[test]
    #[should_panic]
    fn weighted_collapser_empty_states() {
        let collapser = WeightedCollapser {
            weights: Vec::new(),
        };

        let states = [];

        let mut rng = StepRng::new(0, 0);

        collapser.collapse(states, &mut rng);
    }

    #[test]
    #[should_panic]
    fn weighted_collapser_empty_weights() {
        let collapser = WeightedCollapser {
            weights: Vec::new(),
        };

        let states = [0, 1, 2];

        let mut rng = StepRng::new(0, 0);

        collapser.collapse(states, &mut rng);
    }
}
