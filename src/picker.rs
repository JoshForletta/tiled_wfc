use nd_matrix::Matrix;
use rand::{
    seq::{IteratorRandom, SliceRandom},
    Rng,
};

use crate::State;

pub trait Picker<const D: usize> {
    /// Picks a superimposed state from `matrix`. Returns [`None`] if all
    /// states in `matrix` are collapsed.
    fn pick(&mut self, matrix: &Matrix<State, D>) -> Option<usize>;
}

/// Picks the first superimposed state from origin.
pub struct LinearPicker;

impl<const D: usize> Picker<D> for LinearPicker {
    fn pick(&mut self, matrix: &Matrix<State, D>) -> Option<usize> {
        matrix
            .into_iter()
            .enumerate()
            .find(|(_, state)| !state.is_collapsed())
            .map(|(index, _)| index)
    }
}

/// Picks a superimposed state with least number of states.
pub struct LeastEntropicPicker;

impl<const D: usize> Picker<D> for LeastEntropicPicker {
    fn pick(&mut self, matrix: &Matrix<State, D>) -> Option<usize> {
        matrix
            .into_iter()
            .enumerate()
            .filter(|(_, state)| !state.is_collapsed())
            .map(|(index, state)| (index, state.count()))
            .min_by(|(_, min_count), (_, count)| min_count.cmp(count))
            .map(|(index, _)| index)
    }
}

/// Picks a superimposed state with least number of states, adding randomness
/// to states with an equal count.
pub struct RandLeastEntropicPicker<R> {
    rng: R,
    mins: Vec<(usize, usize)>,
}

impl<R> RandLeastEntropicPicker<R>
where
    R: Rng,
{
    pub fn new(rng: R) -> Self {
        Self {
            rng,
            mins: Vec::new(),
        }
    }
}

impl<R, const D: usize> Picker<D> for RandLeastEntropicPicker<R>
where
    R: Rng,
{
    fn pick(&mut self, matrix: &Matrix<State, D>) -> Option<usize> {
        let min = matrix
            .into_iter()
            .filter(|state| !state.is_collapsed())
            .map(|state| state.count())
            .min()?;

        self.mins.clear();
        self.mins.extend(
            matrix
                .into_iter()
                .enumerate()
                .filter(|(_, state)| !state.is_collapsed() && state.count() == min)
                .map(|(index, state)| (index, state.count())),
        );

        self.mins.shuffle(&mut self.rng);
        self.mins.choose(&mut self.rng).map(|(index, _)| *index)
    }
}

/// Picks a superimposed state at random.
pub struct RandPicker<R> {
    rng: R,
}

impl<R> RandPicker<R>
where
    R: Rng,
{
    pub fn new(rng: R) -> Self {
        Self { rng }
    }
}

impl<R, const D: usize> Picker<D> for RandPicker<R>
where
    R: Rng,
{
    fn pick(&mut self, matrix: &Matrix<State, D>) -> Option<usize> {
        matrix
            .into_iter()
            .enumerate()
            .filter(|(_, state)| !state.is_collapsed())
            .choose(&mut self.rng)
            .map(|(index, _)| index)
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use super::*;

    fn test_matrix() -> Matrix<State, 2> {
        Matrix::from_with_dimensions(
            [2, 2],
            [
                State::Collapsed(0),
                State::from_iter([true, false, true]),
                State::from_iter([false, true, false]),
                State::fill(3),
            ],
        )
    }

    fn assert_picker_collapsed<P>(mut picker: P)
    where
        P: Picker<2>,
    {
        let matrix = Matrix::from_with_dimensions(
            [2, 2],
            [
                State::Collapsed(0),
                State::Collapsed(0),
                State::Collapsed(0),
                State::Collapsed(0),
            ],
        );

        assert_eq!(picker.pick(&matrix), None);
    }

    #[test]
    fn linear_picker() {
        let matrix = test_matrix();
        let mut picker = LinearPicker;

        assert_eq!(picker.pick(&matrix), Some(1));
    }

    #[test]
    fn linear_picker_collapsed() {
        assert_picker_collapsed(LinearPicker);
    }

    #[test]
    fn least_entropic_picker() {
        let matrix = test_matrix();
        let mut picker = LeastEntropicPicker;

        assert_eq!(picker.pick(&matrix), Some(2));
    }

    #[test]
    fn least_entropic_picker_collapsed() {
        assert_picker_collapsed(LeastEntropicPicker);
    }

    #[test]
    fn rand_least_entropic_picker() {
        let matrix = test_matrix();
        let mut picker = RandLeastEntropicPicker::new(StepRng::new(0, 0));

        assert_eq!(picker.pick(&matrix), Some(2));
    }

    #[test]
    fn rand_least_entropic_picker_collapsed() {
        assert_picker_collapsed(RandLeastEntropicPicker::new(StepRng::new(0, 0)));
    }

    #[test]
    fn rand_picker() {
        let matrix = test_matrix();
        let mut picker = RandPicker::new(StepRng::new(0, 0));

        assert_eq!(picker.pick(&matrix), Some(3));
    }

    #[test]
    fn rand_picker_collapsed() {
        assert_picker_collapsed(RandPicker::new(StepRng::new(0, 0)));
    }
}
