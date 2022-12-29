use std::error::Error;

use nd_matrix::Matrix;
use rand::rngs::StdRng;

use crate::{AxisPair, Collapser, State, StateError, Tile};

pub struct WFC<'a, T, C, const D: usize> {
    tile_set: &'a [T],
    possible_adjacencies: Vec<[AxisPair<State>; D]>,
    collapser: C,
    rng: StdRng,
    matrix: Matrix<State, D>,
}

impl<'a, T, C, const D: usize> WFC<'a, T, C, D>
where
    T: Tile<D>,
    C: Collapser,
{
    pub fn collapse(&mut self) -> Result<(), Box<dyn Error>> {
        while let Some(index) = self.least_entropic_index() {
            self.matrix
                .get_mut(index)
                .expect("`self.least_entropic_index()` should return a valid index")
                .collapse(&self.collapser, &mut self.rng)?;
            self.propagate(index);
        }

        Ok(())
    }

    pub fn least_entropic_index(&self) -> Option<usize> {
        todo!()
    }

    pub fn propagate(&self, index: usize) {
        todo!()
    }
}
