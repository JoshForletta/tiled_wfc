use nd_matrix::{Matrix, ToIndex};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    state::Superposition, validation::valid_adjacencies_map, AxisPair, Collapser, State,
    StateError, Tile, UnweightedCollapser,
};

pub struct WFCBuilder<'a, T, const D: usize, C> {
    tile_set: Option<&'a [T]>,
    dimensions: Option<[usize; D]>,
    collapser: C,
}

impl<'a, T, const D: usize, C> WFCBuilder<'a, T, D, C>
where
    T: Tile<D>,
    C: Collapser,
{
    pub fn tile_set(mut self, tile_set: &'a [T]) -> Self {
        self.tile_set = Some(tile_set);
        self
    }

    pub fn dimensions(mut self, dimensions: [usize; D]) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn collapser<NC>(self, collapser: NC) -> WFCBuilder<'a, T, D, NC>
    where
        NC: Collapser,
    {
        WFCBuilder {
            tile_set: self.tile_set,
            dimensions: self.dimensions,
            collapser,
        }
    }

    fn check(self) -> (&'a [T], [usize; D], C) {
        let mut missing_field_messages = Vec::new();

        if self.tile_set.is_none() {
            missing_field_messages.push("`tile_set`");
        }

        if self.dimensions.is_none() {
            missing_field_messages.push("`dimension`");
        }

        if missing_field_messages.len() != 0 {
            panic!("missing feilds: {}", missing_field_messages.join(", "));
        }

        (
            self.tile_set.unwrap(),
            self.dimensions.unwrap(),
            self.collapser,
        )
    }
}

impl<'a, T, const D: usize, R> WFCBuilder<'a, T, D, UnweightedCollapser<R>>
where
    T: Tile<D>,
    R: Rng + SeedableRng,
{
    pub fn seed(mut self, seed: <R as SeedableRng>::Seed) -> Self {
        self.collapser = UnweightedCollapser::new(<R as SeedableRng>::from_seed(seed));
        self
    }

    pub fn seed_from_u64(mut self, state: u64) -> Self {
        self.collapser = UnweightedCollapser::new(<R as SeedableRng>::seed_from_u64(state));
        self
    }

    pub fn from_entropy(mut self) -> Self {
        self.collapser = UnweightedCollapser::new(<R as SeedableRng>::from_entropy());
        self
    }
}

impl<'a, T, const D: usize, C> WFCBuilder<'a, T, D, C>
where
    T: Tile<D>,
    C: Collapser,
{
    pub fn build(self) -> WFC<'a, T, D, C> {
        let (tile_set, dimensions, collapser) = self.check();

        WFC {
            tile_set,
            valid_adjacencies_map: valid_adjacencies_map(tile_set),
            collapser,
            matrix: Matrix::fill(dimensions, State::fill(tile_set.len())),
        }
    }
}

pub struct WFC<'a, T, const D: usize, C> {
    tile_set: &'a [T],
    valid_adjacencies_map: Vec<[AxisPair<Superposition>; D]>,
    matrix: Matrix<State, D>,
    collapser: C,
}

impl<'a, T, const D: usize> WFC<'a, T, D, UnweightedCollapser<StdRng>>
where
    T: Tile<D>,
{
    pub fn builder() -> WFCBuilder<'a, T, D, UnweightedCollapser<StdRng>> {
        WFCBuilder {
            tile_set: None,
            dimensions: None,
            collapser: UnweightedCollapser::new(StdRng::from_entropy()),
        }
    }
}

impl<'a, T, const D: usize, C> WFC<'a, T, D, C>
where
    T: Tile<D>,
    C: Collapser,
{
    #[inline(always)]
    pub fn matrix(&self) -> &Matrix<State, D> {
        &self.matrix
    }

    #[inline(always)]
    pub fn tile_set(&self) -> &[T] {
        self.tile_set
    }

    #[inline(always)]
    pub fn collapser(&self) -> &C {
        &self.collapser
    }

    #[inline(always)]
    pub fn get<I>(&self, index: I) -> Option<&State>
    where
        I: ToIndex<D>,
    {
        self.matrix.get(index)
    }

    #[inline(always)]
    pub fn get_mut<I>(&mut self, index: I) -> Option<&mut State>
    where
        I: ToIndex<D>,
    {
        self.matrix.get_mut(index)
    }

    pub fn least_entropic_index(&self) -> Option<usize> {
        todo!()
    }

    pub fn collapse(&mut self) -> Result<(), StateError> {
        todo!()
    }

    pub fn collapse_state(&mut self, index: usize) -> Result<State, StateError> {
        todo!()
    }

    pub fn uncollapse_state(&mut self) -> Result<(), StateError> {
        todo!()
    }

    fn propagate(&mut self, index: usize) -> Result<Vec<(usize, State)>, StateError> {
        todo!()
    }

    fn unpropagate(&mut self, propagation_record: Vec<(usize, State)>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use crate::test_utils::TILE_SET;

    use super::*;

    #[derive(Debug)]
    struct TestTile {
        sockets: [AxisPair<char>; 2],
    }

    impl Tile<2> for TestTile {
        type Socket = char;

        fn sockets(&self) -> [AxisPair<<Self as Tile<2>>::Socket>; 2] {
            self.sockets
        }
    }

    #[test]
    fn builder() {
        let wfc = WFC::builder()
            .tile_set(TILE_SET)
            .dimensions([2, 2])
            .collapser(UnweightedCollapser::new(StepRng::new(0, 0)))
            .build();

        let initial_matrix = Matrix::fill([2, 2], State::fill(3));

        assert_eq!(wfc.matrix(), &initial_matrix);
    }

    #[test]
    fn least_entropic_index() {
        todo!()
    }

    #[test]
    fn collapse_state() {
        todo!()
    }
}
