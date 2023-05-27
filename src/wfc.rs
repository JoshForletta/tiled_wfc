use std::{collections::HashMap, marker::PhantomData};

use nd_matrix::{Matrix, ToIndex};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    AxisPair, Collapser, State, StateError, Tile, UnweightedCollapser, Weighted, WeightedCollapser,
};

pub struct WFCBuilder<'a, T, const D: usize, C> {
    tile_set: Option<&'a [T]>,
    dimensions: Option<[usize; D]>,
    collapser: Option<C>,
}

impl<'a, T, const D: usize, C> WFCBuilder<'a, T, D, C> {
    pub fn tile_set(mut self, tile_set: &'a [T]) -> Self {
        self.tile_set = Some(tile_set);
        self
    }

    pub fn dimensions(mut self, dimensions: [usize; D]) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    fn check(self) -> Result<(&'a [T], [usize; D], C), String> {
        let mut missing_field_messages = Vec::new();

        if self.tile_set.is_none() {
            missing_field_messages.push("`tile_set`");
        }

        if self.dimensions.is_none() {
            missing_field_messages.push("`dimension`");
        }

        if self.collapser.is_none() {
            missing_field_messages.push("`collapser`");
        }

        if missing_field_messages.len() == 0 {
            Ok((
                self.tile_set.unwrap(),
                self.dimensions.unwrap(),
                self.collapser.unwrap(),
            ))
        } else {
            Err(format!(
                "missing feilds: {}",
                missing_field_messages.join(", ")
            ))
        }
    }
}

impl<'a, T, const D: usize, R> WFCBuilder<'a, T, D, UnweightedCollapser<R>>
where
    R: Rng + SeedableRng,
{
    pub fn seed(mut self, seed: <R as SeedableRng>::Seed) -> Self {
        self.collapser = Some(UnweightedCollapser::new(<R as SeedableRng>::from_seed(
            seed,
        )));
        self
    }

    pub fn seed_from_u64(mut self, state: u64) -> Self {
        self.collapser = Some(UnweightedCollapser::new(<R as SeedableRng>::seed_from_u64(
            state,
        )));
        self
    }

    pub fn from_entropy(mut self) -> Self {
        self.collapser = Some(UnweightedCollapser::new(<R as SeedableRng>::from_entropy()));
        self
    }
}

impl<'a, T, const D: usize, C> WFCBuilder<'a, T, D, C>
where
    T: Tile<D>,
{
    pub fn build(self) -> Result<WFC<'a, T, D, C>, String> {
        todo!()
        // let (tile_set, dimensions, rng) = self.check()?;
        //
        // Ok(WFC {
        //     tile_set,
        //     valid_adjacencies_map: valid_adjacencies_map(tile_set),
        //     valid_adjacencies_cache: HashMap::new(),
        //     propagation_stack: Vec::new(),
        //     propagation_records: Vec::new(),
        //     collapser: UnweightedCollapser,
        //     rng,
        //     matrix: Matrix::fill(dimensions, State::fill(true, tile_set.len())),
        // })
    }
}

pub struct WFC<'a, T, const D: usize, C> {
    tile_set: &'a [T],
    valid_adjacencies_map: Vec<[AxisPair<State>; D]>,
    valid_adjacencies_cache: HashMap<State, [AxisPair<State>; D]>,
    propagation_stack: Vec<usize>,
    propagation_records: Vec<(usize, State)>,
    collapser: C,
    matrix: Matrix<State, D>,
}

impl<'a, T, const D: usize> WFC<'a, T, D, UnweightedCollapser<StdRng>>
where
    T: Tile<D>,
{
    pub fn builder() -> WFCBuilder<'a, T, D, UnweightedCollapser<StdRng>> {
        WFCBuilder {
            tile_set: None,
            dimensions: None,
            collapser: Some(UnweightedCollapser::new(StdRng::from_entropy())),
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
    pub fn dimensions(&self) -> &[usize; D] {
        self.matrix.dimensions()
    }

    #[inline(always)]
    pub fn dimension_offsets(&self) -> &[usize; D] {
        self.matrix.dimension_offsets()
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

    #[inline(always)]
    pub fn get_tile(&self, state: &State) -> Option<&T> {
        todo!()
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

    pub fn uncollapse_state(
        &mut self,
        stack: &mut Vec<(usize, State, Vec<(usize, State)>)>,
    ) -> Result<(), StateError> {
        todo!()
    }

    pub fn propagate(&mut self, index: usize) -> Result<Vec<(usize, State)>, StateError> {
        todo!()
    }

    pub fn unpropagate(&mut self, propagation_record: Vec<(usize, State)>) {
        todo!()
    }
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

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
    fn least_entropic_index() {
        todo!()
        // let mut wfc = WFC::<'_, TestTile, 2, _, _> {
        //     tile_set: &[],
        //     valid_adjacencies_map: Vec::new(),
        //     valid_adjacencies_cache: HashMap::new(),
        //     propagation_stack: Vec::new(),
        //     propagation_records: Vec::new(),
        //     collapser: UnweightedCollapser,
        //     rng: StepRng::new(0, 0),
        //     matrix: Matrix::fill([2, 2], State::fill(true, 3)),
        // };
        //
        // let collapser = UnweightedCollapser;
        //
        // let mut rng = StepRng::new(0, 0);
        //
        // wfc.matrix[0]
        //     .collapse(&collapser, &mut rng)
        //     .expect("valid state");
        //
        // wfc.matrix[1].set(0, false);
        //
        // assert_eq!(wfc.least_entropic_index(), Some(1));
    }

    #[test]
    fn collapse_state() {
        todo!()
        // let mut wfc = WFC::<'_, TestTile, 2, StepRng, _> {
        //     tile_set: &[],
        //     valid_adjacencies_map: Vec::new(),
        //     valid_adjacencies_cache: HashMap::new(),
        //     propagation_stack: Vec::new(),
        //     propagation_records: Vec::new(),
        //     collapser: UnweightedCollapser,
        //     rng: StepRng::new(0, 0),
        //     matrix: Matrix::fill([2, 2], State::fill(true, 3)),
        // };
        //
        // let remaining_state = wfc.collapse_state(0).expect("viable state");
        //
        // assert_eq!(wfc.matrix[0], State::new_collapsed(0));
        // assert_eq!(remaining_state, State::with_indexes([1, 2], 3));
    }
}
