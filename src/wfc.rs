use std::{array::from_fn, collections::HashMap, marker::PhantomData};

use nd_matrix::{Matrix, ToIndex};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    validation::valid_adjacencies_map, AxisPair, Collapser, State, StateError, Tile,
    UnweightedCollapser, Weighted, WeightedCollapser,
};

pub struct WFCBuilder<'a, T, const D: usize, C = UnweightedCollapser> {
    tile_set: Option<&'a [T]>,
    dimensions: Option<[usize; D]>,
    _collapser: PhantomData<C>,
    rng: Option<StdRng>,
}

impl<'a, T, C, const D: usize> WFCBuilder<'a, T, D, C> {
    pub fn tile_set(mut self, tile_set: &'a [T]) -> Self {
        self.tile_set = Some(tile_set);
        self
    }

    pub fn dimensions(mut self, dimensions: [usize; D]) -> Self {
        self.dimensions = Some(dimensions);
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.rng = Some(StdRng::seed_from_u64(seed));
        self
    }
}

impl<'a, T, C, const D: usize> WFCBuilder<'a, T, D, C>
where
    T: Weighted,
{
    pub fn weighted(self) -> WFCBuilder<'a, T, D, WeightedCollapser> {
        WFCBuilder {
            tile_set: self.tile_set,
            dimensions: self.dimensions,
            _collapser: PhantomData,
            rng: self.rng,
        }
    }
}

impl<'a, T, const D: usize> WFCBuilder<'a, T, D, UnweightedCollapser>
where
    T: Tile<D>,
{
    pub fn build(self) -> Result<WFC<'a, T, D, UnweightedCollapser>, ()> {
        let WFCBuilder { tile_set: Some(tile_set), dimensions: Some(dimensions), _collapser, rng } = self else {
            return Err(());
        };

        Ok(WFC {
            tile_set,
            valid_adjacencies_map: valid_adjacencies_map(tile_set),
            valid_adjacencies_cache: HashMap::new(),
            propagation_stack: Vec::new(),
            propagation_records: Vec::new(),
            collapser: UnweightedCollapser,
            rng: rng.unwrap_or(StdRng::from_entropy()),
            matrix: Matrix::fill(dimensions, State::fill(true, tile_set.len())),
        })
    }
}

impl<'a, T, const D: usize> WFCBuilder<'a, T, D, WeightedCollapser>
where
    T: Tile<D> + Weighted,
    &'a T: Weighted,
{
    pub fn build(self) -> Result<WFC<'a, T, D, WeightedCollapser>, ()> {
        let WFCBuilder { tile_set: Some(tile_set), dimensions: Some(dimensions), _collapser, rng } = self else {
            return Err(());
        };

        Ok(WFC {
            tile_set,
            valid_adjacencies_map: valid_adjacencies_map(tile_set),
            valid_adjacencies_cache: HashMap::new(),
            propagation_stack: Vec::new(),
            propagation_records: Vec::new(),
            collapser: WeightedCollapser::from(tile_set),
            rng: rng.unwrap_or(StdRng::from_entropy()),
            matrix: Matrix::fill(dimensions, State::fill(true, tile_set.len())),
        })
    }
}

pub struct WFC<'a, T, const D: usize, C = UnweightedCollapser, R = StdRng> {
    tile_set: &'a [T],
    valid_adjacencies_map: Vec<[AxisPair<State>; D]>,
    valid_adjacencies_cache: HashMap<State, [AxisPair<State>; D]>,
    propagation_stack: Vec<usize>,
    propagation_records: Vec<(usize, State)>,
    collapser: C,
    rng: R,
    matrix: Matrix<State, D>,
}

impl<'a, T, C, const D: usize, R> WFC<'a, T, D, C, R>
where
    T: Tile<D>,
    C: Collapser,
    R: Rng,
{
    pub fn builder() -> WFCBuilder<'a, T, D, C> {
        WFCBuilder {
            tile_set: None,
            dimensions: None,
            _collapser: PhantomData,
            rng: None,
        }
    }

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
    pub fn rng(&self) -> &R {
        &self.rng
    }

    #[inline(always)]
    pub fn rng_mut(&mut self) -> &mut R {
        &mut self.rng
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
        self.tile_set.get(state.state_index()?)
    }

    pub fn get_valid_adjacencies(&self, state: &State) -> Result<[AxisPair<State>; D], StateError> {
        let empty_state = State::fill(false, self.tile_set.len());
        let empty_pair = AxisPair::new(empty_state.clone(), empty_state.clone());
        let mut cumulative_valid_adjacencies: [AxisPair<State>; D] =
            from_fn(|_| empty_pair.clone());

        for state_index in state.state_indexes() {
            let valid_adjacencies = self
                .valid_adjacencies_map
                .get(state_index)
                .ok_or(StateError::StateIndexOutOfBounds)?;

            for dimension in 0..D {
                cumulative_valid_adjacencies[dimension].pos |= &valid_adjacencies[dimension].pos;
                cumulative_valid_adjacencies[dimension].neg |= &valid_adjacencies[dimension].neg;
            }
        }

        Ok(cumulative_valid_adjacencies)
    }

    pub fn least_entropic_index(&self) -> Option<usize> {
        self.matrix
            .matrix()
            .into_iter()
            .enumerate()
            .filter(|(_index, state)| !state.is_collapsed())
            .map(|(index, state)| (index, state.count()))
            .min_by(|(_min_index, min_count), (_index, count)| min_count.cmp(count))
            .map(|(index, _count)| index)
    }

    pub fn collapse(&mut self) -> Result<(), StateError> {
        let mut stack = Vec::new();

        while let Some(index) = self.least_entropic_index() {
            if let Ok(remaining_state) = self.collapse_state(index) {
                if let Ok(propagation_stack) = self.propagate(index) {
                    stack.push((index, remaining_state, propagation_stack));

                    continue;
                }
                
                self.matrix[index] = remaining_state;

                continue;
            };

            let (index, remaining_state, propagation_stack) =
                stack.pop().ok_or(StateError::NoViableState)?;

            self.matrix[index] = remaining_state;

            self.unpropagate(propagation_stack);
        }

        Ok(())
    }

    pub fn collapse_state(&mut self, index: usize) -> Result<State, StateError> {
        let state = &mut self.matrix[index];
        let mut remaining_state = state.clone();
        let state_index = state.collapse(&self.collapser, &mut self.rng)?;

        remaining_state.set(state_index, false);

        Ok(remaining_state)
    }

    pub fn propagate(&mut self, index: usize) -> Result<Vec<(usize, State)>, StateError> {
        let mut changed = Vec::from([index]);
        let mut propagation_record = Vec::new();

        while let Some(index) = changed.pop() {
            let state = &self.matrix[index];

            let valid_adjacencies = self.get_valid_adjacencies(state).unwrap();
            let adjacent_indexes = self.matrix.get_adjacent_indexes(index);

            for dimension in 0..D {
                if let Some(adjacent_index) = adjacent_indexes[dimension].pos {
                    let adjacent_state = &mut self.matrix[adjacent_index];
                    let valid_state = &valid_adjacencies[dimension].pos;

                    if !valid_state.contains(&adjacent_state) {
                        if adjacent_state.is_collapsed() {
                            self.unpropagate(propagation_record);
                            return Err(StateError::NoViableState);
                        }

                        propagation_record.push((adjacent_index, adjacent_state.clone()));

                        adjacent_state.constrain(valid_state);

                        changed.push(adjacent_index);
                    }
                }

                if let Some(adjacent_index) = adjacent_indexes[dimension].neg {
                    let adjacent_state = &mut self.matrix[adjacent_index];
                    let valid_state = &valid_adjacencies[dimension].neg;

                    if !valid_state.contains(&adjacent_state) {
                        if adjacent_state.is_collapsed() {
                            self.unpropagate(propagation_record);
                            return Err(StateError::NoViableState);
                        }

                        propagation_record.push((adjacent_index, adjacent_state.clone()));

                        adjacent_state.constrain(valid_state);

                        changed.push(adjacent_index);
                    }
                }
            }
        }

        Ok(propagation_record)
    }

    pub fn unpropagate(&mut self, propagation_record: Vec<(usize, State)>) {
        for (index, state) in propagation_record.into_iter() {
            self.matrix[index] = state;
        }
    }
}

#[cfg(test)]
mod tests {
    use bitvec::vec::BitVec;
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
        let mut wfc = WFC::<'_, TestTile, 2, _> {
            tile_set: &[],
            valid_adjacencies_map: Vec::new(),
            valid_adjacencies_cache: HashMap::new(),
            propagation_stack: Vec::new(),
            propagation_records: Vec::new(),
            collapser: UnweightedCollapser,
            rng: StdRng::from_entropy(),
            matrix: Matrix::fill([2, 2], State::fill(true, 3)),
        };

        let collapser = UnweightedCollapser;

        let mut rng = StepRng::new(0, 0);

        wfc.matrix[0]
            .collapse(&collapser, &mut rng)
            .expect("valid state");

        wfc.matrix[1].set(0, false);

        assert_eq!(wfc.least_entropic_index(), Some(1));
    }

    #[test]
    fn collapse_state() {
        let mut wfc = WFC::<'_, TestTile, 2, _, StepRng> {
            tile_set: &[],
            valid_adjacencies_map: Vec::new(),
            valid_adjacencies_cache: HashMap::new(),
            propagation_stack: Vec::new(),
            propagation_records: Vec::new(),
            collapser: UnweightedCollapser,
            rng: StepRng::new(0, 0),
            matrix: Matrix::fill([2, 2], State::fill(true, 3)),
        };

        let remaining_state = wfc.collapse_state(0).expect("viable state");

        assert_eq!(wfc.matrix[0], State::new_collapsed(0, 3));
        assert_eq!(remaining_state, State::with_indexes([1, 2], 3));
    }
}
