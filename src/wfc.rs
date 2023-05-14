use std::{array::from_fn, collections::HashMap, marker::PhantomData};

use nd_matrix::{Matrix, ToIndex};
use rand::{rngs::StdRng, SeedableRng};

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

pub struct WFC<'a, T, const D: usize, C = UnweightedCollapser> {
    tile_set: &'a [T],
    valid_adjacencies_map: Vec<[AxisPair<State>; D]>,
    valid_adjacencies_cache: HashMap<State, [AxisPair<State>; D]>,
    propagation_stack: Vec<usize>,
    propagation_records: Vec<(usize, State)>,
    collapser: C,
    rng: StdRng,
    matrix: Matrix<State, D>,
}

impl<'a, T, C, const D: usize> WFC<'a, T, D, C>
where
    T: Tile<D>,
    C: Collapser,
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
    pub fn rng(&self) -> &StdRng {
        &self.rng
    }

    #[inline(always)]
    pub fn rng_mut(&mut self) -> &mut StdRng {
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

    pub fn collapse(&mut self) -> Result<(), ()> {
        let mut stack = Vec::new();

        while let Some(index) = self.least_entropic_index() {
            let state = &mut self.matrix[index];
            let mut remaining_state = state.clone();
            let state_index = state
                .collapse(&self.collapser, &mut self.rng)
                .expect("valid state");

            remaining_state.set(state_index, false);

            self.propagation_stack.push(index);

            if let Ok(propagation_depth) = self.propagate() {
                stack.push((index, remaining_state, propagation_depth));
            } else {
                let (index, remaining_state, propagation_depth) = stack.pop().ok_or(())?;

                self.matrix[index] = remaining_state;

                self.unpropagate(propagation_depth);
            };
        }

        Ok(())
    }

    pub fn least_entropic_index(&self) -> Option<usize> {
        self.matrix
            .matrix()
            .into_iter()
            .map(|state| state.count())
            .enumerate()
            .filter_map(|(index, count)| (count > 1).then_some((index, count)))
            .min_by(|(_min_index, min_count), (_index, count)| min_count.cmp(count))
            .map(|(index, _count)| index)
    }

    pub fn propagate(&mut self) -> Result<usize, ()> {
        let mut propagation_depth = 0;

        while let Some(index) = self.propagation_stack.pop() {
            let state = &self.matrix[index];

            // TODO: use HashMap::entry
            let valid_adjacencies = match self.valid_adjacencies_cache.get(state) {
                Some(valid_adjacencies) => valid_adjacencies,
                None => {
                    let valid_adjacencies =
                        self.get_valid_adjacencies(state).expect("`index` is valid");
                    self.valid_adjacencies_cache
                        .insert(state.clone(), valid_adjacencies);
                    &self.valid_adjacencies_cache[state]
                }
            };

            let adjacencies = self.matrix.get_adjacent_indexes(index);

            for dimension in 0..D {
                // TODO: impl IntoIter for AxisPair
                if let Some(positive_adjacency_index) = adjacencies[dimension].pos {
                    let adjacency = &mut self.matrix[positive_adjacency_index];
                    let valid_adjacency = &valid_adjacencies[dimension].pos;

                    if !valid_adjacency.contains(&adjacency) {
                        self.propagation_records
                            .push((positive_adjacency_index, adjacency.clone()));
                        propagation_depth += 1;

                        adjacency.constrain(valid_adjacency);

                        if adjacency.count() == 0 {
                            self.unpropagate(propagation_depth);
                            return Err(());
                        }

                        self.propagation_stack.push(positive_adjacency_index);
                    }
                }

                if let Some(negitive_adjacency_index) = adjacencies[dimension].neg {
                    let adjacency = &mut self.matrix[negitive_adjacency_index];
                    let valid_adjacency = &valid_adjacencies[dimension].neg;

                    if !valid_adjacency.contains(&adjacency) {
                        self.propagation_records
                            .push((negitive_adjacency_index, adjacency.clone()));
                        propagation_depth += 1;

                        adjacency.constrain(valid_adjacency);

                        if adjacency.count() == 0 {
                            self.unpropagate(propagation_depth);
                            return Err(());
                        }

                        self.propagation_stack.push(negitive_adjacency_index);
                    }
                }
            }
        }

        Ok(propagation_depth)
    }

    pub fn unpropagate(&mut self, propagation_depth: usize) {
        for _ in 0..propagation_depth {
            let (index, state) = self
                .propagation_records
                .pop()
                .expect("`propagation_depth` <= `self.propagation_records.len()`");

            self.matrix[index] = state;
        }
    }
}

#[cfg(test)]
mod tests {
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

        wfc.matrix[0] = State::with_index(1, 3);
        wfc.matrix[1].set(0, false);

        assert_eq!(wfc.least_entropic_index(), Some(1));
    }
}
