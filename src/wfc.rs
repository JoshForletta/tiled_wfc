use std::{array::from_fn, collections::HashMap, iter::once, marker::PhantomData};

use nd_matrix::{Matrix, ToIndex};
use rand::{rngs::StdRng, SeedableRng};

use crate::{
    AxisPair, Collapser, State, StateError, Tile, UnweightedCollapser, Weighted, WeightedCollapser,
};

pub fn valid_adjacencies<T, const D: usize>(tile: &T, tile_set: &[T]) -> [AxisPair<State>; D]
where
    T: Tile<D>,
{
    let empty_state = State::fill(false, tile_set.len());
    let empty_pair = AxisPair::new(empty_state.clone(), empty_state.clone());
    let mut valid_adjacencies: [AxisPair<State>; D] = from_fn(|_| empty_pair.clone());

    let sockets = tile.sockets();

    for (index, possible_tile) in tile_set.into_iter().enumerate() {
        for dimension in 0..D {
            let possible_sockets = possible_tile.sockets();
            if sockets[dimension].pos == possible_sockets[dimension].neg {
                valid_adjacencies[dimension].pos.set(index, true);
            }

            if sockets[dimension].neg == possible_sockets[dimension].pos {
                valid_adjacencies[dimension].neg.set(index, true);
            }
        }
    }

    valid_adjacencies
}

pub fn valid_adjacencies_map<T, const D: usize>(tile_set: &[T]) -> Vec<[AxisPair<State>; D]>
where
    T: Tile<D>,
{
    tile_set
        .into_iter()
        .map(|tile| valid_adjacencies(tile, tile_set))
        .collect()
}

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

    pub fn get_adjacent_indexes(&self, index: usize) -> [AxisPair<Option<usize>>; D] {
        let mut adjacencies: [AxisPair<Option<usize>>; D] = [Default::default(); D];
        let coordinate_offsets: Vec<_> = once(&1)
            .chain(self.matrix.dimension_offsets().into_iter())
            .collect();

        for dimension in 0..D {
            let corrdinate_offset = *coordinate_offsets[dimension];

            let dimension_offset = self.matrix.dimension_offsets()[dimension];
            let higher_dimension_index = index / dimension_offset;
            let lower_bound = higher_dimension_index * dimension_offset;
            let upper_bound = (higher_dimension_index + 1) * dimension_offset;

            adjacencies[dimension].pos = index
                .checked_add(corrdinate_offset)
                .filter(|index| index < &upper_bound);

            adjacencies[dimension].neg = index
                .checked_sub(corrdinate_offset)
                .filter(|index| index >= &lower_bound);
        }

        adjacencies
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

    pub fn collapse_state(&mut self, index: usize) -> State {
        self.matrix
            .get_mut(index)
            .expect("`index` is a valid index")
            .collapse(&self.collapser, &mut self.rng)
    }

    pub fn collapse(&mut self) -> Result<(), ()> {
        let mut stack = Vec::new();

        while let Some(index) = self.least_entropic_index() {
            let remaining_state = self.collapse_state(index);

            if let Ok(propagation_records) = self.propagate(index) {
                stack.push((index, remaining_state, propagation_records));
            } else {
                let (index, remaining_state, propagation_records) = stack.pop().ok_or(())?;

                self.matrix[index] = remaining_state;

                self.unpropagate(propagation_records);
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

    pub fn propagate(&mut self, index: usize) -> Result<Vec<(usize, State)>, ()> {
        let mut propagation_records = Vec::new();
        let mut stack = Vec::from([index]);

        while let Some(index) = stack.pop() {
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

            let adjacencies = self.get_adjacent_indexes(index);

            for dimension in 0..D {
                // TODO: impl IntoIter for AxisPair
                if let Some(positive_adjacency_index) = adjacencies[dimension].pos {
                    let adjacency = &mut self.matrix[positive_adjacency_index];
                    let valid_adjacency = &valid_adjacencies[dimension].pos;

                    if !valid_adjacency.contains(&adjacency) {
                        propagation_records.push((positive_adjacency_index, adjacency.clone()));

                        adjacency.constrain(valid_adjacency);

                        if adjacency.count() == 0 {
                            self.unpropagate(propagation_records);
                            return Err(());
                        }

                        stack.push(positive_adjacency_index);
                    }
                }

                if let Some(negitive_adjacency_index) = adjacencies[dimension].neg {
                    let adjacency = &mut self.matrix[negitive_adjacency_index];
                    let valid_adjacency = &valid_adjacencies[dimension].neg;

                    if !valid_adjacency.contains(&adjacency) {
                        propagation_records.push((negitive_adjacency_index, adjacency.clone()));

                        adjacency.constrain(valid_adjacency);

                        if adjacency.count() == 0 {
                            self.unpropagate(propagation_records);
                            return Err(());
                        }

                        stack.push(negitive_adjacency_index);
                    }
                }
            }
        }

        Ok(propagation_records)
    }

    pub fn unpropagate(&mut self, propagation_records: Vec<(usize, State)>) {
        for (index, state) in propagation_records.into_iter() {
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

    impl TestTile {
        const fn new(sockets: [AxisPair<char>; 2]) -> Self {
            Self { sockets }
        }
    }

    impl Tile<2> for TestTile {
        type Socket = char;

        fn sockets(&self) -> [AxisPair<<Self as Tile<2>>::Socket>; 2] {
            self.sockets
        }
    }

    #[test]
    fn test_valid_adjacencies() {
        let tile_set: &[TestTile] = &[
            TestTile::new([AxisPair::new('a', 'b'), AxisPair::new('b', 'a')]),
            TestTile::new([AxisPair::new('a', 'a'), AxisPair::new('a', 'a')]),
            TestTile::new([AxisPair::new('b', 'b'), AxisPair::new('b', 'b')]),
        ];

        let tile = &tile_set[0];

        let output = [
            AxisPair::new(State::with_index(1, 3), State::with_index(2, 3)),
            AxisPair::new(State::with_index(2, 3), State::with_index(1, 3)),
        ];

        assert_eq!(valid_adjacencies(tile, tile_set), output)
    }

    #[test]
    fn test_valid_adjacencies_map() {
        let tile_set: &[TestTile] = &[
            TestTile::new([AxisPair::new('a', 'b'), AxisPair::new('b', 'a')]),
            TestTile::new([AxisPair::new('a', 'a'), AxisPair::new('a', 'a')]),
            TestTile::new([AxisPair::new('b', 'b'), AxisPair::new('b', 'b')]),
        ];

        let output = Vec::from([
            [
                AxisPair::new(State::with_index(1, 3), State::with_index(2, 3)),
                AxisPair::new(State::with_index(2, 3), State::with_index(1, 3)),
            ],
            [
                AxisPair::new(State::with_index(1, 3), State::with_indexes([0, 1], 3)),
                AxisPair::new(State::with_indexes([0, 1], 3), State::with_index(1, 3)),
            ],
            [
                AxisPair::new(State::with_indexes([0, 2], 3), State::with_index(2, 3)),
                AxisPair::new(State::with_index(2, 3), State::with_indexes([0, 2], 3)),
            ],
        ]);

        assert_eq!(valid_adjacencies_map(tile_set), output);
    }

    #[test]
    fn get_adjacent_indexes() {
        let wfc = WFC::<'_, TestTile, 2, _> {
            tile_set: &[],
            valid_adjacencies_map: Vec::new(),
            valid_adjacencies_cache: HashMap::new(),
            collapser: UnweightedCollapser,
            rng: StdRng::from_entropy(),
            matrix: Matrix::fill([2, 2], State::fill(true, 3)),
        };

        let adjacencies = wfc.get_adjacent_indexes(2);

        assert_eq!(
            adjacencies,
            [AxisPair::new(Some(3), None), AxisPair::new(None, Some(0)),]
        );
    }

    #[test]
    fn get_valid_adjacencies() {
        let tile_set: &[TestTile] = &[
            TestTile::new([AxisPair::new('a', 'b'), AxisPair::new('b', 'a')]),
            TestTile::new([AxisPair::new('a', 'a'), AxisPair::new('a', 'a')]),
            TestTile::new([AxisPair::new('b', 'b'), AxisPair::new('b', 'b')]),
        ];

        let valid_adjacencies_map = Vec::from([
            [
                AxisPair::new(State::with_index(1, 3), State::with_index(2, 3)),
                AxisPair::new(State::with_index(2, 3), State::with_index(1, 3)),
            ],
            [
                AxisPair::new(State::with_index(1, 3), State::with_indexes([0, 1], 3)),
                AxisPair::new(State::with_indexes([0, 1], 3), State::with_index(1, 3)),
            ],
            [
                AxisPair::new(State::with_indexes([0, 2], 3), State::with_index(2, 3)),
                AxisPair::new(State::with_index(2, 3), State::with_indexes([0, 2], 3)),
            ],
        ]);

        let wfc = WFC::<'_, TestTile, 2, _> {
            tile_set,
            valid_adjacencies_map,
            valid_adjacencies_cache: HashMap::new(),
            collapser: UnweightedCollapser,
            rng: StdRng::from_entropy(),
            matrix: Matrix::fill([2, 2], State::fill(true, 3)),
        };

        let state = State::with_indexes([0, 1], 3);

        assert_eq!(
            wfc.get_valid_adjacencies(&State::fill(true, 4)),
            Err(StateError::StateIndexOutOfBounds)
        );
        assert_eq!(
            wfc.get_valid_adjacencies(&state).unwrap(),
            [
                AxisPair::new(State::with_index(1, 3), State::fill(true, 3)),
                AxisPair::new(State::fill(true, 3), State::with_index(1, 3)),
            ]
        );
    }

    #[test]
    fn least_entropic_index() {
        let mut wfc = WFC::<'_, TestTile, 2, _> {
            tile_set: &[],
            valid_adjacencies_map: Vec::new(),
            valid_adjacencies_cache: HashMap::new(),
            collapser: UnweightedCollapser,
            rng: StdRng::from_entropy(),
            matrix: Matrix::fill([2, 2], State::fill(true, 3)),
        };

        wfc.matrix[0] = State::with_index(1, 3);
        wfc.matrix[1].set(0, false);

        assert_eq!(wfc.least_entropic_index(), Some(1));
    }
}
