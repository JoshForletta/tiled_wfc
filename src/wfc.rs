use std::{array::from_fn, error::Error, marker::PhantomData};

use nd_matrix::Matrix;
use rand::{rngs::StdRng, SeedableRng};

use crate::{AxisPair, Collapser, State, Tile, UnweightedCollapser, Weighted, WeightedCollapser};

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
            collapser: WeightedCollapser::from(tile_set),
            rng: rng.unwrap_or(StdRng::from_entropy()),
            matrix: Matrix::fill(dimensions, State::fill(true, tile_set.len())),
        })
    }
}

pub struct WFC<'a, T, const D: usize, C = UnweightedCollapser> {
    tile_set: &'a [T],
    valid_adjacencies_map: Vec<[AxisPair<State>; D]>,
    collapser: C,
    rng: StdRng,
    matrix: Matrix<State, D>,
}

impl<'a, T, C, const D: usize> WFC<'a, T, D, C>
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
}
