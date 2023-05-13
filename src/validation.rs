use std::array::from_fn;

use nd_matrix::Matrix;

use crate::{AxisPair, State, Tile};

pub fn valid_adjacencies_from_tile<T, const D: usize>(
    tile: &T,
    tile_set: &[T],
) -> [AxisPair<State>; D]
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

pub fn valid_adjacencies_from_state<const D: usize>(
    state: &State,
    valid_adjacencies_map: &Vec<[AxisPair<State>; D]>,
) -> [AxisPair<State>; D] {
    let empty_state = State::fill(false, valid_adjacencies_map.len());
    let empty_pair = AxisPair::new(empty_state.clone(), empty_state.clone());
    let mut valid_state_adjacencies: [AxisPair<State>; D] = from_fn(|_| empty_pair.clone());

    for state_index in state.state_indexes() {
        let valid_adjacencies = &valid_adjacencies_map[state_index];

        for dimension in 0..D {
            valid_state_adjacencies[dimension].pos |= &valid_adjacencies[dimension].pos;
            valid_state_adjacencies[dimension].neg |= &valid_adjacencies[dimension].neg;
        }
    }

    valid_state_adjacencies
}

pub fn valid_adjacencies_map<T, const D: usize>(tile_set: &[T]) -> Vec<[AxisPair<State>; D]>
where
    T: Tile<D>,
{
    tile_set
        .into_iter()
        .map(|tile| valid_adjacencies_from_tile(tile, tile_set))
        .collect()
}

pub fn validate_solution<const D: usize>(
    matrix: &Matrix<State, D>,
    valid_adjacencies_map: &Vec<[AxisPair<State>; D]>,
) -> bool {
    for (index, state) in matrix.into_iter().enumerate() {
        let valid_adjacencies = valid_adjacencies_from_state(state, &valid_adjacencies_map);
        let adjacencies = matrix.get_adjacencies(index);

        for dimension in 0..D {
            let valid_adjacency_pair = &valid_adjacencies[dimension];
            let adjacency_pair = adjacencies[dimension];

            if let Some(adjacency) = adjacency_pair.pos {
                if !valid_adjacency_pair.pos.contains(adjacency) {
                    return false;
                }
            }

            if let Some(adjacency) = adjacency_pair.neg {
                if !valid_adjacency_pair.neg.contains(adjacency) {
                    return false;
                }
            }
        }
    }

    true
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
    fn test_valid_adjacencies_from_tile() {
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

        assert_eq!(valid_adjacencies_from_tile(tile, tile_set), output)
    }

    #[test]
    fn get_valid_adjacencies() {
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

        let state = State::with_indexes([0, 1], 3);

        assert_eq!(
            valid_adjacencies_from_state(&state, &valid_adjacencies_map),
            [
                AxisPair::new(State::with_index(1, 3), State::fill(true, 3)),
                AxisPair::new(State::fill(true, 3), State::with_index(1, 3)),
            ]
        );
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
    fn test_validate_solution() {
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

        let matrix = Matrix::from_with_dimensions(
            [2, 2],
            [
                State::with_index(0, 3),
                State::with_index(1, 3),
                State::with_index(2, 3),
                State::with_index(0, 3),
            ],
        );

        assert!(validate_solution(&matrix, &valid_adjacencies_map));
    }
}
