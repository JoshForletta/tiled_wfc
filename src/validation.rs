use std::array::from_fn;

use nd_matrix::Matrix;

use crate::{state::Superposition, AxisPair, State, Tile};

pub type Adjacencies<T, const D: usize> = [AxisPair<T>; D];

/// Returns all valid adjacent states of `tile` given `tile_set`.
///
/// # Panics
///
/// if `tile_set.len()` exceeds `Superposition::MAX`.
pub fn valid_adjacencies_from_tile<T, const D: usize>(
    tile: &T,
    tile_set: &[T],
) -> Adjacencies<Superposition, D>
where
    T: Tile<D>,
{
    let mut valid_adjacencies: [AxisPair<Superposition>; D] = from_fn(|_| AxisPair::default());

    let sockets = tile.sockets();

    for (state, possible_tile) in tile_set.into_iter().enumerate() {
        for dimension in 0..D {
            let possible_sockets = possible_tile.sockets();

            if sockets[dimension].pos == possible_sockets[dimension].neg {
                valid_adjacencies[dimension].pos.insert_state(state);
            }

            if sockets[dimension].neg == possible_sockets[dimension].pos {
                valid_adjacencies[dimension].neg.insert_state(state);
            }
        }
    }

    valid_adjacencies
}

/// Returns all valid adjacent states of `state` given `valid_adjacencies_map`.
///
/// # Panics
///
/// if `state` as `State::Collapsed` exceeds `Superposition::MAX`.
pub fn valid_adjacencies_from_state<const D: usize>(
    state: &State,
    valid_adjacencies_map: &Vec<Adjacencies<Superposition, D>>,
) -> Adjacencies<Superposition, D> {
    let mut cumulative_valid_adjacencies: [AxisPair<Superposition>; D] =
        from_fn(|_| AxisPair::default());

    for state in state.into_iter() {
        let valid_adjacencies = &valid_adjacencies_map[state];

        for dimension in 0..D {
            cumulative_valid_adjacencies[dimension]
                .pos
                .insert_superposition(&valid_adjacencies[dimension].pos);
            cumulative_valid_adjacencies[dimension]
                .neg
                .insert_superposition(&valid_adjacencies[dimension].neg);
        }
    }

    cumulative_valid_adjacencies
}

/// Returns a map of valid adjacent states for `tile_set`.
///
/// # Panics
///
/// if `tile_set.len()` exceeds `Superposition::MAX`.
pub fn valid_adjacencies_map<T, const D: usize>(
    tile_set: &[T],
) -> Vec<Adjacencies<Superposition, D>>
where
    T: Tile<D>,
{
    tile_set
        .into_iter()
        .map(|tile| valid_adjacencies_from_tile(tile, tile_set))
        .collect()
}

/// Return `true` if `matrix` is in a valid state given `valid_adjacencies_map`.
///
/// # Panics
///
/// if state from matrix as `State::Collapsed` exceeds `Superposition::MAX`.
pub fn validate_matrix_state<const D: usize>(
    matrix: &Matrix<State, D>,
    valid_adjacencies_map: &Vec<Adjacencies<Superposition, D>>,
) -> bool {
    for (index, state) in matrix.into_iter().enumerate() {
        let valid_adjacencies = valid_adjacencies_from_state(state, &valid_adjacencies_map);
        let adjacencies = matrix.get_adjacencies(index);

        for dimension in 0..D {
            let valid_adjacency_pair = &valid_adjacencies[dimension];
            let adjacency_pair = adjacencies[dimension];

            if let Some(adjacency) = adjacency_pair.pos {
                match adjacency {
                    State::Collapsed(adjacency) => {
                        if !valid_adjacency_pair.pos.contains_state(*adjacency) {
                            return false;
                        }
                    }
                    State::Superimposed(adjacency) => {
                        if !valid_adjacency_pair.pos.contains_superposition(adjacency) {
                            return false;
                        }
                    }
                }
            }

            if let Some(adjacency) = adjacency_pair.neg {
                match adjacency {
                    State::Collapsed(adjacency) => {
                        if !valid_adjacency_pair.neg.contains_state(*adjacency) {
                            return false;
                        }
                    }
                    State::Superimposed(adjacency) => {
                        if !valid_adjacency_pair.neg.contains_superposition(adjacency) {
                            return false;
                        }
                    }
                }
            }
        }
    }

    true
}

#[cfg(test)]
mod tests {
    use nd_matrix::{AxisPair, Matrix};

    use crate::{
        state::Superposition, test_utils::TILE_SET, validation::valid_adjacencies_map, State,
    };

    #[test]
    fn valid_adjacencies_from_tile() {
        let tile = &TILE_SET[0];

        let output = [
            AxisPair::new(
                Superposition::from_iter([false, true, false]),
                Superposition::from_iter([false, false, true]),
            ),
            AxisPair::new(
                Superposition::from_iter([false, false, true]),
                Superposition::from_iter([false, true, false]),
            ),
        ];

        assert_eq!(super::valid_adjacencies_from_tile(tile, TILE_SET), output)
    }

    #[test]
    fn valid_adjacencies_from_state() {
        let valid_adjacencies_map = valid_adjacencies_map(TILE_SET);

        let state = State::Superimposed(Superposition::from_iter([true, true, false]));

        assert_eq!(
            super::valid_adjacencies_from_state(&state, &valid_adjacencies_map),
            [
                AxisPair::new(
                    Superposition::from_iter([false, true, false]),
                    Superposition::from_iter([true, true, true])
                ),
                AxisPair::new(
                    Superposition::from_iter([true, true, true]),
                    Superposition::from_iter([false, true, false])
                ),
            ]
        );

        let state = State::Collapsed(2);

        assert_eq!(
            super::valid_adjacencies_from_state(&state, &valid_adjacencies_map),
            [
                AxisPair::new(
                    Superposition::from_iter([true, false, true]),
                    Superposition::from_iter([false, false, true]),
                ),
                AxisPair::new(
                    Superposition::from_iter([false, false, true]),
                    Superposition::from_iter([true, false, true]),
                ),
            ]
        )
    }

    #[test]
    fn test_valid_adjacencies_map() {
        let output = Vec::from([
            [
                AxisPair::new(
                    Superposition::from_iter([false, true, false]),
                    Superposition::from_iter([false, false, true]),
                ),
                AxisPair::new(
                    Superposition::from_iter([false, false, true]),
                    Superposition::from_iter([false, true, false]),
                ),
            ],
            [
                AxisPair::new(
                    Superposition::from_iter([false, true, false]),
                    Superposition::from_iter([true, true, false]),
                ),
                AxisPair::new(
                    Superposition::from_iter([true, true, false]),
                    Superposition::from_iter([false, true, false]),
                ),
            ],
            [
                AxisPair::new(
                    Superposition::from_iter([true, false, true]),
                    Superposition::from_iter([false, false, true]),
                ),
                AxisPair::new(
                    Superposition::from_iter([false, false, true]),
                    Superposition::from_iter([true, false, true]),
                ),
            ],
        ]);

        assert_eq!(super::valid_adjacencies_map(TILE_SET), output);
    }

    #[test]
    fn test_validate_solution() {
        let valid_adjacencies_map = valid_adjacencies_map(TILE_SET);

        let matrix = Matrix::from_with_dimensions(
            [2, 2],
            [
                State::Superimposed(Superposition::from_iter([true, false, false])),
                State::Superimposed(Superposition::from_iter([false, true, false])),
                State::Superimposed(Superposition::from_iter([false, false, true])),
                State::Superimposed(Superposition::from_iter([true, false, false])),
            ],
        );

        assert!(super::validate_matrix_state(
            &matrix,
            &valid_adjacencies_map
        ));
    }
}
