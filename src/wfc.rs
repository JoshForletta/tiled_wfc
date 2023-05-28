use nd_matrix::{Matrix, ToIndex};
use rand::{rngs::StdRng, Rng, SeedableRng};

use crate::{
    state::Superposition,
    validation::{valid_adjacencies_from_state, valid_adjacencies_map, Adjacencies},
    Collapser, State, StateError, Tile, UnweightedCollapser,
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
    valid_adjacencies_map: Vec<Adjacencies<Superposition, D>>,
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
    pub fn tile_set(&self) -> &[T] {
        self.tile_set
    }

    #[inline(always)]
    pub fn valid_adjacencies_map(&self) -> &Vec<Adjacencies<Superposition, D>> {
        &self.valid_adjacencies_map
    }

    #[inline(always)]
    pub fn matrix(&self) -> &Matrix<State, D> {
        &self.matrix
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

    /// Returns the index of the least enstopic state in the matrix. Returns
    /// [`None`] if all states are collapsed.
    pub fn least_entropic_index(&self) -> Option<usize> {
        self.matrix()
            .into_iter()
            .enumerate()
            .filter(|(_, state)| !state.is_collapsed())
            .map(|(index, state)| (index, state.count()))
            .min_by(|(_, min_count), (_, count)| min_count.cmp(count))
            .map(|(index, _)| index)
    }

    /// Collapses all states in `matrix`.
    ///
    /// # Errors
    ///
    /// if there is no viable solution.
    pub fn collapse(&mut self) -> Result<(), StateError> {
        let mut collapse_records = Vec::with_capacity(self.matrix.len());

        while let Some(index) = self.least_entropic_index() {
            match self.collapse_state(index) {
                Ok(collapse_record) => collapse_records.push(collapse_record),
                Err(_) => loop {
                    let collapse_record =
                        collapse_records.pop().ok_or(StateError::NoViableState)?;

                    if self.uncollapse_state(collapse_record).is_ok() {
                        break;
                    }
                },
            };
        }

        Ok(())
    }

    /// Collapses state at `index` returning a [`CollapseRecord`].
    ///
    /// # Errors
    ///
    /// - if `collapser` errors.
    /// - if collapse results in a state with no viable state after propagation.
    pub fn collapse_state(&mut self, index: usize) -> Result<CollapseRecord, StateError> {
        let initial_state = self.matrix[index].clone();

        self.collapser.collapse(&mut self.matrix[index])?;

        let propagation_records = match self.propagate(index) {
            Ok(propagation_record) => propagation_record,
            Err(e) => {
                self.matrix[index] = initial_state;
                return Err(e);
            }
        };

        let mut remaining_state = initial_state;
        let collapsed_state = *self.matrix[index]
            .collapsed()
            .expect("state after collapse to be `State::Collapsed`");

        remaining_state
            .superimposed_mut()
            .expect("state at `index` to be superimposed")
            .remove_state(collapsed_state);

        Ok(CollapseRecord {
            index,
            remaining_state,
            propagation_records,
        })
    }

    /// Unpropagates and uncollapses states in `collapse_record`.
    ///
    /// # Errors
    ///
    /// - if `remaining_state` has no viable state.
    pub fn uncollapse_state(
        &mut self,
        CollapseRecord {
            index,
            remaining_state,
            mut propagation_records,
        }: CollapseRecord,
    ) -> Result<(), StateError> {
        self.matrix[index] = remaining_state;

        self.unpropagate(&mut propagation_records);

        if self.matrix[index].count() == 0 {
            Err(StateError::NoViableState)
        } else {
            Ok(())
        }
    }

    /// Recusively propagates constraints, starting at `index`.
    ///
    /// # Errors
    ///
    /// - if propagation results in a superposition with no viable state.
    /// - if any state adjacent to `index` is [`State::Collapsed`] and not
    /// contained within `valid_states`.
    fn propagate(&mut self, index: usize) -> Result<Vec<PropagationRecord>, StateError> {
        let mut propagation_stack = Vec::from([index]);
        let mut propagation_records = Vec::new();

        while let Some(index) = propagation_stack.pop() {
            self.propagate_adjacent_states(
                &mut propagation_stack,
                &mut propagation_records,
                index,
            )?;
        }

        Ok(propagation_records)
    }

    /// Constrains all states adjacent to `index`. On error this function
    /// unpropagates, clearing `propagation_stack` and `propagation_records`.
    ///
    /// # Errors
    ///
    /// - if propagation results in a superposition with no viable state.
    /// - if any state adjacent to `index` is [`State::Collapsed`] and not
    /// contained within `valid_states`.
    ///
    /// # Panics
    ///
    /// if `index` is out of bounds.
    fn propagate_adjacent_states(
        &mut self,
        propagation_stack: &mut Vec<usize>,
        propagation_records: &mut Vec<PropagationRecord>,
        index: usize,
    ) -> Result<(), StateError> {
        let valid_adjacencies =
            &valid_adjacencies_from_state(&self.matrix[index], &self.valid_adjacencies_map);
        let adjacent_indexes = self.matrix.get_adjacent_indexes(index);

        for (adjacent_index_pair, valid_adjacency_pair) in
            adjacent_indexes.into_iter().zip(valid_adjacencies)
        {
            for (adjacent_index, valid_adjacency) in adjacent_index_pair
                .into_iter()
                .zip(valid_adjacency_pair)
                .filter(|(adjacent_index, _)| adjacent_index.is_some())
                .map(|(adjacent_index, valid_adjacency)| (adjacent_index.unwrap(), valid_adjacency))
            {
                self.propagate_state(
                    propagation_stack,
                    propagation_records,
                    adjacent_index,
                    valid_adjacency,
                )?;
            }
        }

        Ok(())
    }

    /// Constrains state at `index` with `valid_states`, pushing the record
    /// onto `propagation_records` and the index onto `propagation_stack`. On
    /// Error this function unpropagates, clearing the `propagation_stack` and
    /// `propagation_records`.
    ///
    /// # Errors
    ///
    /// - if propagation results in a superposition with no viable state.
    /// - if state at `index` is [`State::Collapsed`] and not contained within
    /// `valid_states`.
    ///
    /// # Panics
    ///
    /// if `index` is out of bounds.
    fn propagate_state(
        &mut self,
        propagation_stack: &mut Vec<usize>,
        propagation_records: &mut Vec<PropagationRecord>,
        index: usize,
        valid_states: &Superposition,
    ) -> Result<(), StateError> {
        let adjacency = &mut self.matrix[index];

        match adjacency {
            State::Collapsed(state) if !valid_states.contains_state(*state) => {
                return Err(StateError::NoViableState);
            }
            State::Superimposed(state) if !valid_states.contains_superposition(state) => {
                let initial_state = state.clone();

                state.constrain(valid_states);

                if state.count() == 0 {
                    *state = initial_state;

                    self.unpropagate(propagation_records);
                    propagation_stack.clear();

                    return Err(StateError::NoViableState);
                }

                propagation_records.push(PropagationRecord {
                    index,
                    state: State::Superimposed(initial_state),
                });
                propagation_stack.push(index);
            }
            _ => (),
        }

        Ok(())
    }

    /// Unpropagates states in `propagation_records`, leaving it empty.
    fn unpropagate(&mut self, propagation_records: &mut Vec<PropagationRecord>) {
        while let Some(propagation_record) = propagation_records.pop() {
            self.matrix[propagation_record.index] = propagation_record.state;
        }
    }
}

#[derive(Debug, Clone)]
pub struct CollapseRecord {
    index: usize,
    remaining_state: State,
    propagation_records: Vec<PropagationRecord>,
}

#[derive(Debug, Clone)]
struct PropagationRecord {
    index: usize,
    state: State,
}

#[cfg(test)]
mod tests {
    use rand::rngs::mock::StepRng;

    use crate::{test_utils::TILE_SET, validation::validate_matrix_state};

    use super::*;

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
        let wfc = WFC {
            tile_set: TILE_SET,
            valid_adjacencies_map: valid_adjacencies_map(TILE_SET),
            matrix: Matrix::from_with_dimensions(
                [2, 2],
                [
                    // Not valid matrix state
                    State::Collapsed(0),
                    State::from_iter([false, true, false]),
                    State::from_iter([false, true, true]),
                    State::fill(3),
                ],
            ),
            collapser: UnweightedCollapser::new(StepRng::new(0, 0)),
        };

        assert_eq!(wfc.least_entropic_index(), Some(1));
    }

    #[test]
    fn least_entropic_index_collapsed() {
        let wfc = WFC {
            tile_set: TILE_SET,
            valid_adjacencies_map: valid_adjacencies_map(TILE_SET),
            matrix: Matrix::from_with_dimensions(
                [2, 2],
                [
                    State::Collapsed(0),
                    State::Collapsed(1),
                    State::Collapsed(2),
                    State::Collapsed(0),
                ],
            ),
            collapser: UnweightedCollapser::new(StepRng::new(0, 0)),
        };

        assert_eq!(wfc.least_entropic_index(), None);
    }

    #[test]
    fn collapse() {
        let mut wfc = WFC::builder()
            .tile_set(TILE_SET)
            .dimensions([2, 2])
            .collapser(UnweightedCollapser::new(StepRng::new(0, 0)))
            .build();

        assert!(wfc.collapse().is_ok());

        assert!(validate_matrix_state(
            wfc.matrix(),
            wfc.valid_adjacencies_map()
        ))
    }
}
