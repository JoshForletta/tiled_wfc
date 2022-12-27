use rand::Rng;

use crate::Collapser;

pub struct State;
impl State {
    pub fn collapse<C, R>(&self, collapser: &C, rng: &mut R) -> Result<(), ()>
    where
        C: Collapser,
        R: Rng,
    {
        todo!()
    }
}
