use rand::Rng;

pub trait Collapser {
    fn collapse<I, R>(states: I, rng: &mut R) -> Result<usize, ()>
    where
        I: Iterator<Item = usize>,
        R: Rng;
}
