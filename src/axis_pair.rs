#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct AxisPair<T> {
    pub pos: T,
    pub neg: T,
}

impl<T> AxisPair<T> {
    pub const fn new(pos: T, neg: T) -> Self {
        Self { pos, neg }
    }
}
