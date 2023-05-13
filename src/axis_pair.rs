#[derive(Debug, Default, Clone, Copy, PartialEq)]
pub struct AxisPair<T> {
    pub pos: T,
    pub neg: T,
}

impl<T> AxisPair<T> {
    pub const fn new(pos: T, neg: T) -> Self {
        Self { pos, neg }
    }

    pub fn iter(&self) -> <&Self as IntoIterator>::IntoIter {
        self.into_iter()
    }

    pub fn iter_mut(&mut self) -> <&mut Self as IntoIterator>::IntoIter {
        self.into_iter()
    }
}

impl<T> IntoIterator for AxisPair<T> {
    type Item = <[T; 2] as IntoIterator>::Item;
    type IntoIter = <[T; 2] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        [self.pos, self.neg].into_iter()
    }
}

impl<'a, T> IntoIterator for &'a AxisPair<T> {
    type Item = <[&'a T; 2] as IntoIterator>::Item;
    type IntoIter = <[&'a T; 2] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        [&self.pos, &self.neg].into_iter()
    }
}

impl<'a, T> IntoIterator for &'a mut AxisPair<T> {
    type Item = <[&'a mut T; 2] as IntoIterator>::Item;
    type IntoIter = <[&'a mut T; 2] as IntoIterator>::IntoIter;

    fn into_iter(self) -> Self::IntoIter {
        [&mut self.pos, &mut self.neg].into_iter()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn into_iter() {
        let axis_pair = AxisPair::new(6, 9);

        let mut axis_pair_iter = axis_pair.into_iter();

        assert_eq!(axis_pair_iter.next(), Some(6));
        assert_eq!(axis_pair_iter.next(), Some(9));
        assert_eq!(axis_pair_iter.next(), None);
    }

    #[test]
    fn iter() {
        let axis_pair = AxisPair::new(6, 9);

        let mut axis_pair_iter = axis_pair.iter();

        assert_eq!(axis_pair_iter.next(), Some(&6));
        assert_eq!(axis_pair_iter.next(), Some(&9));
        assert_eq!(axis_pair_iter.next(), None);
    }

    #[test]
    fn iter_mut() {
        let mut axis_pair = AxisPair::new(6, 9);

        let mut axis_pair_iter = axis_pair.iter_mut();

        assert_eq!(axis_pair_iter.next(), Some(&mut 6));
        assert_eq!(axis_pair_iter.next(), Some(&mut 9));
        assert_eq!(axis_pair_iter.next(), None);
    }
}
