use crate::AxisPair;

pub trait Tile<const D: usize>
where
    <Self as Tile<D>>::Socket: PartialEq,
{
    type Socket;

    fn sockets(&self) -> [AxisPair<<Self as Tile<D>>::Socket>; D];
}

pub trait Weighted {
    fn weight(&self) -> u32;
}
