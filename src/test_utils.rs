use nd_matrix::AxisPair;

use crate::Tile;

pub const TILE_SET: &[TestTile] = &[
    TestTile::new([AxisPair::new('a', 'b'), AxisPair::new('b', 'a')]),
    TestTile::new([AxisPair::new('a', 'a'), AxisPair::new('a', 'a')]),
    TestTile::new([AxisPair::new('b', 'b'), AxisPair::new('b', 'b')]),
];

#[derive(Debug)]
pub struct TestTile {
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
