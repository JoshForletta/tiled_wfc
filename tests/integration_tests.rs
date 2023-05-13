use tiled_wfc::{AxisPair, Tile, WFC};

use Socket::*;

const TILE_SET: &[CharTile] = &[
    CharTile::new('─', Normal, Normal, Empty, Empty),
    CharTile::new('│', Empty, Empty, Normal, Normal),
    CharTile::new('┌', Normal, Empty, Empty, Normal),
    CharTile::new('┐', Empty, Normal, Empty, Normal),
    CharTile::new('└', Normal, Empty, Normal, Empty),
    CharTile::new('┘', Empty, Normal, Normal, Empty),
];

#[derive(Debug, Clone, Copy)]
pub struct CharTile {
    pub character: char,
    pub sockets: [AxisPair<Socket>; 2],
}

impl CharTile {
    pub const fn new(
        character: char,
        pos_x: Socket,
        neg_x: Socket,
        pos_y: Socket,
        neg_y: Socket,
    ) -> Self {
        Self {
            character,
            sockets: [AxisPair::new(pos_x, neg_x), AxisPair::new(pos_y, neg_y)],
        }
    }
}

impl Tile<2> for CharTile {
    type Socket = Socket;

    fn sockets(&self) -> [AxisPair<Self::Socket>; 2] {
        self.sockets
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Socket {
    Empty,
    Normal,
}

#[test]
fn char_tile() {
    let mut wfc = WFC::builder()
        .tile_set(TILE_SET)
        .dimensions([20, 10])
        .seed(69)
        .build()
        .unwrap();

    wfc.collapse().unwrap();

    let lines = wfc.matrix().matrix().chunks(wfc.dimensions()[0]).rev();

    for states in lines {
        let line: String = states
            .into_iter()
            .map(|state| wfc.get_tile(state).map_or(' ', |tile| tile.character))
            .collect();

        println!("{}", line);
    }
}
