use tiled_wfc::{
    validation::{valid_adjacencies_map, validate_matrix_state},
    AxisPair, Tile, Weighted, WFC,
};

use Socket::*;

const TILE_SET: &[CharTile] = &[
    CharTile::new(' ', Empty, Empty, Empty, Empty),
    CharTile::new('─', Normal, Normal, Empty, Empty),
    CharTile::new('━', Bold, Bold, Empty, Empty),
    CharTile::new('│', Empty, Empty, Normal, Normal),
    CharTile::new('┃', Empty, Empty, Bold, Bold),
    CharTile::new('┄', Normal, Normal, Empty, Empty),
    CharTile::new('┅', Bold, Bold, Empty, Empty),
    CharTile::new('┆', Empty, Empty, Normal, Normal),
    CharTile::new('┇', Empty, Empty, Bold, Bold),
    CharTile::new('┈', Normal, Normal, Empty, Empty),
    CharTile::new('┉', Bold, Bold, Empty, Empty),
    CharTile::new('┊', Empty, Empty, Normal, Normal),
    CharTile::new('┋', Empty, Empty, Bold, Bold),
    CharTile::new('┌', Normal, Empty, Empty, Normal),
    CharTile::new('┏', Bold, Empty, Empty, Bold),
    CharTile::new('┐', Empty, Normal, Empty, Normal),
    CharTile::new('┓', Empty, Bold, Empty, Bold),
    CharTile::new('└', Normal, Empty, Normal, Empty),
    CharTile::new('┗', Bold, Empty, Bold, Empty),
    CharTile::new('┘', Empty, Normal, Normal, Empty),
    CharTile::new('┛', Empty, Bold, Bold, Empty),
    CharTile::new('├', Normal, Empty, Normal, Normal),
    CharTile::new('┣', Bold, Empty, Bold, Bold),
    CharTile::new('┤', Empty, Normal, Normal, Normal),
    CharTile::new('┫', Empty, Bold, Bold, Bold),
    CharTile::new('┬', Normal, Normal, Empty, Normal),
    CharTile::new('┳', Bold, Bold, Empty, Bold),
    CharTile::new('┴', Normal, Normal, Normal, Empty),
    CharTile::new('┻', Bold, Bold, Bold, Empty),
    CharTile::new('┼', Normal, Normal, Normal, Normal),
    CharTile::new('╋', Bold, Bold, Bold, Bold),
    CharTile::new('╌', Normal, Normal, Empty, Empty),
    CharTile::new('╍', Bold, Bold, Empty, Empty),
    CharTile::new('╎', Empty, Empty, Normal, Normal),
    CharTile::new('╏', Empty, Empty, Bold, Bold),
    CharTile::new('═', Double, Double, Empty, Empty),
    CharTile::new('║', Empty, Empty, Double, Double),
    CharTile::new('╔', Double, Empty, Empty, Double),
    CharTile::new('╗', Empty, Double, Empty, Double),
    CharTile::new('╚', Double, Empty, Double, Empty),
    CharTile::new('╝', Empty, Double, Double, Empty),
    CharTile::new('╠', Double, Empty, Double, Double),
    CharTile::new('╣', Empty, Double, Double, Double),
    CharTile::new('╦', Double, Double, Empty, Double),
    CharTile::new('╩', Double, Double, Double, Empty),
    CharTile::new('╬', Double, Double, Double, Double),
    CharTile::new('╭', Normal, Empty, Empty, Normal),
    CharTile::new('╮', Empty, Normal, Empty, Normal),
    CharTile::new('╯', Empty, Normal, Normal, Empty),
    CharTile::new('╰', Normal, Empty, Normal, Empty),
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

impl Weighted for CharTile {
    fn weight(&self) -> u32 {
        0
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Socket {
    Empty,
    Normal,
    Bold,
    Double,
}

#[test]
fn char_tile() {
    let mut wfc = WFC::builder()
        .tile_set(TILE_SET)
        .dimensions([80, 40])
        .seed_from_u64(422)
        .build()
        .unwrap();

    wfc.collapse().expect("solution");

    let lines = wfc.matrix().matrix().chunks(wfc.dimensions()[0]).rev();

    for states in lines {
        let line: String = states
            .into_iter()
            .map(|state| wfc.get_tile(state).map_or(' ', |tile| tile.character))
            .collect();

        println!("{}", line);
    }

    assert!(validate_matrix_state(
        wfc.matrix(),
        &valid_adjacencies_map(TILE_SET)
    ));
}
