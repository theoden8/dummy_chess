extern crate dummy_chess;

use root::*;
use dummy_chess::*;

fn main() {
  let starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  let fen: FEN = FEN::new(starting_pos);
  println!("fen: {}", fen.str());
  unsafe {
    let board = Board::new(&fen.raw(), 1usize << 20);
    board.print();
  };
  println!("EXIT");
}
