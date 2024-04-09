extern crate dummy_chess;

use dummy_chess::*;

fn main() {
  println!("MAIN");
  let starting_pos = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1";
  let fen: FEN = FEN::new(starting_pos);
  println!("fen: {}", fen.str());
  println!("EXIT");
}
