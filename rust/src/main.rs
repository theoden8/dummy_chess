extern crate dummy_chess;


use dummy_chess::*;


fn main() {
  let starting_pos = String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  let fen: FEN = FEN::new(&starting_pos);
  println!("fen: {}", fen.str());
  let mut board = Board::new(&fen);
  board.show();
  board.make_move_pgn(&String::from("e2e4"));
  board.show();
  board.make_move_pgn(&String::from("e5"));
  board.show();
  println!("EXIT");
}

