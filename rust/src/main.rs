extern crate dummy_chess;


use dummy_chess::*;


fn main() {
  let starting_pos = String::from("rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1");
  let fen: FEN = FEN::new(&starting_pos);
  println!("fen: {}", fen.str());
  let mut chess = Chess::new(&fen);
  chess.make_move_pgn(&String::from("e4"));
  chess.make_move_pgn(&String::from("e5"));
  chess.show();
  for i in 0..3 {
    let peval = chess.start_thinking_depth(i);
    println!("---------------------------------------------------------------------------------");
    println!("{} | {}", peval.eval, peval.mline.str(&mut chess));
    println!("---------------------------------------------------------------------------------");
  }
  println!("EXIT");
}

