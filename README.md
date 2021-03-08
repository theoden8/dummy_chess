# dummy chess

## building

```
$ make
```

## run it

```
./dummy_chess_playouts [depth=4] [fen=starting_pos]
```

and copy the resulting PGNs into lichess.com's analysis board

## tools

* c++-20
* [m42](https://github.com/sinandredemption/M42) library for magic bitboards
* ncurses (tui)

## external links

* https://www.chessprogramming.org/Bitboards
* https://graphics.stanford.edu/~seander/bithacks.html - helps to implement bit-boards
* https://www.chessprogramming.org/Stockfish - list of features