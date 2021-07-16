# dummy chess

* [**Source Code**](https://codeberg.org/theoden8/dummy_chess)
* [**Challenge it**](https://lichess.org/@/theoden8_uci)

## building

```
$ make
```

## features

* alpha-beta, iddfs, transposition tables, pvsearch
* uci partially supported
* variants:
    - standard
    - chess960
    - crazyhouse

## tools

* c++-20
* [m42](https://github.com/sinandredemption/M42) library for magic bitboards
* ncurses (tui)

## external links

* https://www.chessprogramming.org/Bitboards
* https://graphics.stanford.edu/~seander/bithacks.html - helps to implement bit-boards
* https://www.chessprogramming.org/Stockfish - list of features
