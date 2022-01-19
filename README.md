# dummy chess

* [**Source Code**](https://codeberg.org/theoden8/dummy_chess)
* [**Challenge it**](https://lichess.org/@/theoden8_uci)

## building

```
$ make
```

## features

* fail-soft alpha-beta, iddfs, transposition tables, pvsearch
* null-move reductions, late-move reductions, delta pruning
* uci partially supported (pondering, hash table size)
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

## tipping

[<img src="https://www.getmonero.org/press-kit/symbols/monero-symbol-480.png" alt="xmr" height="20" width="20">](https://getmonero.org) XMR: 8B2g8b87KeuGpwvqYTDFLwKfqy35NRJstHabUa5GLmpB54ecwoKU44tgwLditAoHgW2Mw8H4a281GYi7JaFsPtQs9yZL7Sr
