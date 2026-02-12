# dummy chess

<center>
<a href="https://github.com/theoden8/dummy_chess/actions">
    <img src="https://github.com/theoden8/dummy_chess/actions/workflows/ci.yml/badge.svg" alt="GitHub CI"/>
</a>
<a href="https://ci.codeberg.org/repos/13782">
    <img src="https://ci.codeberg.org/api/badges/13782/status.svg?branch=master" alt="Codeberg CI"/>
</a>
<a href="https://gitlab.com/theoden8/dummy_chess/-/commits/master">
    <img src="https://gitlab.com/theoden8/dummy_chess/badges/master/pipeline.svg" alt="Gitlab CI"/>
</a>
<a href="https://lichess.org/@/theoden8_uci">
    <img src="https://img.shields.io/badge/Lichess-theoden8__uci-blue?logo=lichess" alt="Challenge on Lichess"/>
</a>
</center>

* [**Source Code**](https://codeberg.org/theoden8/dummy_chess)

## building

```
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build
```

## features

* fail-soft alpha-beta, iddfs, transposition tables, pvsearch
* null-move reductions, late-move reductions, delta pruning
* syzygy endgame tablebases
* uci partially supported (pondering, hash table size, variants)
* variants:
    - standard
    - chess960
    - crazyhouse

## tools

* c++-20
* [ncurses](https://invisible-island.net/ncurses/announce.html) interface
* [m42](https://github.com/sinandredemption/M42) library for magic bitboards
* [syzygy](https://github.com/syzygy1/tb) tool to generate tablebases
* [fathom](https://github.com/jdart1/Fathom) c++ interface to syzygy tablebases

## external links

* https://www.chessprogramming.org/Bitboards - inspiration for performance
* https://graphics.stanford.edu/~seander/bithacks.html - helps to implement bit-boards

## tipping

[<img src="https://www.getmonero.org/press-kit/symbols/monero-symbol-480.png" alt="xmr" height="20" width="20">](https://getmonero.org) [XMR: 8B2g8b87KeuGpwvqYTDFLwKfqy35NRJstHabUa5GLmpB54ecwoKU44tgwLditAoHgW2Mw8H4a281GYi7JaFsPtQs9yZL7Sr](monero:8B2g8b87KeuGpwvqYTDFLwKfqy35NRJstHabUa5GLmpB54ecwoKU44tgwLditAoHgW2Mw8H4a281GYi7JaFsPtQs9yZL7Sr)
