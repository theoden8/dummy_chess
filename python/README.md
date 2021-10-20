# python bindings

The idea is to interface this engine as an environment.

## building

```bash
mkdir -p build
cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
```

There should be a `dummy_chess.so` file. Simply import it as `dummy_chess` module.

## performance

Depending on the usage, it currently caps at 80% of the UCI binary.

## usage

```python
#!/usr/bin/env python3


from pprint import pprint
import dummy_chess


if __name__ == "__main__":
    chess = dummy_chess.ChessDummy('1r4k1/1r3pp1/3b3p/3p1qnP/Q1pP3R/2P2PP1/PP4K1/R1B3N1 b - - 2 24')
    print('fen', chess.fen)

    # fen 1r4k1/1r3pp1/3b3p/3p1qnP/Q1pP3R/2P2PP1/PP4K1/R1B3N1 b - - 2 24
    print('legal moves', chess.legal_moves)
    # legal moves ['Qb1', 'Qc2+', 'Qd3', 'Qxf3+', 'Qh3+', 'Qe4', 'Qf4', 'Qg4', 'Qe5', 'Qe6', 'Qf6', 'Qg6', 'Qd7', 'Qh7', 'Qc8', 'Nxf3', 'Nh3', 'Ne4', 'Ne6', 'Nh7', 'Ba3', 'Bxg3', 'Bb4', 'Bf4', 'Bc5', 'Be5', 'Bc7', 'Be7', 'Bf8', 'Rxb2+', 'Rb3', 'Rb4', 'Rb5', 'Rb6', 'Ra7', 'Rc7', 'Rd7', 'Re7', 'f6', 'g6', 'Ra8', 'Rc8', 'Rd8', 'Re8', 'Rf8', 'Kh7', 'Kf8', 'Kh8']
    random_move = chess.sample()
    print('random move', random_move)
    # random move Qg4
    chess.step(random_move)
    print('new fen', chess.fen)
    chess.undo()
    print('status', chess.status)
    # status ONGOING
    print('fixed depth move', chess.get_depth_move(3))
    # fixed depth move ('Rxb2+', -1.728906273841858)
    def func(info: dict) -> bool:
        if info['important']:
            for k in info.keys():
                if type(info[k]) == float:
                    info[k] = f'%.3f' % info[k]
            pprint(info)
        return info['depth'] < 12
    chess.start_thinking(func)
    # ...
```
