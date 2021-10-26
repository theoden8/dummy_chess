#pragma once


#include <array>
#include <vector>
#include <stack>
#include <bitset>

#include <Piece.hpp>
#include <Board.hpp>


namespace nn {

using scalar_acc_t = int32_t;
using scalar_eval_t = int8_t;
template <typename T, size_t N> using vec_t = std::array<T, N>;

template <typename IN_T, size_t IN, typename OUT_T, size_t OUT> struct Affine {
  template <typename T, size_t N, size_t M>
  using weight_mat_t = std::array<T, N*M>;

  // forward
//  vec_t<IN_T, IN> x;
  weight_mat_t<IN_T, IN, OUT> A;
  vec_t<OUT_T, OUT> b;
  vec_t<OUT_T, OUT> y;
  // backprop
//  mat_t<IN, OUT> dLdA;
//  vec_t<OUT> dLdx;

  Affine()
  {}

  const vec_t<OUT_T, OUT> &forward(const vec_t<IN_T, IN> &x) {
//    x = _x;
    for(size_t i = 0; i < OUT; ++i) {
      OUT_T dotprod = b[i];
      for(size_t j = 0; j < IN; ++j) {
        dotprod += OUT_T(A[i * IN + j]) * OUT_T(x[j]);
      }
      y[i] = dotprod;
    }
    return y;
  }

//  const vec_t<IN> &backward(const vec_t<OUT> &dLdy) override {
//    // dydx = A^T
//    // dLdx = dLdy . dydx
//    for(size_t i = 0; i < IN; ++i) {
//      float dotprod = .0;
//      for(size_t j = 0; j < OUT; ++j) {
//        // A is transposed
//        dotprod += dLdy[j] * A[j * IN + i];
//      }
//      dLdx[i] = dotprod;
//    }
//    // dLdW = x^T . dLdy
//    for(size_t i = 0; i < IN; ++i) {
//      for(size_t j = 0; j < OUT; ++j) {
//        dLdA[i * OUT + j] = x[i] * dLdy[j];
//      }
//    }
//    return dLdx;
//  }

//  void apply_gradient(float step_size) override {
//    for(size_t i = 0; i < IN; ++i) {
//      for(size_t j = 0; j < OUT; ++j) {
//        A[i * OUT + j] += step_size * dLdA[i * OUT + j];
//      }
//    }
//  }
};

template <typename IN_T, size_t N, typename OUT_T> struct ClippedReLU {
  // forward
//  vec_t<IN_T, N> x;
  vec_t<OUT_T, N> y;
  // backprop
//  vec_t<N> dLdx;

  ClippedReLU()
  {}

  INLINE static OUT_T activate(IN_T val) {
    return std::min<IN_T>(std::max<IN_T>(val, 0), 127);
  }

  const vec_t<OUT_T,N> &forward(const vec_t<IN_T,N> &x) {
//    x = _x;
    for(size_t i = 0; i < N; ++i) {
      y[i] = activate(x[i]);
    }
    return y;
  }

//  const vec_t<N> &backward(const vec_t<N> &dLdy) override {
//    // dydx = [x < 0] * [x > 0]
//    for(size_t i = 0; i < N; ++i) {
//      float dydx = (x[i] > 0 && x[i] < 1) ? 1 : 0;
//      dLdx[i] = dLdy[i] * dydx;
//    }
//    return dLdx;
//  }

//  void apply_gradient(float step_size) override {
//    ;
//  }
};

template <typename IN_T, size_t N, typename OUT_T> struct NNUEReLU {
  // forward
//  vec_t<IN_T, N> x;
  vec_t<OUT_T, N> y;
  // backprop
//  vec_t<N> dLdx;

  NNUEReLU()
  {}

  INLINE static OUT_T activate(IN_T val) {
    return std::max<OUT_T>(std::min<OUT_T>(val / 64, 127), 0);
  }

  const vec_t<OUT_T,N> &forward(const vec_t<IN_T,N> &x) {
//    x = _x;
    for(size_t i = 0; i < N; ++i) {
      y[i] = activate(x[i]);
    }
    return y;
  }

//  const vec_t<N> &backward(const vec_t<N> &dLdy) override {
//    // dydx = [x < 0] * [x > 0]
//    for(size_t i = 0; i < N; ++i) {
//      float dydx = (x[i] > 0 && x[i] < 1) ? 1 : 0;
//      dLdx[i] = dLdy[i] * dydx;
//    }
//    return dLdx;
//  }

//  void apply_gradient(float step_size) override {
//    ;
//  }
};


struct Model {
  static constexpr size_t HALFKP_SIZE = 41024;
  static constexpr size_t FEATURE_TRANSFORMER_SIZE = 256;
  static constexpr size_t HIDDEN_DENSE_SIZE = 32;
  static constexpr size_t OUTPUT_SIZE = 1;

  template <size_t N, size_t M>
  struct FeatureTransformer {
    const Board &board;
    Affine<bool, N, int32_t, M> affine;

    std::array<int8_t, M*2> y;

    explicit FeatureTransformer(const Board &board):
      board(board),
      affine()
    {}

    const vec_t<int8_t, M*2> &forward(const std::array<std::bitset<HALFKP_SIZE>, NO_COLORS> &inputs) {
      // concatenation
      const COLOR c = board.activePlayer();
      size_t y_start = 0;
      for(COLOR c : {c, enemy_of(c)}) {
        // each half is affine transform (+ clipped relu: later)
        for(size_t i = 0; i < M; ++i) {
          int32_t dotprod = affine.b[i];
          // very inefficient (sparsity)
          for(size_t j = 0; j < N; ++j) {
            if(inputs[c][j]) {
              dotprod += affine.A[i * N + j];
            }
          }
          y[y_start + i] = ClippedReLU<int32_t, M*2, int8_t>::activate(dotprod);
        }
        y_start += M;
      }
      return y;
    }
  };

  template <size_t N, size_t M>
  struct AffineNNUEReLU {
    Affine<int8_t, N, int32_t, M> affine;
    NNUEReLU<int32_t, M, int8_t> nnue_relu;

    AffineNNUEReLU():
      affine(), nnue_relu()
    {}

    const vec_t<int8_t, M> &forward(const vec_t<int8_t, N> &x) {
      return nnue_relu.forward(affine.forward(x));
    }
  };

  FeatureTransformer<HALFKP_SIZE, FEATURE_TRANSFORMER_SIZE> feature_transformer;
  AffineNNUEReLU<FEATURE_TRANSFORMER_SIZE*2, HIDDEN_DENSE_SIZE> hidden1;
  AffineNNUEReLU<HIDDEN_DENSE_SIZE, HIDDEN_DENSE_SIZE> hidden2;
  Affine<int8_t, HIDDEN_DENSE_SIZE, int32_t, 1> output;

  explicit Model(const Board &board):
    feature_transformer(board), hidden1(), hidden2(), output()
  {}

  int32_t forward(const std::array<std::bitset<HALFKP_SIZE>, NO_COLORS> &inputs) {
    return output.forward(hidden2.forward(hidden1.forward(feature_transformer.forward(inputs)))).front();
  }

  template <typename AffineT>
  void read_affine(FILE **fp, AffineT &layer) {
    using wtType = typename decltype(layer.A)::value_type;
    using outType = typename decltype(layer.b)::value_type;
    fread(layer.b.data(), sizeof(outType), layer.b.size(), *fp);
    fread(layer.A.data(), sizeof(wtType), layer.A.size(), *fp);
  }

  void load(const std::string &filename) {
    FILE *fp = fopen(filename.c_str(), "rb");
    if(fp == nullptr) {
      str::perror("error: no such file <", filename, ">");
      abort();
    }
    uint32_t version; fread(&version, sizeof(uint32_t), 1, fp);
    uint32_t nethash; fread(&nethash, sizeof(uint32_t), 1, fp);
    uint32_t netsize; fread(&netsize, sizeof(uint32_t), 1, fp);
    str::print("version", version);
    str::print("hash", nethash);
    str::print("size", netsize);
    char netarch[netsize + 1];
    fread(netarch, sizeof(char), netsize, fp);
    netarch[netsize] = '\0';
    str::print("netarch", (const char *)netarch);
    uint32_t header; fread(&header, sizeof(uint32_t), 1, fp);
    str::print("header", header);
    read_affine(&fp, feature_transformer.affine);
    uint32_t header2; fread(&header2, sizeof(uint32_t), 1, fp);
    str::print("header", header2);
    read_affine(&fp, hidden1.affine);
    read_affine(&fp, hidden2.affine);
    read_affine(&fp, output);
    fclose(fp);
  }
};


struct halfkp {
  Model model;
  std::array<std::bitset<Model::HALFKP_SIZE>, NO_COLORS> inputs;

  //std::stack<std::array<float, SIZE_H1>> prev_acc[NO_COLORS];

  explicit halfkp(const Board &board):
    model(board)
  {
    init_halfkp_features(board);
  }

  static size_t piece_index(PIECE p, COLOR c) {
    if(p == EMPTY)return 0;
    pos_t pieceval = 0;
    switch(p) {
      case PAWN:  pieceval=1; break;
      case KNIGHT:pieceval=2; break;
      case BISHOP:pieceval=3; break;
      case ROOK:  pieceval=4; break;
      case QUEEN: pieceval=5; break;
      case EMPTY: case KING:
        abort();
      break;
    }
    return size_t(pieceval * 2 + c) * board::SIZE + 1;
  }

  static INLINE pos_t orient(bool iswhitepov, pos_t pos) {
    return (board::SIZE - 1) * (iswhitepov ? 1 : 0) ^ pos;
  }

  static INLINE size_t make_halfkp_index(bool iswhitepov, pos_t kingpos, pos_t pos, PIECE p) {
    return orient(iswhitepov, pos) + halfkp::piece_index(p, iswhitepov?WHITE:BLACK) + 10 * kingpos;
  }

  void init_halfkp_features(const Board &board) {
    const COLOR _c = board.activePlayer();
    for(COLOR c : {_c, enemy_of(_c)}) {
      const pos_t kingpos = board.pos_king[c];
      for(PIECE p: {PAWN, KNIGHT, BISHOP, ROOK, QUEEN}) {
        bitmask::foreach(board.get_mask(Piece(p, c)), [&](pos_t pos) mutable noexcept -> void {
          inputs[c].set(make_halfkp_index(c == WHITE, orient(c == WHITE, kingpos), pos, p));
        });
      }
    }
  }

  int32_t init_forward_pass() {
    return model.forward(inputs);
  }

//  void update_feat_persp(
//      const std::vector<size_t> &removed_features,
//      const std::vector<size_t> &added_features,
//      COLOR c
//  )
//  {
//    for(size_t r : removed_features) {
//      for(size_t i = 0; i < model.FEATURE_TRANSFORMER_SIZE; ++i) {
//        y1[c][i] -= A1[c][r * Model::FEATURE_TRANSFORMER_SIZE + i];
//      }
//    }
//    for(size_t a : added_features) {
//      for(size_t i = 0; i < model.FEATURE_TRANSFORMER_SIZE; ++i) {
//        y1[c][i] += A1[c][a * Model::FEATURE_TRANSFORMER_SIZE + i];
//      }
//    }
//  }

  static float value_to_centipawn(int32_t nn_value) {
    return float(((nn_value / 16) * 100) / 208) / 100;
  }
};

} // namespace nn
