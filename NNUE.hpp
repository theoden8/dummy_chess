#pragma once


#include <cctype>
#include <array>
#include <vector>
#include <stack>
#include <bitset>

#include <Piece.hpp>
#include <Board.hpp>


#define ENABLE_TRAINING 0


namespace nn {

template <typename T, size_t N> using vec_t = std::array<T, N>;
template <typename T, size_t N, size_t M> using mat_t = std::array<T, N*M>;

template <typename IN_T, size_t IN, typename OUT_T, size_t OUT>
struct Layer {
  static size_t consteval input_size() {
    return IN;
  }
  static size_t consteval output_size() {
    return OUT;
  }
  virtual const vec_t<OUT_T, OUT> &forward(const vec_t<IN_T, IN> &x) = 0;
//  virtual const vec_t<IN_T, IN> &backward(const vec_t<OUT_T, OUT> &dLdy) = 0;
//  virtual void apply_gradient(float step_size) = 0;
};


template <typename IN_T, size_t IN, typename WT_T, typename OUT_T, size_t OUT> struct Affine : public nn::Layer<IN_T, IN, OUT_T, OUT> {
  // forward
  mat_t<WT_T, IN, OUT> A;
  vec_t<OUT_T, OUT> b;
  vec_t<OUT_T, OUT> y;
  // backprop
//  mat_t<IN_T, IN, OUT> dLdA;
//  vec_t<OUT_T, OUT> dLdx;

  Affine():
    nn::Layer<IN_T, IN, OUT_T, OUT>()
  {}

  const vec_t<OUT_T, OUT> &forward(const vec_t<IN_T, IN> &x) override {
//    x = _x;
    for(size_t i = 0; i < IN; ++i) {
      for(size_t j = 0; j < OUT; ++j) {
        y[j] += OUT_T(A[i * OUT + j]) * OUT_T(x[j]);
      }
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

template <typename IN_T, size_t N, typename OUT_T> struct ClippedReLU : public nn::Layer<IN_T, N, OUT_T, N> {
  // forward
//  vec_t<IN_T, N> x;
  vec_t<OUT_T, N> y;
  // backprop
//  vec_t<N> dLdx;

  ClippedReLU():
    nn::Layer<IN_T, N, OUT_T, N>()
  {}

  INLINE static OUT_T activate(IN_T val) {
    return std::min<IN_T>(std::max<IN_T>(val, 0), 127);
  }

  const vec_t<OUT_T,N> &forward(const vec_t<IN_T,N> &x) override {
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

template <typename IN_T, size_t N, typename OUT_T> struct NNUEReLU : public nn::Layer<IN_T, N, OUT_T, N> {
  // forward
//  vec_t<IN_T, N> x;
  vec_t<OUT_T, N> y;
  // backprop
//  vec_t<N> dLdx;

  NNUEReLU():
    nn::Layer<IN_T, N, OUT_T, N>()
  {}

  INLINE static OUT_T activate(IN_T val) {
    return std::max<OUT_T>(std::min<OUT_T>(val / 64, 127), 0);
  }

  const vec_t<OUT_T, N> &forward(const vec_t<IN_T, N> &x) override {
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


enum class NETARCH : int8_t {
  HALFKP,
  UNKNOWN,
};

struct NNUEFileInfo {
  const std::string filename;
  uint32_t version=0, nethash=0, netsize=0;

  NETARCH netarch = NETARCH::UNKNOWN;
  std::string netarch_info;

  explicit NNUEFileInfo(const std::string &filename):
    filename(filename)
  {}

  void load_info() {
    if(version != 0)return;
    {
      FILE *fp = fopen(filename.c_str(), "rb");
      if(fp == nullptr) {
        str::perror("error: no such file <", filename, ">");
        abort();
      }
      fread(&version, sizeof(uint32_t), 1, fp);
      fread(&nethash, sizeof(uint32_t), 1, fp);
      fread(&netsize, sizeof(uint32_t), 1, fp);
      char *netarch_s = new char[netsize + 1];
      fread(netarch_s, sizeof(char), netsize, fp);
      netarch_s[netsize] = '\0';
      netarch_info = netarch_s;
      delete [] netarch_s;
      fclose(fp);
    }
    const std::string fkey = "Features="s;
    const size_t fval_start = netarch_info.find(fkey) + fkey.length();
    size_t fval_end = fval_start;
    for(; fval_end < netarch_info.length(); ++fval_end) {
      if(!isalnum(netarch_info[fval_end]))break;
    }
    const std::string fval = netarch_info.substr(fval_start, fval_end - fval_start);
    if(fval == "HalfKP"s) {
      netarch = NETARCH::HALFKP;
    } else {
      str::perror("unknown feature type"s, fval);
      netarch = NETARCH::UNKNOWN;
    }
  }

  void show() const {
    str::print("version", version);
    str::print("hash", nethash);
    str::print("size", netsize);
  }

  operator NETARCH() {
    load_info();
    return netarch;
  }

  operator std::string() {
    switch(NETARCH(*this)) {
      case NETARCH::HALFKP:return "halfkp"s;
      case NETARCH::UNKNOWN:return "unknown"s;
    }
    abort();
  }
};


template <NETARCH> struct Model;

template <>
struct Model<NETARCH::UNKNOWN> {
//  void update_unset_position(const Board &board, pos_t pos) = 0;
//  void update_set_position(const Board &board, pos_t pos) = 0;
};


template <>
struct Model<NETARCH::HALFKP> : public Model<NETARCH::UNKNOWN> {
  static constexpr size_t HALFKP_SIZE = 41024;
  static constexpr size_t FEATURE_TRANSFORMER_SIZE = 256;
  static constexpr size_t HIDDEN_DENSE_SIZE = 32;
  static constexpr size_t OUTPUT_SIZE = 1;

  template <size_t N, size_t M>
  struct FeatureTransformer {
    Affine<bool, N, int16_t, int16_t, M> affine;

    std::array<int8_t, M*NO_COLORS> y;

    FeatureTransformer():
      affine()
    {}

    const vec_t<int8_t, M*NO_COLORS> &forward(const std::array<std::bitset<HALFKP_SIZE>, NO_COLORS> &inputs) {
      // concatenation
      std::array<int16_t, M> _y;
      for(pos_t c = 0; c < NO_COLORS; ++c) {
        _y.fill(0);
        const auto &sparse_features = inputs[c];
        // affine transform
        for(size_t i = 0; i < N; ++i) {
          // very inefficient
          if(!sparse_features[i])continue;
          for(size_t j = 0; j < M; ++j) {
            _y[j] += int16_t(affine.A[i * M + j]);
          }
        }
        // clipped relu
        for(size_t j = 0; j < M; ++j) {
          y[M * c + j] = ClippedReLU<int16_t, M*NO_COLORS, int8_t>::activate(_y[j]);
        }
      }
      return y;
    }
  };

  template <size_t N, size_t M>
  struct AffineNNUEReLU : public nn::Layer<int8_t, N, int8_t, M> {
    Affine<int8_t, N, int8_t, int32_t, M> affine;
    NNUEReLU<int32_t, M, int8_t> nnue_relu;

    AffineNNUEReLU():
      affine(), nnue_relu()
    {}

    const vec_t<int8_t, M> &forward(const vec_t<int8_t, N> &x) override {
      return nnue_relu.forward(affine.forward(x));
    }
  };

  FeatureTransformer<HALFKP_SIZE, FEATURE_TRANSFORMER_SIZE> feature_transformer;
  AffineNNUEReLU<FEATURE_TRANSFORMER_SIZE*2, HIDDEN_DENSE_SIZE> hidden1;
  AffineNNUEReLU<HIDDEN_DENSE_SIZE, HIDDEN_DENSE_SIZE> hidden2;
  Affine<int8_t, HIDDEN_DENSE_SIZE, int8_t, int32_t, 1> output;

  Model():
    feature_transformer(), hidden1(), hidden2(), output()
  {}

  int32_t forward_transformed(const vec_t<int8_t, FEATURE_TRANSFORMER_SIZE*2> &transformed) {
    return output.forward(hidden2.forward(hidden1.forward(transformed))).front();
  }

  int32_t forward(const std::array<std::bitset<HALFKP_SIZE>, NO_COLORS> &inputs) {
    return forward_transformed(feature_transformer.forward(inputs));
  }

  template <typename AffineT>
  void read_affine(FILE **fp, AffineT &layer) {
    using wtType = typename decltype(layer.A)::value_type;
    using outType = typename decltype(layer.b)::value_type;
    str::pdebug("read affine", sizeof(outType), sizeof(wtType));
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
    str::pdebug("version", version);
    str::pdebug("hash", nethash);
    str::pdebug("size", netsize);
    char *netarch = new char[netsize + 1];
    fread(netarch, sizeof(char), netsize, fp);
    netarch[netsize] = '\0';
    str::pdebug("netarch", (const char *)netarch);
    delete [] netarch;
    uint32_t header; fread(&header, sizeof(uint32_t), 1, fp);
    str::pdebug("header", header);
    read_affine(&fp, feature_transformer.affine);
    uint32_t header2; fread(&header2, sizeof(uint32_t), 1, fp);
    str::pdebug("header", header2);
    read_affine(&fp, hidden1.affine);
    read_affine(&fp, hidden2.affine);
    read_affine(&fp, output);
    size_t current_pos = ftell(fp);
    str::pdebug("current pos", current_pos);
    fseek(fp, 0, 2);
    assert(ftell(fp) - current_pos == 0);
    fclose(fp);
  }
};


struct halfkp {
  Model<NETARCH::HALFKP> model;
  std::array<std::bitset<Model<NETARCH::HALFKP>::HALFKP_SIZE>, NO_COLORS> inputs;

  //std::stack<std::array<float, SIZE_H1>> prev_acc[NO_COLORS];

  explicit halfkp(const Board &board):
    model()
  {
    init_halfkp_features(board);
  }

  static size_t piece_index(PIECE p, bool iswhitepov) {
    if(p == EMPTY)return 0;
    size_t pieceval = 0;
    switch(p) {
      case PAWN:  pieceval=0; break;
      case KNIGHT:pieceval=2; break;
      case BISHOP:pieceval=4; break;
      case ROOK:  pieceval=6; break;
      case QUEEN: pieceval=8; break;
      case EMPTY: case KING: abort(); break;
    }
    if(!iswhitepov)++pieceval;
    return pieceval * board::SIZE + 1;
  }

  static INLINE pos_t orient(bool iswhitepov, pos_t pos) {
    return iswhitepov ? pos : (board::SIZE - pos - 1);
  }

  static INLINE piece_bitboard_t orient_mask(bool iswhitepov, piece_bitboard_t mask) {
    return iswhitepov ? mask : bitmask::reverse_bits(mask);
  }

  static INLINE size_t make_halfkp_index(const Board &board, COLOR c, size_t kingpos, pos_t pos) {
    const bool iswhitepov = (c == WHITE);
    const bool ispiecepov = (c == board[pos].color);
    const PIECE p = board[pos].value;
    str::print("halfkp (", kingpos, pos, halfkp::piece_index(p, ispiecepov), iswhitepov?"True":"False", ") {",
               orient(iswhitepov, pos), halfkp::piece_index(p, ispiecepov), (10u * board::SIZE + 1) * kingpos,
               ispiecepov?"True":"False", "}",
               orient(iswhitepov, pos) + halfkp::piece_index(p, ispiecepov) + (10u * board::SIZE + 1) * kingpos,
               std::string() + Piece(p, ispiecepov?BLACK:WHITE).str(), board::_pos_str(pos));
    return orient(iswhitepov, pos) + halfkp::piece_index(p, ispiecepov) + (10u * board::SIZE + 1) * kingpos;
  }

  void init_halfkp_features(const Board &board) {
    std::cout << "init" << std::endl;
    for(pos_t color_index = 0; color_index < NO_COLORS; ++color_index) {
      const COLOR c = std::array<COLOR,2>{board.activePlayer(), enemy_of(board.activePlayer())}[color_index];
      inputs[color_index].reset();
      const bool iswhitepov = (c == WHITE);
      const size_t orient_kingpos = orient(iswhitepov, board.pos_king[c]);
      const piece_bitboard_t occ = (board.bits[WHITE]|board.bits[BLACK]);
      bitmask::foreach_reversed(occ & ~board.get_king_bits(), [&](pos_t pos) mutable noexcept -> void {
        inputs[color_index].set(make_halfkp_index(board, c, orient_kingpos, pos));
      });
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
