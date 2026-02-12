#pragma once
// HalfKP feature extraction for training
// Uses real FEN.hpp header for decompression

#include <cstdint>
#include <cstddef>
#include <vector>

#include "FEN.hpp"

namespace preprocess {

// Extract HalfKP features from compressed FEN bytes
// 
// Parameters:
//   data: pointer to compressed FEN bytes
//   length: length of data
//   flip: if true, swap white/black perspectives
//   white_indices: output array for white features (must have space for 32 elements)
//   black_indices: output array for black features (must have space for 32 elements)
//   n_white: output count of white features
//   n_black: output count of black features  
//   stm: output side to move (0=white, 1=black)
//
inline void extract_halfkp_features(
    const uint8_t* data,
    size_t length,
    bool flip,
    int32_t* white_indices,
    int32_t* black_indices,
    int* n_white,
    int* n_black,
    int64_t* stm
) {
    *n_white = 0;
    *n_black = 0;
    *stm = 0;
    
    if (length == 0) return;
    
    // Use real FEN decompression
    std::vector<uint8_t> v(data, data + length);
    fen::FEN f = fen::compress::decompress_fen(v);
    
    // Set side to move (0 = white, 1 = black), flip if requested
    int64_t stm_val = (f.active_player == WHITE) ? 0 : 1;
    *stm = flip ? (1 - stm_val) : stm_val;
    
    // Parse board string to find pieces and king positions
    int wk = -1, bk = -1;
    int piece_squares[32];
    int piece_types[32];
    bool piece_colors[32];
    int n_pieces = 0;
    
    int sq = 56; // Start at a8 (rank 8, file a)
    for (char c : f.board) {
        if (sq < 0) break;
        
        int piece_type = -1;
        bool is_white = false;
        
        switch (c) {
            case 'P': piece_type = 0; is_white = true; break;
            case 'N': piece_type = 1; is_white = true; break;
            case 'B': piece_type = 2; is_white = true; break;
            case 'R': piece_type = 3; is_white = true; break;
            case 'Q': piece_type = 4; is_white = true; break;
            case 'K': wk = sq; break;
            case 'p': piece_type = 0; is_white = false; break;
            case 'n': piece_type = 1; is_white = false; break;
            case 'b': piece_type = 2; is_white = false; break;
            case 'r': piece_type = 3; is_white = false; break;
            case 'q': piece_type = 4; is_white = false; break;
            case 'k': bk = sq; break;
            case ' ': break; // empty square
            default: break;
        }
        
        if (piece_type >= 0 && n_pieces < 32) {
            piece_squares[n_pieces] = sq;
            piece_types[n_pieces] = piece_type;
            piece_colors[n_pieces] = is_white;
            ++n_pieces;
        }
        
        // Move to next square (row by row, left to right)
        sq++;
        if (sq % 8 == 0) {
            sq -= 16; // Move to start of previous rank
        }
    }
    
    // Need both kings to compute features
    if (wk < 0 || bk < 0) return;
    
    // Compute HalfKP features for each piece
    for (int i = 0; i < n_pieces; ++i) {
        int psq = piece_squares[i];
        int pt = piece_types[i];
        bool is_white = piece_colors[i];
        
        // White perspective: friend=0, enemy=1
        int w_idx = is_white ? 0 : 1;
        // Black perspective: friend=0, enemy=1 (inverted)
        int b_idx = is_white ? 1 : 0;
        
        // Feature index formula: king_sq * 641 + piece_index * 64 + piece_sq + 1
        // piece_index = piece_type * 2 + color_offset
        int32_t white_feat = wk * 641 + (pt * 2 + w_idx) * 64 + psq + 1;
        int32_t black_feat = (63 - bk) * 641 + (pt * 2 + b_idx) * 64 + (63 - psq) + 1;
        
        if (flip) {
            white_indices[i] = black_feat;
            black_indices[i] = white_feat;
        } else {
            white_indices[i] = white_feat;
            black_indices[i] = black_feat;
        }
    }
    
    *n_white = n_pieces;
    *n_black = n_pieces;
}

} // namespace preprocess
