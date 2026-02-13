#pragma once

// NNUE-Enabled Engine Extension
// Provides NNUE evaluation as an alternative to traditional heuristics

#include "Engine.hpp"
#include "NNUE.hpp"

#include <memory>
#include <string>

// Compile-time flag to enable/disable NNUE
#ifndef ENABLE_NNUE
#define ENABLE_NNUE 1
#endif

#define self (*this)

class NNUEEngine : public Engine {
public:
    // NNUE components
    std::unique_ptr<nnue::NetworkWeights> nnue_weights_;
    nnue::Evaluator nnue_eval_;
    bool nnue_enabled_ = false;
    
    NNUEEngine() : Engine() {
        #if ENABLE_NNUE
        nnue_weights_ = std::make_unique<nnue::NetworkWeights>();
        #endif
    }
    
    // Load NNUE network from file
    bool load_nnue(const std::string& filename) {
        #if ENABLE_NNUE
        if (!nnue::NetworkLoader::load(filename.c_str(), *nnue_weights_)) {
            str::perror("Failed to load NNUE file:", filename);
            nnue_enabled_ = false;
            return false;
        }
        nnue_eval_.set_weights(nnue_weights_.get());
        nnue_eval_.reset();
        nnue_enabled_ = true;
        str::print("NNUE loaded:", filename);
        return true;
        #else
        (void)filename;
        return false;
        #endif
    }
    
    // Enable/disable NNUE at runtime
    void set_nnue_enabled(bool enabled) {
        nnue_enabled_ = enabled && (nnue_weights_ != nullptr);
    }
    
    bool is_nnue_enabled() const {
        return nnue_enabled_;
    }
    
    // Override make_move to update NNUE accumulators
    INLINE void make_move(pos_t i, pos_t j) {
        #if ENABLE_NNUE
        if (nnue_enabled_) {
            // Compute delta before making the move
            move_t m = bitmask::_pos_pair(i, j);
            nnue::AccumulatorDelta delta_stm, delta_nstm;
            nnue_eval_.compute_delta(as_board(), m, delta_stm, delta_nstm);
            
            // Push accumulator
            nnue_eval_.push_accumulator();
            
            // Check if king moved (requires full refresh)
            const Piece moving = self[i];
            if (moving.value == KING) {
                nnue_eval_.current_accumulator().computed[activePlayer()] = false;
            } else {
                // Apply incremental updates
                nnue_eval_.apply_delta(activePlayer(), delta_stm);
                nnue_eval_.apply_delta(enemy_of(activePlayer()), delta_nstm);
            }
        }
        #endif
        
        Engine::make_move(i, j);
    }
    
    INLINE void make_move(move_t m) {
        return make_move(bitmask::first(m), bitmask::second(m));
    }
    
    // Override retract_move to pop NNUE accumulators
    INLINE void retract_move() {
        Engine::retract_move();
        
        #if ENABLE_NNUE
        if (nnue_enabled_) {
            nnue_eval_.pop_accumulator();
        }
        #endif
    }
    
    // NNUE evaluation (replaces heuristic evaluation)
    INLINE score_t nnue_evaluate() {
        #if ENABLE_NNUE
        if (!nnue_enabled_) {
            return heuristic_of(activePlayer()) - heuristic_of(enemy_of(activePlayer()));
        }
        
        // Ensure accumulators are computed
        auto& acc = nnue_eval_.current_accumulator();
        if (!acc.computed[WHITE]) {
            nnue_eval_.refresh_accumulator(as_board(), WHITE);
        }
        if (!acc.computed[BLACK]) {
            nnue_eval_.refresh_accumulator(as_board(), BLACK);
        }
        
        // Get raw NNUE score and convert to centipawns
        int32_t raw = nnue_eval_.forward(activePlayer());
        score_t cp = nnue::Evaluator::to_centipawns(raw);
        
        // Scale to match our internal score representation
        return cp * CENTIPAWN / 100;
        #else
        return heuristic_of(activePlayer()) - heuristic_of(enemy_of(activePlayer()));
        #endif
    }
    
    // Override evaluate to use NNUE when enabled
    INLINE score_t evaluate(score_t wdl_score=NOSCORE) {
        if(self.is_draw_nogenmoves()) {
            return 0;
        } else if(!self.can_skip_genmoves()) {
            make_move_finalize();
            if(self.is_draw_with_genmoves()){
                return 0;
            } else if(self.is_checkmate()) {
                return -MATERIAL_KING;
            }
        }
        
        // Handle tablebase
        if(self.tb_can_probe()) {
            if(wdl_score == NOSCORE) {
                wdl_score = self.tb_probe_wdl();
            }
            if(wdl_score == NOSCORE) {
                wdl_score = 0;
            } else if(wdl_score == 0) {
                return 0;
            }
        } else {
            wdl_score = 0;
        }
        
        // Use NNUE or traditional evaluation
        #if ENABLE_NNUE
        if (nnue_enabled_) {
            return wdl_score + nnue_evaluate();
        }
        #endif
        
        const COLOR c = activePlayer();
        return wdl_score + heuristic_of(c) - heuristic_of(enemy_of(c));
    }
    
    // Reset NNUE state (call when setting up a new position)
    void reset_nnue() {
        #if ENABLE_NNUE
        if (nnue_enabled_) {
            nnue_eval_.reset();
        }
        #endif
    }
    
    // Refresh NNUE accumulators from current position
    void refresh_nnue() {
        #if ENABLE_NNUE
        if (nnue_enabled_) {
            nnue_eval_.refresh_accumulator(as_board(), WHITE);
            nnue_eval_.refresh_accumulator(as_board(), BLACK);
        }
        #endif
    }
};

#undef self
