#pragma once

// dc0 Monte Carlo Tree Search with PUCT.
//
// Usage:
//   1. Create MCTSTree with a root Board position
//   2. Call run_simulations(N, eval_fn) where eval_fn returns (policy, value)
//   3. Call get_policy() to get visit-count distribution
//   4. Call select_move(temperature) to pick a move
//   5. Call advance(move) to reuse the subtree for the next position

#include <cmath>
#include <cstring>
#include <vector>
#include <memory>
#include <optional>
#include <random>
#include <functional>
#include <algorithm>
#include <numeric>
#include <cassert>

#include <Piece.hpp>
#include <Bitmask.hpp>
#include <Board.hpp>

#include <MoveEncoding.hpp>
#include <BoardEncoding.hpp>

namespace dc0 {

// Forward declaration
struct MCTSNode;

// Edge: a move from parent to child
struct MCTSEdge {
    move_t move;                  // engine move_t
    int policy_index;             // index into policy vector (0-4671)
    float prior;                  // P(s, a) from neural network
    uint32_t visit_count = 0;     // N(s, a)
    float value_sum = 0.0f;       // W(s, a) = sum of backed-up values
    int32_t virtual_loss = 0;     // pending virtual losses for batched MCTS
    std::unique_ptr<MCTSNode> child;  // nullptr until expanded

    float q() const {
        uint32_t n = visit_count + virtual_loss;
        if (n == 0) return 0.0f;
        // Virtual losses count as losses (value = -1 each)
        return (value_sum - float(virtual_loss)) / float(n);
    }
};

// Node: a position in the search tree
struct MCTSNode {
    std::vector<MCTSEdge> edges;
    uint32_t visit_count = 0;     // N(s) = sum of child visit counts + 1
    bool is_terminal = false;     // checkmate or draw
    float terminal_value = 0.0f;  // value if terminal (1=win for side that moved here, -1=loss, 0=draw)
    bool is_expanded = false;

    // Expand this node: set edges from policy vector and legal move mask.
    // policy: raw logits or probabilities for legal moves (POLICY_SIZE floats)
    // legal_mask: boolean mask (POLICY_SIZE)
    // The policy is softmaxed over legal moves only.
    void expand(const float* policy, const bool* legal_mask) {
        assert(!is_expanded);
        is_expanded = true;

        // Collect legal moves and their raw policy values
        struct MoveInfo {
            move_t move;
            int policy_index;
            float raw_policy;
        };
        std::vector<MoveInfo> legal_moves;

        for (int idx = 0; idx < POLICY_SIZE; ++idx) {
            if (!legal_mask[idx]) continue;
            if (!is_valid_policy_index(idx)) continue;
            DecodedMove dm = decode_move(idx);
            move_t m = decode_to_move_t(idx);
            legal_moves.push_back({m, idx, policy[idx]});
        }

        if (legal_moves.empty()) {
            // Terminal node (no legal moves) — handled by caller
            return;
        }

        // Softmax over legal moves
        float max_val = -1e30f;
        for (auto& mi : legal_moves) {
            max_val = std::max(max_val, mi.raw_policy);
        }
        float sum_exp = 0.0f;
        for (auto& mi : legal_moves) {
            mi.raw_policy = std::exp(mi.raw_policy - max_val);
            sum_exp += mi.raw_policy;
        }

        // Create edges
        edges.reserve(legal_moves.size());
        for (auto& mi : legal_moves) {
            MCTSEdge edge;
            edge.move = mi.move;
            edge.policy_index = mi.policy_index;
            edge.prior = mi.raw_policy / sum_exp;
            edges.push_back(std::move(edge));
        }
    }

    // Apply Dirichlet noise to root priors for exploration.
    // P'(a) = (1 - epsilon) * P(a) + epsilon * Dir(alpha)
    void apply_dirichlet_noise(float alpha, float epsilon, std::mt19937& rng) {
        if (edges.empty()) return;

        // Sample from Dirichlet by sampling independent Gamma(alpha, 1)
        std::gamma_distribution<float> gamma(alpha, 1.0f);
        std::vector<float> noise(edges.size());
        float noise_sum = 0.0f;
        for (size_t i = 0; i < edges.size(); ++i) {
            noise[i] = gamma(rng);
            noise_sum += noise[i];
        }
        // Normalize
        if (noise_sum > 0.0f) {
            for (size_t i = 0; i < edges.size(); ++i) {
                noise[i] /= noise_sum;
            }
        }
        // Mix with prior
        for (size_t i = 0; i < edges.size(); ++i) {
            edges[i].prior = (1.0f - epsilon) * edges[i].prior + epsilon * noise[i];
        }
    }
};

// Evaluation function signature:
// Given a board position, return (policy_logits[4672], value_for_current_player).
// value is from the perspective of the side to move: +1 = winning, -1 = losing.
using EvalFunction = std::function<std::pair<std::vector<float>, float>(const Board&)>;

// Batched evaluation: encode N boards, return N (policy, value) pairs.
// Input: encoded planes (N * ENCODING_SIZE floats), legal masks (N * POLICY_SIZE bools), count N.
// Output: policies (N * POLICY_SIZE floats as logits), values (N floats).
using BatchedEvalFunction = std::function<void(
    const float* planes_batch,     // N * ENCODING_SIZE
    const bool* masks_batch,       // N * POLICY_SIZE
    int batch_size,
    float* out_policies,           // N * POLICY_SIZE (logits)
    float* out_values              // N
)>;

// MCTS search tree
class MCTSTree {
public:
    // Configuration
    float c_puct = 2.5f;               // exploration constant
    float dirichlet_alpha = 0.3f;       // Dirichlet noise alpha
    float dirichlet_epsilon = 0.25f;    // Dirichlet noise weight
    bool add_noise = false;             // whether to add Dirichlet noise (training)

    MCTSTree() : rng_(std::random_device{}()) {}

    // Initialize the tree with a root position.
    // board is copied internally.
    void set_root(const Board& board) {
        root_ = std::make_unique<MCTSNode>();
        root_board_ = std::make_unique<Board>(board);
    }

    // Run N simulations from the root.
    void run_simulations(int n_simulations, const EvalFunction& eval_fn) {
        if (!root_ || !root_board_) return;

        // Expand root if needed
        if (!root_->is_expanded) {
            float root_value = expand_node(root_.get(), *root_board_, eval_fn);
            root_->visit_count += 1;  // count the root expansion
            (void)root_value;  // root value not backed up further
            if (add_noise) {
                root_->apply_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon, rng_);
            }
        }

        for (int sim = 0; sim < n_simulations; ++sim) {
            // Make a working copy of the board for tree traversal
            Board board(*root_board_);
            run_single_simulation(root_.get(), board, eval_fn);
        }
    }

    // Run N simulations using batched neural network evaluation.
    // Collects up to batch_size leaves per GPU call for much higher throughput.
    void run_simulations_batched(
        int n_simulations,
        const BatchedEvalFunction& batch_eval_fn,
        int batch_size = 64
    ) {
        if (!root_ || !root_board_) return;

        // Expand root if needed (single eval)
        if (!root_->is_expanded) {
            float planes[ENCODING_SIZE];
            encode_board(*root_board_, planes);
            bool legal_mask[POLICY_SIZE];
            encode_legal_moves(*root_board_, legal_mask);

            float policy[POLICY_SIZE];
            float value;
            batch_eval_fn(planes, legal_mask, 1, policy, &value);

            if (root_board_->is_checkmate()) {
                root_->is_terminal = true;
                root_->terminal_value = -1.0f;
                root_->is_expanded = true;
            } else if (root_board_->is_draw()) {
                root_->is_terminal = true;
                root_->terminal_value = 0.0f;
                root_->is_expanded = true;
            } else {
                root_->expand(policy, legal_mask);
            }
            root_->visit_count += 1;
            if (add_noise) {
                root_->apply_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon, rng_);
            }
        }

        if (root_->is_terminal) return;

        // Pre-allocate buffers for batch encoding
        std::vector<float> planes_buf(batch_size * ENCODING_SIZE);
        // Note: std::vector<bool> is bit-packed and has no .data().
        // Use a flat array of bool instead.
        std::unique_ptr<bool[]> masks_buf(new bool[batch_size * POLICY_SIZE]);
        std::vector<float> policies_buf(batch_size * POLICY_SIZE);
        std::vector<float> values_buf(batch_size);

        int sims_done = 0;
        while (sims_done < n_simulations) {
            int batch_target = std::min(batch_size, n_simulations - sims_done);

            // Phase 1: Select leaves with virtual loss
            std::vector<PendingLeaf> pending;
            pending.reserve(batch_target);
            int nn_count = 0;  // leaves needing NN eval

            for (int b = 0; b < batch_target; ++b) {
                PendingLeaf leaf = select_leaf_for_batch();
                if (leaf.needs_nn_eval) {
                    // Encode board into batch buffer at position nn_count
                    std::memcpy(&planes_buf[nn_count * ENCODING_SIZE],
                                leaf.planes, ENCODING_SIZE * sizeof(float));
                    std::memcpy(&masks_buf.get()[nn_count * POLICY_SIZE],
                                leaf.legal_mask, POLICY_SIZE * sizeof(bool));
                    leaf.batch_index = nn_count;
                    nn_count++;
                }
                pending.push_back(std::move(leaf));
            }

            // Phase 2: Batch NN evaluation (single GPU call)
            if (nn_count > 0) {
                batch_eval_fn(planes_buf.data(), masks_buf.get(), nn_count,
                              policies_buf.data(), values_buf.data());
            }

            // Phase 3: Expand leaves and backpropagate
            for (auto& leaf : pending) {
                float value;
                if (leaf.needs_nn_eval) {
                    float* policy = &policies_buf[leaf.batch_index * POLICY_SIZE];
                    value = values_buf[leaf.batch_index];
                    leaf.node->expand(policy, leaf.legal_mask);
                } else {
                    value = leaf.terminal_value;
                }

                // Remove virtual loss and do real backprop
                remove_virtual_loss(leaf.path);
                backpropagate(leaf.path, value);
            }

            sims_done += batch_target;
        }
    }

    // Get the visit-count policy distribution at the root.
    // Returns a vector of (move_t, visit_count, prior, Q-value) for each child.
    struct MoveStats {
        move_t move;
        int policy_index;
        uint32_t visits;
        float prior;
        float q_value;
    };

    std::vector<MoveStats> get_move_stats() const {
        std::vector<MoveStats> stats;
        if (!root_) return stats;
        for (auto& edge : root_->edges) {
            stats.push_back({
                edge.move,
                edge.policy_index,
                edge.visit_count,
                edge.prior,
                edge.q()
            });
        }
        return stats;
    }

    // Get the MCTS policy as visit-count distribution over POLICY_SIZE.
    // Normalized to sum to 1. Uses temperature.
    void get_policy(float* out_policy, float temperature = 1.0f) const {
        std::memset(out_policy, 0, POLICY_SIZE * sizeof(float));
        if (!root_ || root_->edges.empty()) return;

        if (temperature < 1e-6f) {
            // Deterministic: pick the most visited
            uint32_t max_visits = 0;
            int best_idx = -1;
            for (auto& edge : root_->edges) {
                if (edge.visit_count > max_visits) {
                    max_visits = edge.visit_count;
                    best_idx = edge.policy_index;
                }
            }
            if (best_idx >= 0) out_policy[best_idx] = 1.0f;
        } else {
            // Proportional to visit_count^(1/temperature)
            float inv_temp = 1.0f / temperature;
            float sum = 0.0f;
            for (auto& edge : root_->edges) {
                float val = std::pow(float(edge.visit_count), inv_temp);
                out_policy[edge.policy_index] = val;
                sum += val;
            }
            if (sum > 0.0f) {
                for (auto& edge : root_->edges) {
                    out_policy[edge.policy_index] /= sum;
                }
            }
        }
    }

    // Select a move based on visit counts and temperature.
    move_t select_move(float temperature = 0.0f) {
        if (!root_ || root_->edges.empty()) return board::nullmove;

        if (temperature < 1e-6f) {
            // Pick most visited
            uint32_t max_visits = 0;
            move_t best = board::nullmove;
            for (auto& edge : root_->edges) {
                if (edge.visit_count > max_visits) {
                    max_visits = edge.visit_count;
                    best = edge.move;
                }
            }
            return best;
        }

        // Sample proportional to visit_count^(1/temperature)
        float inv_temp = 1.0f / temperature;
        std::vector<float> weights(root_->edges.size());
        for (size_t i = 0; i < root_->edges.size(); ++i) {
            weights[i] = std::pow(float(root_->edges[i].visit_count), inv_temp);
        }
        std::discrete_distribution<size_t> dist(weights.begin(), weights.end());
        size_t idx = dist(rng_);
        return root_->edges[idx].move;
    }

    // Advance the tree: reuse the subtree rooted at the child matching `move`.
    // Discards the rest of the tree.
    void advance(move_t move, const Board& new_board) {
        if (!root_) {
            set_root(new_board);
            return;
        }

        for (auto& edge : root_->edges) {
            if (edge.move == move && edge.child) {
                root_ = std::move(edge.child);
                root_board_ = std::make_unique<Board>(new_board);
                return;
            }
        }

        // Move not found in tree — start fresh
        set_root(new_board);
    }

    // Get total visit count at root
    uint32_t root_visits() const {
        return root_ ? root_->visit_count : 0;
    }

    // Check if root is terminal
    bool is_terminal() const {
        return root_ && root_->is_terminal;
    }

    MCTSNode* root() { return root_.get(); }
    const Board& root_board() const { return *root_board_; }

    // --- Public types for cross-game batched MCTS ---

    struct PathEntry {
        MCTSNode* node;
        MCTSEdge* edge;
    };

    // A leaf selected during batch collection, waiting for NN eval.
    struct PendingLeaf {
        std::vector<PathEntry> path;
        MCTSNode* node = nullptr;           // the newly created child node
        float planes[ENCODING_SIZE];        // encoded board (only if needs_nn_eval)
        bool legal_mask[POLICY_SIZE];       // legal moves (only if needs_nn_eval)
        bool needs_nn_eval = false;         // false if terminal
        float terminal_value = 0.0f;        // value if terminal
        int batch_index = -1;               // index into NN batch output
        int game_index = -1;                // which game this leaf belongs to (for multi-game)
    };

    // Ensure root is expanded using the batched eval function.
    // Returns true if root is ready for search (expanded or terminal).
    bool ensure_root_expanded(const BatchedEvalFunction& batch_eval_fn) {
        if (!root_ || !root_board_) return false;
        if (root_->is_expanded) return true;

        float planes[ENCODING_SIZE];
        encode_board(*root_board_, planes);
        bool legal_mask[POLICY_SIZE];
        encode_legal_moves(*root_board_, legal_mask);

        float policy[POLICY_SIZE];
        float value;
        batch_eval_fn(planes, legal_mask, 1, policy, &value);

        if (root_board_->is_checkmate()) {
            root_->is_terminal = true;
            root_->terminal_value = -1.0f;
            root_->is_expanded = true;
        } else if (root_board_->is_draw()) {
            root_->is_terminal = true;
            root_->terminal_value = 0.0f;
            root_->is_expanded = true;
        } else {
            root_->expand(policy, legal_mask);
        }
        root_->visit_count += 1;
        if (add_noise) {
            root_->apply_dirichlet_noise(dirichlet_alpha, dirichlet_epsilon, rng_);
        }
        return true;
    }

    // Select a single leaf for batched evaluation (public for cross-game batching).
    PendingLeaf select_leaf() {
        return select_leaf_for_batch();
    }

    // Process a leaf after NN evaluation: expand and backpropagate.
    void process_leaf(PendingLeaf& leaf, const float* policy, float value) {
        if (leaf.needs_nn_eval && policy) {
            leaf.node->expand(policy, leaf.legal_mask);
        } else {
            value = leaf.terminal_value;
        }
        remove_virtual_loss(leaf.path);
        backpropagate(leaf.path, value);
    }

private:
    std::unique_ptr<MCTSNode> root_;
    std::unique_ptr<Board> root_board_;
    std::mt19937 rng_;

    // Select a leaf for batched evaluation. Walks the tree using PUCT (with
    // virtual loss applied to previously selected paths). Creates the child
    // node, applies virtual loss along the path, and encodes the board if
    // the leaf needs NN evaluation.
    PendingLeaf select_leaf_for_batch() {
        PendingLeaf leaf;
        Board board(*root_board_);
        MCTSNode* node = root_.get();

        // Selection: walk down picking best edges
        while (node->is_expanded && !node->is_terminal) {
            if (node->edges.empty()) break;

            MCTSEdge* edge = select_edge(node);
            leaf.path.push_back({node, edge});

            // Apply virtual loss to discourage re-selecting this path
            edge->virtual_loss += 1;

            board.make_move(edge->move);

            if (!edge->child) {
                // Leaf found: create child
                edge->child = std::make_unique<MCTSNode>();
                node = edge->child.get();
                leaf.node = node;

                // Check terminal
                if (board.is_checkmate()) {
                    node->is_terminal = true;
                    node->terminal_value = -1.0f;
                    node->is_expanded = true;
                    leaf.needs_nn_eval = false;
                    leaf.terminal_value = -1.0f;
                } else if (board.is_draw()) {
                    node->is_terminal = true;
                    node->terminal_value = 0.0f;
                    node->is_expanded = true;
                    leaf.needs_nn_eval = false;
                    leaf.terminal_value = 0.0f;
                } else {
                    // Needs NN eval — encode the board
                    encode_board(board, leaf.planes);
                    encode_legal_moves(board, leaf.legal_mask);
                    leaf.needs_nn_eval = true;
                }
                return leaf;
            }

            node = edge->child.get();
        }

        // Reached a terminal or already-expanded node with no edges.
        // This happens when all subtrees are terminal or tree is saturated.
        leaf.node = node;
        leaf.needs_nn_eval = false;
        leaf.terminal_value = node->is_terminal ? node->terminal_value : 0.0f;
        return leaf;
    }

    // Remove virtual loss along a path (called before real backprop).
    void remove_virtual_loss(std::vector<PathEntry>& path) {
        for (auto& entry : path) {
            entry.edge->virtual_loss -= 1;
        }
    }

    // Select the best child edge using PUCT formula.
    // UCB(s,a) = Q(s,a) + c_puct * P(s,a) * sqrt(N(s)) / (1 + N(s,a))
    // Virtual losses are included in Q and N for batched MCTS diversification.
    MCTSEdge* select_edge(MCTSNode* node) {
        assert(node->is_expanded && !node->edges.empty());

        // Include virtual losses in parent visit count
        int32_t total_vl = 0;
        for (auto& edge : node->edges) total_vl += edge.virtual_loss;
        float sqrt_parent = std::sqrt(float(node->visit_count + total_vl));

        float best_score = -1e30f;
        MCTSEdge* best = nullptr;

        for (auto& edge : node->edges) {
            float q = edge.q();  // already accounts for virtual loss
            float n_eff = float(edge.visit_count + edge.virtual_loss);
            float u = c_puct * edge.prior * sqrt_parent / (1.0f + n_eff);
            float score = q + u;
            if (score > best_score) {
                best_score = score;
                best = &edge;
            }
        }
        return best;
    }

    // Expand a leaf node: evaluate with the neural network and create edges.
    // Returns the value of this position from the side-to-move's perspective.
    float expand_node(MCTSNode* node, Board& board, const EvalFunction& eval_fn) {
        // Check for terminal position
        if (board.is_checkmate()) {
            node->is_terminal = true;
            // The side to move is checkmated -> loss for current player
            node->terminal_value = -1.0f;
            node->is_expanded = true;
            return -1.0f;
        }
        if (board.is_draw()) {
            node->is_terminal = true;
            node->terminal_value = 0.0f;
            node->is_expanded = true;
            return 0.0f;
        }

        // Get legal move mask
        bool legal_mask[POLICY_SIZE];
        encode_legal_moves(board, legal_mask);

        // Evaluate position with neural network
        auto [policy, value] = eval_fn(board);

        // Expand with policy
        node->expand(policy.data(), legal_mask);

        return value;
    }

    // Run a single MCTS simulation: select -> expand -> backprop.
    // board is modified during traversal and restored after.
    void run_single_simulation(MCTSNode* root, Board& board, const EvalFunction& eval_fn) {
        // Path from root to leaf for backpropagation
        std::vector<PathEntry> path;

        MCTSNode* node = root;

        // Selection: walk down the tree picking best edges
        while (node->is_expanded && !node->is_terminal) {
            if (node->edges.empty()) break;

            MCTSEdge* edge = select_edge(node);
            path.push_back({node, edge});

            // Make the move on the board
            board.make_move(edge->move);

            if (!edge->child) {
                // Leaf: create and expand child, get value in one NN call
                edge->child = std::make_unique<MCTSNode>();
                float leaf_value = expand_node(edge->child.get(), board, eval_fn);

                // Backpropagate (value is from leaf's perspective)
                backpropagate(path, leaf_value);

                // Undo all moves
                for (size_t i = path.size(); i > 0; --i) {
                    board.retract_move();
                }
                return;
            }

            node = edge->child.get();
        }

        // Reached a terminal node or an expanded node with no edges
        float value;
        if (node->is_terminal) {
            value = node->terminal_value;
        } else {
            // Shouldn't happen in normal flow, but handle gracefully
            auto [policy, val] = eval_fn(board);
            value = val;
        }

        backpropagate(path, value);

        // Undo all moves
        for (size_t i = path.size(); i > 0; --i) {
            board.retract_move();
        }
    }

    // Backpropagate value up the path.
    // Value is from the perspective of the player at the leaf (bottom of path).
    // As we go up, we negate at each level (alternating players).
    void backpropagate(std::vector<PathEntry>& path, float leaf_value) {
        float value = leaf_value;
        // Walk from leaf to root
        for (int i = (int)path.size() - 1; i >= 0; --i) {
            auto& entry = path[i];
            // The edge value is from the perspective of the player who made the move
            // (i.e., the parent's player). The leaf value is from the child's perspective.
            // So we negate: parent sees -child_value.
            entry.edge->visit_count += 1;
            entry.edge->value_sum += -value;  // negate for parent's perspective
            entry.node->visit_count += 1;
            value = -value;  // flip for next level up
        }
    }
};

} // namespace dc0
