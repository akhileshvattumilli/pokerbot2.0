'''
This file contains the base class and implementation for your pokerbot.
Elite AI Edition: Bitwise-optimized hand evaluation and Redraw Sabotage.
'''
import numpy as np
import numba
from .actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from .states import STARTING_STACK, BIG_BLIND

# Card Constants
RANK_MAP = {r: i for i, r in enumerate('23456789TJQKA')}
SUIT_MAP = {s: i for i, s in enumerate('shdc')}

@numba.njit(fastmath=True)
def evaluate_5_cards(cards):
    '''Bitwise-optimized 5-card evaluator. Avoids untyped lists for Numba.'''
    r_mask = 0
    s_masks = np.zeros(4, dtype=np.uint16)
    r_counts = np.zeros(13, dtype=np.uint8)
    for i in range(5):
        r, s = cards[i] // 4, cards[i] % 4
        r_mask |= (1 << r)
        s_masks[s] |= (1 << r)
        r_counts[r] += 1
    
    # Flush
    is_f = False
    for i in range(4):
        m, cnt = s_masks[i], 0
        while m: m &= m - 1; cnt += 1
        if cnt == 5: is_f = True; break
    # Straight
    is_s, s_h = False, -1
    for r in range(8, -1, -1):
        if (r_mask >> r) & 0x1F == 0x1F:
            is_s, s_h = True, r + 4; break
    if not is_s and r_mask == 0x100F: # A,2,3,4,5
        is_s, s_h = True, 3
    if is_f and is_s: return (8 << 20) + s_h
    
    # Groups: 4-of-a-kind, Full House, 3-of-a-kind, 2-pair, 1-pair, High-card
    # Using arrays instead of lists to avoid Numba typing errors
    c4, c3 = -1, -1
    c2 = np.full(2, -1, dtype=np.int32)
    c1 = np.full(5, -1, dtype=np.int32)
    c2_idx, c1_idx = 0, 0
    for r in range(12, -1, -1):
        if r_counts[r] == 4: c4 = r
        elif r_counts[r] == 3: c3 = r
        elif r_counts[r] == 2: c2[c2_idx] = r; c2_idx += 1
        elif r_counts[r] == 1: c1[c1_idx] = r; c1_idx += 1
        
    if c4 != -1: return (7 << 20) + (c4 << 4) + c1[0]
    if c3 != -1 and c2[0] != -1: return (6 << 20) + (c3 << 4) + c2[0]
    if is_f:
        score, idx = 5 << 20, 0
        for r in range(12, -1, -1):
            if (r_mask >> r) & 1: score += (r << (4 * (4 - idx))); idx += 1
        return score
    if is_s: return (4 << 20) + s_h
    if c3 != -1: return (3 << 20) + (c3 << 8) + (c1[0] << 4) + (c1[1] if c1[1] != -1 else 0)
    if c2[1] != -1: return (2 << 20) + (c2[0] << 8) + (c2[1] << 4) + (c1[0] if c1[0] != -1 else 0)
    if c2[0] != -1: return (1 << 20) + (c2[0] << 12) + (c1[0] << 8) + (c1[1] << 4) + (c1[2] if c1[2] != -1 else 0)
    score, idx = 0, 0
    for r in range(12, -1, -1):
        if (r_mask >> r) & 1: score += (r << (4 * (4 - idx))); idx += 1
    return score

@numba.njit(fastmath=True)
def evaluate_7_cards(cards):
    max_s = 0
    for i in range(7):
        for j in range(i + 1, 7):
            f_c = np.empty(5, dtype=np.int32)
            idx = 0
            for k in range(7):
                if k != i and k != j: f_c[idx] = cards[k]; idx += 1
            s = evaluate_5_cards(f_c)
            if s > max_s: max_s = s
    return max_s

@numba.njit(fastmath=True)
def monte_carlo_refined(my_h, board, redraws_u, iterations):
    n_b = len(board)
    deck, idx, used = np.empty(52, dtype=np.int32), 0, np.zeros(52, dtype=np.bool_)
    for i in range(2): used[my_h[i]] = True
    for i in range(n_b): used[board[i]] = True
    for i in range(52):
        if not used[i]: deck[idx] = i; idx += 1
    d_sz, wins = idx, 0.0
    for _ in range(iterations):
        indices = np.random.choice(d_sz, 2 + (5 - n_b) + 2, replace=False)
        opp_h = np.array([deck[indices[0]], deck[indices[1]]])
        c_idx = 2
        sim_b = np.empty(5, dtype=np.int32)
        for i in range(n_b): sim_b[i] = board[i]
        for i in range(5 - n_b): sim_b[n_b + i] = deck[indices[c_idx]]; c_idx += 1
        m_sim_h, opp_sim_h = my_h.copy(), opp_h.copy()
        if not redraws_u[0] and n_b < 5:
            m_sim_h[0 if (m_sim_h[0]//4 < m_sim_h[1]//4) else 1] = deck[indices[c_idx]]; c_idx += 1
        if not redraws_u[1] and n_b < 5:
            opp_sim_h[0 if (opp_sim_h[0]//4 < opp_sim_h[1]//4) else 1] = deck[indices[c_idx]]; c_idx += 1
        m_hand, o_hand = np.empty(7, dtype=np.int32), np.empty(7, dtype=np.int32)
        m_hand[0], m_hand[1], o_hand[0], o_hand[1] = m_sim_h[0], m_sim_h[1], opp_sim_h[0], opp_sim_h[1]
        for i in range(5): m_hand[2+i], o_hand[2+i] = sim_b[i], sim_b[i]
        m_s, o_s = evaluate_7_cards(m_hand), evaluate_7_cards(o_hand)
        if m_s > o_s: wins += 1.0
        elif m_s == o_s: wins += 0.5
    return wins / iterations

def parse_card(card_str):
    if not card_str or card_str == '??': return -1
    return RANK_MAP[card_str[:-1]] * 4 + SUIT_MAP[card_str[-1]]

class Bot():
    def __init__(self):
        self.redraw_used = False
        self.stats = np.zeros(8, dtype=np.float64)
        # Warmup Numba (Elite strategy)
        self._warmup()

    def _warmup(self):
        h = np.array([0, 1], dtype=np.int32)
        b = np.array([2, 3, 4], dtype=np.int32)
        u = np.array([False, False], dtype=np.bool_)
        monte_carlo_refined(h, b, u, 1)

    def handle_new_round(self, game_state, round_state, active):
        self.redraw_used = False

    def handle_round_over(self, game_state, terminal, active):
        self.stats[0] += 1
        prev, opp_idx = terminal.previous_state, 1 - active
        if prev and len(prev.hands[opp_idx]) > 0 and prev.hands[opp_idx][0] != '??':
            opp_cards = [parse_card(c) for c in prev.hands[opp_idx]]
            board = [parse_card(c) for c in prev.board]
            if -1 not in opp_cards and -1 not in board:
                s = evaluate_7_cards(np.array(opp_cards + board, dtype=np.int32))
                if prev.pips[opp_idx] > 40 and s < (2 << 20): self.stats[6] += 1 
        curr = prev
        while curr:
            old = curr.previous_state
            if old:
                actor = old.button % 2
                if actor == opp_idx:
                    d = curr.pips[opp_idx] - old.pips[opp_idx]
                    if d > 0:
                        if old.street == 0:
                            self.stats[1] += 1 
                            if d > 2: self.stats[2] += 1 
                        if d > 2: self.stats[4] += 1
                        else: self.stats[3] += 1
            curr = old

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        my_cards = np.array([parse_card(c) for c in round_state.hands[active]], dtype=np.int32)
        board = np.array([parse_card(c) for c in round_state.board], dtype=np.int32)
        clock = game_state.game_clock
        it = 5000 if clock > 120 else (1000 if clock > 40 else 200)
        p_win = monte_carlo_refined(my_cards, board, np.array([self.redraw_used, round_state.redraws_used[1-active]]), it)
        p_agg = (self.stats[4]) / max(1.0, self.stats[3] + self.stats[4])
        omega = 0.8 if p_agg > 0.6 else (1.2 if p_agg < 0.3 else 1.0)
        my_p, opp_p = round_state.pips[active], round_state.pips[1-active]
        cost, pot = opp_p - my_p, (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
        ev_adapted = (p_win * (pot + cost)) - (cost * omega)
        
        # Redraw Saboteur
        if not self.redraw_used and round_state.street >= 3 and RedrawAction in legal:
            board_ranks = [c // 4 for c in board]
            if len(board_ranks) >= 3 and any(board_ranks.count(r) >= 2 for r in set(board_ranks)):
                target = -1
                for idx, r in enumerate(board_ranks):
                    if board_ranks.count(r) >= 2: target = idx; break
                if target != -1:
                    self.redraw_used = True
                    return RedrawAction('board', target, RaiseAction(round_state.raise_bounds()[0]) if RaiseAction in legal else CheckAction())

        if ev_adapted > 25 or p_win > 0.85:
            if RaiseAction in legal: return RaiseAction(round_state.raise_bounds()[1])
        if ev_adapted > 0:
            if RaiseAction in legal and p_win > 0.7:
                tar = my_p + cost + int(0.5 * pot) # 50% pot raise
                return RaiseAction(max(round_state.raise_bounds()[0], min(round_state.raise_bounds()[1], tar)))
            if CallAction in legal: return CallAction()
        if CheckAction in legal: return CheckAction()
        return FoldAction()
