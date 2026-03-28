'''
This file contains the base class and implementation for your pokerbot.
Optimized for Build4Good 2026 Poker Challenge.
'''
import numpy as np
import numba
from .actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from .states import STARTING_STACK

# Card Constants for Numba
# Ranks: 2=0, 3=1, ..., T=8, J=9, Q=10, K=11, A=12
# Suits: s=0, h=1, d=2, c=3
RANK_MAP = {r: i for i, r in enumerate('23456789TJQKA')}
SUIT_MAP = {s: i for i, s in enumerate('shdc')}

@numba.njit
def evaluate_5_cards(cards):
    '''
    Evaluates 5 cards and returns a numerical score.
    Score structure: [Category: 4 bits][Tie-breakers: 20 bits]
    Categories: 0: High Card, 1: Pair, 2: Two Pair, 3: Trips, 4: Straight, 5: Flush, 6: Full House, 7: Quads, 8: Straight Flush
    '''
    ranks = np.zeros(13, dtype=np.int32)
    suits = np.zeros(4, dtype=np.int32)
    rank_list = np.zeros(5, dtype=np.int32)
    
    for i in range(5):
        r = cards[i] // 4
        s = cards[i] % 4
        ranks[r] += 1
        suits[s] += 1
        rank_list[i] = r
    
    rank_list = np.sort(rank_list)[::-1]
    
    is_flush = False
    for i in range(4):
        if suits[i] == 5:
            is_flush = True
            break
    
    # Check for straight
    is_straight = False
    distinct_ranks = 0
    min_rank = 13
    max_rank = -1
    for r in range(13):
        if ranks[r] > 0:
            distinct_ranks += 1
            if r < min_rank: min_rank = r
            if r > max_rank: max_rank = r
            
    straight_high = -1
    if distinct_ranks == 5:
        if max_rank - min_rank == 4:
            is_straight = True
            straight_high = max_rank
        elif max_rank == 12 and ranks[0] > 0 and ranks[1] > 0 and ranks[2] > 0 and ranks[3] > 0:
            # Wheel: A, 2, 3, 4, 5
            is_straight = True
            straight_high = 3 # 5-high
    
    if is_flush and is_straight:
        return (8 << 20) + straight_high
        
    # Analyze counts
    cnt4 = -1
    cnt3 = -1
    cnt2 = [] # Numba handles list of ints
    cnt1 = []
    
    for r in range(12, -1, -1):
        if ranks[r] == 4: cnt4 = r
        elif ranks[r] == 3: cnt3 = r
        elif ranks[r] == 2: cnt2.append(r)
        elif ranks[r] == 1: cnt1.append(r)
            
    if cnt4 != -1: 
        return (7 << 20) + (cnt4 << 4) + cnt1[0]
    
    if cnt3 != -1 and len(cnt2) > 0: 
        return (6 << 20) + (cnt3 << 4) + cnt2[0]
    
    if is_flush:
        score = 5 << 20
        for i in range(5): score += rank_list[i] << (4 * (4 - i))
        return score
        
    if is_straight:
        return (4 << 20) + straight_high
        
    if cnt3 != -1: 
        score = (3 << 20) + (cnt3 << 8)
        score += cnt1[0] << 4
        score += cnt1[1]
        return score
        
    if len(cnt2) >= 2: 
        score = (2 << 20) + (cnt2[0] << 8) + (cnt2[1] << 4)
        score += cnt1[0]
        return score
        
    if len(cnt2) == 1: 
        score = (1 << 20) + (cnt2[0] << 12)
        score += cnt1[0] << 8
        score += cnt1[1] << 4
        score += cnt1[2]
        return score
        
    # High Card
    score = 0
    for i in range(5): score += rank_list[i] << (4 * (4 - i))
    return score

@numba.njit
def evaluate_7_cards(cards):
    max_score = 0
    # 21 combinations
    for i in range(7):
        for j in range(i + 1, 7):
            five_cards = np.empty(5, dtype=np.int32)
            idx = 0
            for k in range(7):
                if k != i and k != j:
                    five_cards[idx] = cards[k]
                    idx += 1
            score = evaluate_5_cards(five_cards)
            if score > max_score:
                max_score = score
    return max_score

@numba.njit
def monte_carlo_equity(my_hole, board, iterations):
    n_board = len(board)
    deck = np.empty(52, dtype=np.int32)
    idx = 0
    used = np.zeros(52, dtype=np.bool_)
    for i in range(2): used[my_hole[i]] = True
    for i in range(n_board): used[board[i]] = True
    
    for i in range(52):
        if not used[i]:
            deck[idx] = i
            idx += 1
    
    deck_size = idx
    wins = 0.0
    
    for _ in range(iterations):
        indices = np.random.choice(deck_size, 2 + (5 - n_board), replace=False)
        opp_hole = np.array([deck[indices[0]], deck[indices[1]]])
        
        sim_board = np.empty(5, dtype=np.int32)
        for i in range(n_board): sim_board[i] = board[i]
        for i in range(5 - n_board): sim_board[n_board + i] = deck[indices[2 + i]]
        
        my_hand = np.empty(7, dtype=np.int32)
        my_hand[0], my_hand[1] = my_hole[0], my_hole[1]
        for i in range(5): my_hand[2+i] = sim_board[i]
        
        opp_hand = np.empty(7, dtype=np.int32)
        opp_hand[0], opp_hand[1] = opp_hole[0], opp_hole[1]
        for i in range(5): opp_hand[2+i] = sim_board[i]
        
        m_s = evaluate_7_cards(my_hand)
        o_s = evaluate_7_cards(opp_hand)
        
        if m_s > o_s: wins += 1.0
        elif m_s == o_s: wins += 0.5
            
    return wins / iterations

def parse_card(card_str):
    if not card_str or card_str == '??': return -1
    return RANK_MAP[card_str[:-1]] * 4 + SUIT_MAP[card_str[-1]]

class Bot():
    '''
    Lead AI Poker Bot with Numba-optimized Monte Carlo engine.
    Optimized for Redraw mechanism and aggressive value betting.
    '''

    def __init__(self):
        self.redraw_used = False
        self.opp_deltas = []

    def handle_new_round(self, game_state, round_state, active):
        self.redraw_used = False

    def handle_round_over(self, game_state, terminal_state, active):
        # Tracking opponent performance
        self.opp_deltas.append(terminal_state.deltas[1 - active])

    def get_action(self, game_state, round_state, active):
        legal_actions = round_state.legal_actions()
        my_cards = np.array([parse_card(c) for c in round_state.hands[active]], dtype=np.int32)
        board_cards = np.array([parse_card(c) for c in round_state.board], dtype=np.int32)
        
        # Time Management
        it = 1000 if game_state.game_clock > 40 else 100
        
        # Calculate Base Equity
        base_eq = monte_carlo_equity(my_cards, board_cards, it)
        
        # Redraw Analysis
        if not self.redraw_used and round_state.street < 5:
            # Requirement: if potential Redraw Equity > Base Equity + 0.12, redraw.
            # Simplified: redraw when base equity is weak (< 48%)
            if base_eq < 0.48:
                ranks = [c // 4 for c in my_cards]
                weakest_idx = 0 if ranks[0] < ranks[1] else 1
                
                if RedrawAction in legal_actions:
                    inner = CheckAction() if CheckAction in legal_actions else CallAction()
                    self.redraw_used = True
                    return RedrawAction('hole', weakest_idx, inner)

        # Betting Logic
        my_p = round_state.pips[active]
        opp_p = round_state.pips[1 - active]
        cost = opp_p - my_p
        
        if base_eq > 0.8: # Go All-in
            if RaiseAction in legal_actions:
                _, max_r = round_state.raise_bounds()
                return RaiseAction(max_r)
            elif CallAction in legal_actions: return CallAction()
        
        if base_eq > 0.65: # Value Bet 3/4 Pot
            if RaiseAction in legal_actions:
                min_r, max_r = round_state.raise_bounds()
                pot = (STARTING_STACK - round_state.stacks[0]) + (STARTING_STACK - round_state.stacks[1])
                target = my_p + cost + int(0.75 * pot)
                return RaiseAction(max(min_r, min(max_r, target)))
            elif CallAction in legal_actions: return CallAction()
                
        if base_eq > 0.45:
            if CheckAction in legal_actions: return CheckAction()
            if CallAction in legal_actions and cost < 20: return CallAction()
                
        if CheckAction in legal_actions: return CheckAction()
        if cost <= 2: return CallAction()
        return FoldAction()
