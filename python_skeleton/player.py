'''
Elite Tournament-Ready Poker Bot for Build4Good 2026.
All strategy logic lives here in the Player class.
'''
import numpy as np
from skeleton.actions import FoldAction, CallAction, CheckAction, RaiseAction, RedrawAction
from skeleton.states import STARTING_STACK, BIG_BLIND, RoundState
from skeleton.bot import Bot
from skeleton.runner import parse_args, run_bot

# Import engine from python_skeleton/ (not skeleton/)
from master_engine import monte_carlo_redraw_aware, evaluate_7_cards

# ---------------------------------------------------------------------------
# Safety patch: the provided skeleton's RoundState.proceed() crashes when the
# opponent redraws because their hand list is empty (the bot never sees
# opponent hole cards until showdown).  We fix this without modifying
# states.py by patching the method at import time.
# ---------------------------------------------------------------------------
_original_proceed = RoundState.proceed

def _safe_proceed(self, action):
    if isinstance(action, RedrawAction):
        active = self.button % 2
        target_type = action.target_type
        target_index = action.target_index
        inner_action = action.action

        if self._is_valid_redraw_target(active, target_type, target_index):
            hands = [list(self.hands[0]), list(self.hands[1])]
            board = list(self.board)
            redraws_used = list(self.redraws_used)
            if target_type == 'hole' and len(hands[active]) > target_index:
                hands[active][target_index] = '??'
            elif target_type == 'board' and len(board) > target_index:
                board[target_index] = '??'
            redraws_used[active] = True
            state_after = RoundState(
                self.button, self.street, list(self.pips), list(self.stacks),
                hands, board, redraws_used, self,
            )
            return state_after._proceed_betting_action(inner_action)
        return self._proceed_betting_action(inner_action)
    return self._proceed_betting_action(action)

RoundState.proceed = _safe_proceed
# ---------------------------------------------------------------------------

# Card Constants
RANK_MAP = {r: i for i, r in enumerate('23456789TJQKA')}
SUIT_MAP = {s: i for i, s in enumerate('shdc')}


def parse_card(card_str):
    if not card_str or card_str == '??':
        return -1
    return RANK_MAP[card_str[:-1]] * 4 + SUIT_MAP[card_str[-1]]


class Player(Bot):
    '''
    Elite poker bot with Numba-optimised Monte Carlo simulation,
    opponent profiling, and adaptive strategy stances.
    '''

    def __init__(self):
        self.redraw_used = False
        self.stats = np.zeros(8, dtype=np.float64)
        # Warm up JIT compiler to avoid first-round timeout
        self._warmup()

    def _warmup(self):
        h = np.array([0, 1], dtype=np.int32)
        b = np.array([2, 3, 4], dtype=np.int32)
        monte_carlo_redraw_aware(h, b, np.array([False, False]), 1)

    def handle_new_round(self, game_state, round_state, active):
        self.redraw_used = False

    def handle_round_over(self, game_state, terminal_state, active):
        self.stats[0] += 1
        prev = terminal_state.previous_state
        opp_idx = 1 - active

        # Statistical profiling from showdown data
        if (prev and
            len(prev.hands[opp_idx]) > 0 and
            prev.hands[opp_idx][0] != '??'):
            opp_cards = [parse_card(c) for c in prev.hands[opp_idx]]
            board_cards = [parse_card(c) for c in prev.board]
            if -1 not in opp_cards and -1 not in board_cards and len(opp_cards + board_cards) == 7:
                s = evaluate_7_cards(np.array(opp_cards + board_cards, dtype=np.int32))
                if prev.pips[opp_idx] > (20 * BIG_BLIND) and s < (2 << 20):
                    self.stats[6] += 1  # bluffer flag

        # Traverse action history for opponent profiling
        curr = prev
        while curr:
            old = curr.previous_state
            if old:
                actor = old.button % 2
                if actor == opp_idx:
                    d = curr.pips[opp_idx] - old.pips[opp_idx]
                    if d > 0:
                        if old.street == 0:
                            self.stats[1] += 1  # VPIP
                            if d > BIG_BLIND:
                                self.stats[2] += 1  # PFR
                        if d > BIG_BLIND:
                            self.stats[4] += 1  # Aggressive
                        else:
                            self.stats[3] += 1  # Passive
            curr = old

    def get_action(self, game_state, round_state, active):
        legal = round_state.legal_actions()
        my_cards = np.array(
            [parse_card(c) for c in round_state.hands[active]], dtype=np.int32
        )
        board = np.array(
            [parse_card(c) for c in round_state.board], dtype=np.int32
        )

        # Adaptive Monte Carlo scaling based on remaining clock
        clock = game_state.game_clock
        it = 5000 if clock > 120 else (1000 if clock > 40 else 200)

        redraws = np.array(
            [self.redraw_used, round_state.redraws_used[1 - active]]
        )
        p_win = monte_carlo_redraw_aware(my_cards, board, redraws, it)

        # Opponent profile: Maniac vs Rock
        p_agg = self.stats[4] / max(1.0, self.stats[3] + self.stats[4])
        omega = 0.8 if p_agg > 0.6 else (1.2 if p_agg < 0.3 else 1.0)

        my_p = round_state.pips[active]
        opp_p = round_state.pips[1 - active]
        cost = opp_p - my_p
        pot = (STARTING_STACK - round_state.stacks[0]) + \
              (STARTING_STACK - round_state.stacks[1])

        ev_adapted = (p_win * (pot + cost)) - (cost * omega)

        # --- Strategic Redraw: The Saboteur ---
        if (not self.redraw_used
                and round_state.street >= 3
                and RedrawAction in legal):
            board_ranks = [parse_card(c) // 4
                           for c in round_state.board if c != '??']
            if len(board_ranks) >= 3:
                target = -1
                for idx, r in enumerate(board_ranks):
                    if board_ranks.count(r) >= 2:
                        target = idx
                        break
                if target != -1 and target < len(round_state.board):
                    self.redraw_used = True
                    if RaiseAction in legal:
                        act = RaiseAction(round_state.raise_bounds()[0])
                    elif CheckAction in legal:
                        act = CheckAction()
                    else:
                        act = CallAction()
                    return RedrawAction('board', target, act)

        # --- Tactical Decisioning ---
        if ev_adapted > (STARTING_STACK * 0.1) or p_win > 0.85:
            if RaiseAction in legal:
                return RaiseAction(round_state.raise_bounds()[1])

        if ev_adapted > 0:
            if RaiseAction in legal and p_win > 0.7:
                v_bet = my_p + cost + int(0.5 * pot)
                lo, hi = round_state.raise_bounds()
                return RaiseAction(max(lo, min(hi, v_bet)))
            if CallAction in legal:
                return CallAction()

        if CheckAction in legal:
            return CheckAction()
        return FoldAction()


if __name__ == '__main__':
    run_bot(Player(), parse_args())
