import sys
import os
import numpy as np

# Add skeleton to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'python_skeleton')))

from skeleton.bot import Bot, monte_carlo_equity, parse_card, RANK_MAP, SUIT_MAP

def test_equity():
    print("Testing Monte Carlo Equity...")
    # AA vs unknown
    my_hole = np.array([parse_card('As'), parse_card('Ad')], dtype=np.int32)
    board = np.array([], dtype=np.int32)
    eq = monte_carlo_equity(my_hole, board, 500)
    print(f"AA vs Random (Preflop): {eq:.4f}")
    assert eq > 0.8
    
    # 72o vs unknown
    my_hole = np.array([parse_card('7s'), parse_card('2h')], dtype=np.int32)
    eq = monte_carlo_equity(my_hole, board, 500)
    print(f"72o vs Random (Preflop): {eq:.4f}")
    assert eq < 0.4
    
    # Nut Flush Draw on Flop
    my_hole = np.array([parse_card('As'), parse_card('Ks')], dtype=np.int32)
    board = np.array([parse_card('Qs'), parse_card('Js'), parse_card('2d')], dtype=np.int32)
    eq = monte_carlo_equity(my_hole, board, 500)
    print(f"AKs on QsJs2d (Nut Flush Draw + Gutshot): {eq:.4f}")
    assert eq > 0.5

if __name__ == "__main__":
    try:
        test_equity()
        print("Bot logic test PASSED.")
    except Exception as e:
        print(f"Bot logic test FAILED: {e}")
        import traceback
        traceback.print_exc()
