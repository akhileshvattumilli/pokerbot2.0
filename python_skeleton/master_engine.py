'''
Elite Numba-optimized Engine for Build4Good 2026 Poker Challenge.
Contains bitwise evaluator and Monte Carlo simulation.
'''
import numpy as np
import numba

@numba.njit(fastmath=True)
def evaluate_5_cards(cards):
    '''Bitwise-optimized 5-card evaluator. Avoids untyped lists.'''
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
    
    c4, c3 = -1, -1
    c2 = np.full(2, -1, dtype=np.int32)
    c1 = np.full(5, -1, dtype=np.int32)
    c2_ptr, c1_ptr = 0, 0
    for r in range(12, -1, -1):
        if r_counts[r] == 4: c4 = r
        elif r_counts[r] == 3: c3 = r
        elif r_counts[r] == 2: c2[c2_ptr] = r; c2_ptr += 1
        elif r_counts[r] == 1: c1[c1_ptr] = r; c1_ptr += 1
        
    if c4 != -1: return (7 << 20) + (c4 << 4) + c1[0]
    if c3 != -1 and c2[0] != -1: return (6 << 20) + (c3 << 4) + c2[0]
    if is_f:
        sc, idx = 5 << 20, 0
        for r in range(12, -1, -1):
            if (r_mask >> r) & 1: sc += (r << (4 * (4 - idx))); idx += 1
        return sc
    if is_s: return (4 << 20) + s_h
    if c3 != -1: return (3 << 20) + (c3 << 8) + (c1[0] << 4) + (c1[1] if c1[1] != -1 else 0)
    if c2[1] != -1: return (2 << 20) + (c2[0] << 8) + (c2[1] << 4) + (c1[0] if c1[0] != -1 else 0)
    if c2[0] != -1: return (1 << 20) + (c2[0] << 12) + (c1[0] << 8) + (c1[1] << 4) + (c1[2] if c1[2] != -1 else 0)
    sc, idx = 0, 0
    for r in range(12, -1, -1):
        if (r_mask >> r) & 1: sc += (r << (4 * (4 - idx))); idx += 1
    return sc

@numba.njit(fastmath=True)
def evaluate_7_cards(cards):
    max_s = 0
    for i in range(7):
        for j in range(i + 1, 7):
            f_c = np.empty(5, dtype=np.int32)
            ptr = 0
            for k in range(7):
                if k != i and k != j: f_c[ptr] = cards[k]; ptr += 1
            s = evaluate_5_cards(f_c)
            if s > max_s: max_s = s
    return max_s

@numba.njit(fastmath=True)
def monte_carlo_redraw_aware(my_h, board, redraws_u, iterations):
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
        c_p = 2
        sim_b = np.empty(5, dtype=np.int32)
        for i in range(n_b): sim_b[i] = board[i]
        for i in range(5 - n_b): sim_b[n_b + i] = deck[indices[c_p]]; c_p += 1
        m_sim_h, o_sim_h = my_h.copy(), opp_h.copy()
        if not redraws_u[0] and n_b < 5:
            m_sim_h[0 if (m_sim_h[0]//4 < m_sim_h[1]//4) else 1] = deck[indices[c_p]]; c_p += 1
        if not redraws_u[1] and n_b < 5:
            o_sim_h[0 if (o_sim_h[0]//4 < o_sim_h[1]//4) else 1] = deck[indices[c_p]]; c_p += 1
        mh, oh = np.empty(7, dtype=np.int32), np.empty(7, dtype=np.int32)
        mh[0], mh[1], oh[0], oh[1] = m_sim_h[0], m_sim_h[1], o_sim_h[0], o_sim_h[1]
        for i in range(5): mh[2+i], oh[2+i] = sim_b[i], sim_b[i]
        ms, os = evaluate_7_cards(mh), evaluate_7_cards(oh)
        if ms > os: wins += 1.0
        elif ms == os: wins += 0.5
    return wins / iterations
