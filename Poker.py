#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Poker AI Assistant — supports 2-max up to 9-max tables.

Key features
- 7-card hand evaluator (accelerated by eval7 when available; falls back to Numba/Python)
- Range parsing (now supports eval7's extended PokerStove syntax & weights; graceful fallback)
- Equity calculator with exact river/turn/flop rollouts + Monte Carlo otherwise (eval7-powered when available)
- Multiway equity estimation (2–9 active players)
- Heads-up DCFR spot-solver for per-street trees (bet/raise/call/fold)
- Full GTOSolver: multi-street, bucketed, range-vs-range DCFR with chance sampling
- Action abstraction (size sets for bet/raise)
- Opponent modeling (nit/tag/lag/station) feeding solver priors
- Tournament tracker integrating with mitmproxy websocket feed
- Position mapping from 2-max to 9-max (BTN, SB, BB, UTG, UTG1, UTG2, LJ, HJ, CO)
- Preflop RFI priors per position across table sizes

Optimizations:
- Async computation to avoid blocking proxy
- Memory-optimized caching with size limits and TTL
- Reduced cache sizes and sampling counts for better memory usage
"""

import xml.etree.ElementTree as ET
from collections import defaultdict, OrderedDict
from typing import Dict, List, Optional, Set, Any, Tuple
import math
import random
import numpy as np
import time
import copy
import threading
import itertools
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

# ---------------------------------------------------------
# Optional eval7 (fast Cython evaluator & range parser)
# ---------------------------------------------------------
try:
    import eval7 as _eval7  # type: ignore
    _EVAL7_AVAILABLE = True
except Exception:
    _eval7 = None
    _EVAL7_AVAILABLE = False

# Pre-create a cache of eval7.Card objects for zero-alloc hot paths
_EVAL7_CARD_CACHE: Dict[str, Any] = {}
if _EVAL7_AVAILABLE:
    # eval7 suits: s, h, d, c ; ranks: 2..A
    for r in "23456789TJQKA":
        for s in "shdc":
            cs = f"{r}{s}"
            try:
                _EVAL7_CARD_CACHE[cs] = _eval7.Card(cs)
            except Exception:
                # Defensive: if eval7 rejects (shouldn't)
                pass

def _to_eval7_card(card_str: str):
    """Fast conversion from 'As' => eval7.Card using cache."""
    if not _EVAL7_AVAILABLE:
        return None
    return _EVAL7_CARD_CACHE.get(card_str) or _eval7.Card(card_str)

# ---------------------------------------------------------
# Optional numba (graceful fallback)
# ---------------------------------------------------------
try:
    from numba import jit as _numba_jit  # type: ignore

    def _jit(**kwargs):
        return _numba_jit(**kwargs)
except Exception:
    def _jit(**kwargs):
        def deco(fn):
            return fn
        return deco

# ==========================================================
# Memory-optimized LRU Cache with TTL
# ==========================================================
class OptimizedLRUCache:
    def __init__(self, capacity: int = 256, ttl_seconds: float = 300.0):
        self.capacity = capacity
        self.ttl = ttl_seconds
        self._data = OrderedDict()
        self._timestamps = {}

    def get(self, key):
        now = time.monotonic()
        if key in self._data:
            # Check TTL
            if now - self._timestamps.get(key, 0) > self.ttl:
                self._remove_key(key)
                return None
            self._data.move_to_end(key)
            return self._data[key]
        return None

    def set(self, key, value):
        now = time.monotonic()
        self._data[key] = value
        self._timestamps[key] = now
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            old_key, _ = self._data.popitem(last=False)
            self._timestamps.pop(old_key, None)

    def _remove_key(self, key):
        self._data.pop(key, None)
        self._timestamps.pop(key, None)

    def cleanup_expired(self):
        now = time.monotonic()
        expired = [k for k, ts in self._timestamps.items() if now - ts > self.ttl]
        for k in expired:
            self._remove_key(k)

# ==========================================================
# HandEvaluator (7-card value ranking) — eval7-accelerated
# ==========================================================
from typing import List as _List, Tuple as _Tuple
import numpy.typing as npt


class HandEvaluator:
    # Order chosen to make rank indexes monotonic ascending with hand strength computation
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    SUITS = ['h', 'd', 'c', 's']

    def __init__(self):
        self.rank_values = {rank: i for i, rank in enumerate(self.RANKS)}
        self.suit_values = {suit: i for i, suit in enumerate(self.SUITS)}

    @staticmethod
    @_jit(nopython=True)
    def _calculate_hand_value(ranks: npt.NDArray[np.int32], suits: npt.NDArray[np.int32]) -> int:
        """
        Compact 7-card evaluator that returns an integer for total ordering.
        Categories (hi→lo):
        8: Straight Flush, 7: Quads, 6: Full House, 5: Flush, 4: Straight,
        3: Trips, 2: Two Pair, 1: One Pair, 0: High Card.
        Tie breakers packed into lower bits. Higher value => better hand.
        """
        # Count occurrences
        rank_count = np.zeros(13, dtype=np.int32)
        suit_count = np.zeros(4, dtype=np.int32)
        for i in range(len(ranks)):
            rank_count[ranks[i]] += 1
            suit_count[suits[i]] += 1

        # Helper: compute straight top rank index (ace-low handled)
        def straight_top(occ):
            cnt = 0
            best = -1
            for i in range(13):
                if occ[i] > 0:
                    cnt += 1
                    if cnt >= 5:
                        best = i
                else:
                    cnt = 0
            # Ace-low (A2345): ranks 12,0,1,2,3 present => treat as 5-high straight (top rank=3)
            if best < 0:
                if occ[12] > 0 and occ[0] > 0 and occ[1] > 0 and occ[2] > 0 and occ[3] > 0:
                    return 3  # '5' index
            return best

        # Straight Flush?
        flush_suit = -1
        for s in range(4):
            if suit_count[s] >= 5:
                flush_suit = s
                break
        if flush_suit >= 0:
            occ = np.zeros(13, dtype=np.int32)
            for i in range(len(ranks)):
                if suits[i] == flush_suit:
                    occ[ranks[i]] = 1
            st = straight_top(occ)
            if st >= 0:
                return (8 << 24) | (st << 16)

        # Quads
        quad_rank = -1
        for r in range(12, -1, -1):
            if rank_count[r] == 4:
                quad_rank = r
                break
        if quad_rank >= 0:
            kicker = -1
            for r in range(12, -1, -1):
                if r != quad_rank and rank_count[r] > 0:
                    kicker = r
                    break
            return (7 << 24) | (quad_rank << 16) | (kicker << 8)

        # Trips + Pair => Full House (choose best trips, then best pair/trips)
        trip_ranks = []
        pair_ranks = []
        for r in range(12, -1, -1):
            if rank_count[r] >= 3:
                trip_ranks.append(r)
            elif rank_count[r] >= 2:
                pair_ranks.append(r)
        if trip_ranks:
            best_trip = trip_ranks[0]
            # second trip can act as pair
            best_pair = -1
            if len(trip_ranks) > 1:
                best_pair = trip_ranks[1]
            if pair_ranks:
                best_pair = max(best_pair, pair_ranks[0])
            if best_pair >= 0:
                return (6 << 24) | (best_trip << 16) | (best_pair << 8)

        # Straight
        st2 = straight_top(rank_count)
        if st2 >= 0:
            return (4 << 24) | (st2 << 16)

        # Flush (top 5 kickers in suit)
        if flush_suit >= 0:
            suit_ranks = []
            for i in range(len(ranks)):
                if suits[i] == flush_suit:
                    suit_ranks.append(ranks[i])
            suit_ranks.sort(reverse=True)
            suit_ranks = suit_ranks[:5]
            val = (5 << 24)
            shift = 16
            for r in suit_ranks:
                val |= (r << shift)
                shift -= 3
            return val

        # Trips
        if trip_ranks:
            t = trip_ranks[0]
            kickers = []
            for r in range(12, -1, -1):
                if r != t and rank_count[r] > 0:
                    kickers.append(r)
                if len(kickers) == 2:
                    break
            return (3 << 24) | (t << 16) | (kickers[0] << 8) | (kickers[1] if len(kickers) > 1 else 0)

        # Two Pair
        if len(pair_ranks) >= 2:
            hi, lo = pair_ranks[0], pair_ranks[1]
            kicker = -1
            for r in range(12, -1, -1):
                if r != hi and r != lo and rank_count[r] > 0:
                    kicker = r
                    break
            return (2 << 24) | (hi << 16) | (lo << 8) | (kicker if kicker >= 0 else 0)

        # One Pair
        if len(pair_ranks) == 1:
            p = pair_ranks[0]
            kickers = []
            for r in range(12, -1, -1):
                if r != p and rank_count[r] > 0:
                    kickers.append(r)
                if len(kickers) == 3:
                    break
            out = (1 << 24) | (p << 16)
            if kickers:
                out |= (kickers[0] << 12)
            if len(kickers) > 1:
                out |= (kickers[1] << 8)
            if len(kickers) > 2:
                out |= (kickers[2] << 4)
            return out

        # High Card
        kickers = []
        for r in range(12, -1, -1):
            if rank_count[r] > 0:
                kickers.append(r)
            if len(kickers) == 5:
                break
        out = 0
        shift = 20
        for r in kickers:
            out |= (r << shift)
            shift -= 4
        return out

    def evaluate_hand(self, cards: _List[str]) -> _Tuple[int, str]:
        """
        Evaluate a 5- to 7-card poker hand.
        If eval7 is available, use its Cython evaluator; otherwise use the built-in fallback.
        Returns (numeric_value, category_name) where a higher numeric_value is stronger.
        """
        if len(cards) < 5:
            return 0, "High Card"

        if _EVAL7_AVAILABLE:
            # Convert to eval7.Card and evaluate
            try:
                e7_cards = [_to_eval7_card(c) for c in cards]
                score = _eval7.evaluate(e7_cards)
                # eval7.handtype returns human-readable category like 'Pair', 'Straight', etc.
                cat = _eval7.handtype(score) if hasattr(_eval7, "handtype") else "Unknown"
                return int(score), str(cat)
            except Exception:
                # Fall through to Python evaluator on any unexpected error
                pass

        # Fallback evaluator
        ranks = np.array([self.rank_values[c[0]] for c in cards], dtype=np.int32)
        suits = np.array([self.suit_values[c[1]] for c in cards], dtype=np.int32)
        value = self._calculate_hand_value(ranks, suits)
        cat_idx = value >> 24
        categories = {
            8: "Straight Flush",
            7: "Four of a Kind",
            6: "Full House",
            5: "Flush",
            4: "Straight",
            3: "Three of a Kind",
            2: "Two Pair",
            1: "One Pair",
            0: "High Card",
        }
        return value, categories.get(cat_idx, "High Card")


# ==========================================================
# GTO Math Utilities
# ==========================================================
class GTOUtils:
    @staticmethod
    def caller_pot_odds_from_size(bet: float, pot: float) -> float:
        """Caller pot odds = s / (2s + 1), where s = bet/pot."""
        if pot <= 0:
            return 0.0
        s = max(0.0, bet / pot)
        return s / (2.0 * s + 1.0)

    @staticmethod
    def optimal_bluff_ratio(bet: float, pot: float) -> float:
        """Polar mix single decision: optimal bluff:value ratio = s / (1 + s)."""
        if pot <= 0:
            return 0.0
        s = max(0.0, bet / pot)
        return s / (1.0 + s)

    @staticmethod
    def mdf(bet: float, pot: float) -> float:
        """Minimum Defense Frequency = pot / (pot + bet)."""
        if bet <= 0 or pot <= 0:
            return 1.0
        return max(0.0, min(1.0, pot / (pot + bet)))


# ==========================================================
# RangeParser (eval7-backed, supports weights; fallback to custom)
# ==========================================================
class RangeParser:
    # Note: These are used only for fallback and some deck creations elsewhere.
    RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    SUITS = ['s', 'h', 'd', 'c']

    def __init__(self):
        self._init_lookups()

    def _init_lookups(self):
        # Fallback lookup tables (used when eval7 is unavailable)
        self._pair_combos: Dict[str, List[str]] = {}
        self._suited_combos: Dict[str, List[str]] = {}
        self._offsuit_combos: Dict[str, List[str]] = {}
        for r in self.RANKS:
            key = r + r
            combos = []
            for i, s1 in enumerate(self.SUITS):
                for s2 in self.SUITS[i + 1:]:
                    combos.append(r + s1 + r + s2)
            self._pair_combos[key] = combos
        for i, r1 in enumerate(self.RANKS):
            for r2 in self.RANKS[i + 1:]:
                s_key = r1 + r2 + 's'
                o_key = r1 + r2 + 'o'
                self._suited_combos[s_key] = [r1 + s + r2 + s for s in self.SUITS]
                offs = []
                for s1 in self.SUITS:
                    for s2 in self.SUITS:
                        if s1 != s2:
                            offs.append(r1 + s1 + r2 + s2)
                self._offsuit_combos[o_key] = offs

    # ---------------------- eval7-backed ----------------------
    def _parse_with_eval7(self, range_str: str) -> Set[str]:
        """
        Use eval7.HandRange to parse a range string (supports weights, '+' syntax, spans, etc.).
        Returns a set of 4-char combo strings like 'AsKs', 'AhAd'.
        """
        if not _EVAL7_AVAILABLE:
            return set()
        try:
            hr = _eval7.HandRange(range_str or "")
            combos: Set[str] = set()
            # hr.hands is a list of ((Card, Card), weight) entries; include duplicates by unique combo
            for hand_tuple, weight in getattr(hr, "hands", []):
                if not hand_tuple:
                    continue
                c1, c2 = hand_tuple
                s1 = str(c1)
                s2 = str(c2)
                if len(s1) == 2 and len(s2) == 2:
                    combos.add(s1 + s2)
            return combos
        except Exception:
            # If eval7 parsing fails (invalid string), fall back
            return set()

    # ---------------------- fallback ----------------------
    def _get_combinations(self, hand_str: str) -> Set[str]:
        if len(hand_str) == 2:
            return set(self._pair_combos.get(hand_str, []))
        if len(hand_str) == 3 and hand_str.endswith('s'):
            return set(self._suited_combos.get(hand_str, []))
        if len(hand_str) == 3 and hand_str.endswith('o'):
            return set(self._offsuit_combos.get(hand_str, []))
        if len(hand_str) == 4:
            return {hand_str}
        return set()

    def _expand_plus(self, base: str) -> List[str]:
        out = []
        if len(base) == 2:
            top_idx = self.RANKS.index(base[0])
            for r in self.RANKS[:top_idx + 1]:
                out.append(r + r)
        elif len(base) == 3:
            r1, r2, suf = base[0], base[1], base[2]
            start_idx = self.RANKS.index(r2)
            for r in self.RANKS[start_idx:]:
                if r != r1:
                    out.append(r1 + r + suf)
        return out

    def _expand_hyphen(self, token: str) -> List[str]:
        try:
            lo, hi = token.split('-')
        except ValueError:
            return [token]
        if len(lo) != len(hi):
            return [token]
        if len(lo) == 3 and (lo[2] == hi[2]) and (lo[2] in ('s', 'o')):
            r1 = lo[0]
            suf = lo[2]
            r2_lo = lo[1]
            r2_hi = hi[1]
            i_lo = self.RANKS.index(r2_lo)
            i_hi = self.RANKS.index(r2_hi)
            if i_lo > i_hi:
                i_lo, i_hi = i_hi, i_lo
            return [r1 + self.RANKS[i] + suf for i in range(i_lo, i_hi + 1)]
        return [token]

    def parse_range(self, range_str: str) -> Set[str]:
        """
        Parse a PokerStove-style range string.
        - If eval7 is available, we use its robust parser (weights ignored here; see notes).
        - Otherwise, we fall back to a simpler parser (supports '+', hyphen spans, s/o/pairs).
        Returns a set of combos as 4-char strings (e.g., 'AsKs', 'AhAd').
        """
        # Try eval7 first
        if _EVAL7_AVAILABLE:
            combos = self._parse_with_eval7(range_str)
            if combos:
                return combos  # Success

        # Fallback parser
        hands: Set[str] = set()
        cleaned = (range_str or "").replace(" ", "")
        if not cleaned:
            return hands
        for part in cleaned.split(","):
            if not part:
                continue
            parts = []
            if '-' in part and '+' not in part:
                parts.extend(self._expand_hyphen(part))
            else:
                parts.append(part)
            for p in parts:
                if p.endswith("+"):
                    for b in self._expand_plus(p[:-1]):
                        hands.update(self._get_combinations(b))
                else:
                    hands.update(self._get_combinations(p))
        return hands


# ==========================================================
# EquityCalculator (exact flop/turn boards + MC; LRU cache)
#    Now eval7-accelerated for showdowns & sampling
# ==========================================================
class EquityCalculator:
    MAX_VILLAIN_SAMPLES = 100  # Reduced from 160 for memory
    PREFLOP_ITERS_PER = 2048   # Reduced from 4096
    POSTFLOP_ITERS_PER = 1024  # Reduced from 2048

    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.range_parser = RangeParser()
        # Deck in consistent ordering
        self.deck = [r + s for r in 'AKQJT98765432' for s in 'shdc']
        self._cache = OptimizedLRUCache(256, 300)  # Smaller cache, 5min TTL

    @staticmethod
    def _split_cards(s: str) -> List[str]:
        return [s[i:i + 2] for i in range(0, len(s), 2)] if s else []

    def _available_deck(self, used: Set[str]) -> List[str]:
        return [c for c in self.deck if c not in used]

    def _enum_remaining_boards(self, board_cards: List[str], deck_left: List[str], need: int):
        if need <= 0:
            yield board_cards
            return
        for picks in itertools.combinations(deck_left, need):
            yield board_cards + list(picks)

    def _eval_showdown(self, hands: List[str], board: List[str]) -> Dict[str, float]:
        """
        Evaluate showdown equities for provided full hands vs shared board.
        Returns mapping hand -> share (1.0 for sole winner, split for ties).
        Uses eval7 if available; falls back to Python evaluator otherwise.
        """
        if _EVAL7_AVAILABLE:
            # Convert board once
            e7_board = [_to_eval7_card(c) for c in board]
            values: List[Tuple[str, int]] = []
            for hand in hands:
                c1, c2 = hand[:2], hand[2:]
                e7_cards = [_to_eval7_card(c1), _to_eval7_card(c2)] + e7_board
                v = int(_eval7.evaluate(e7_cards))
                values.append((hand, v))
        else:
            values = []
            for hand in hands:
                cards7 = [hand[:2], hand[2:]] + board
                v = self.hand_evaluator.evaluate_hand(cards7)[0]
                values.append((hand, v))

        max_v = max(v for _, v in values)
        winners = [h for h, v in values if v == max_v]
        if len(winners) == 1:
            return {winners[0]: 1.0}
        share = 1.0 / len(winners)
        return {w: share for w in winners}

    def _equity_exact_board(self, hands: List[str], board_cards: List[str], deck_left: List[str], need: int) -> Dict[str, float]:
        total = 0.0
        acc = {h: 0.0 for h in hands}
        for board in self._enum_remaining_boards(board_cards, deck_left, need):
            res = self._eval_showdown(hands, board)
            for h, v in res.items():
                acc[h] += v
            total += 1.0
        if total == 0:
            return {h: 1.0 / len(hands) for h in hands}
        return {h: acc[h] / total for h in hands}

    def calculate_vs_range(self, hand: str, villain_range: str, board: str = "") -> float:
        """
        Approximate equity of 'hand' vs a villain range on an optional partial board.
        Uses eval7 for fast showdowns; samples villain combos when range is large.
        """
        key = ("HVR2", hand, villain_range, board)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        vill_all = list(self.range_parser.parse_range(villain_range))
        if not vill_all:
            self._cache.set(key, 0.5)
            return 0.5

        board_cards = self._split_cards(board)
        dead = set(board_cards + [hand[:2], hand[2:]])
        vill_all = [v for v in vill_all if ({v[:2], v[2:]} & dead) == set()]
        if not vill_all:
            self._cache.set(key, 0.5)
            return 0.5

        seed = (hash(key) & 0xFFFFFFFF)
        rng = random.Random(seed)
        if len(vill_all) > self.MAX_VILLAIN_SAMPLES:
            vill = rng.sample(vill_all, self.MAX_VILLAIN_SAMPLES)
        else:
            vill = vill_all

        total = 0.0
        valid = 0
        for v_hand in vill:
            eqs = self.calculate_equity([hand, v_hand], board, n_iterations=None)
            total += eqs.get(hand, 0.5)
            valid += 1
        res = (total / valid) if valid else 0.5
        self._cache.set(key, res)
        return res

    def calculate_equity(self, hands: List[str], board: str = "", n_iterations: Optional[int] = None) -> Dict[str, float]:
        """
        Compute equities for the provided concrete hands on an optional board.
        - Exact rollout if river/turn with <=2 missing cards.
        - Otherwise Monte Carlo with eval7-accelerated showdowns if available.
        """
        board_cards = self._split_cards(board)
        used_cards = set()
        for hand in hands:
            used_cards.add(hand[:2])
            used_cards.add(hand[2:])
        used_cards.update(board_cards)
        deck_left = self._available_deck(used_cards)

        # Exact on river/turn/flop when feasible
        remain = 5 - len(board_cards)
        if remain <= 2 and (n_iterations is None or n_iterations >= 128):
            return self._equity_exact_board(hands, board_cards, deck_left, remain)

        # Otherwise Monte Carlo
        iters = n_iterations or (self.POSTFLOP_ITERS_PER if len(board_cards) >= 3 else self.PREFLOP_ITERS_PER)
        wins = {hand: 0.0 for hand in hands}
        ties = {hand: 0.0 for hand in hands}
        seed = hash(("EQ2", tuple(hands), board, iters)) & 0xFFFFFFFF
        rng = random.Random(seed)

        if _EVAL7_AVAILABLE:
            # Pre-convert hand hole cards to eval7.Card
            hands_e7 = {h: (_to_eval7_card(h[:2]), _to_eval7_card(h[2:])) for h in hands}
            e7_board_fixed = [_to_eval7_card(c) for c in board_cards]

        for _ in range(iters):
            missing = 5 - len(board_cards)
            board_drawn = rng.sample(deck_left, missing) if missing > 0 else []
            full_board = board_cards + board_drawn

            if _EVAL7_AVAILABLE:
                e7_board = e7_board_fixed + [_to_eval7_card(c) for c in board_drawn]
                scores: List[Tuple[str, int]] = []
                for h in hands:
                    c1, c2 = hands_e7[h]
                    val = int(_eval7.evaluate([c1, c2] + e7_board))
                    scores.append((h, val))
                max_val = max(v for _, v in scores)
                winners = [h for h, v in scores if v == max_val]
            else:
                res = self._eval_showdown(hands, full_board)
                if len(res) == 1:
                    h = next(iter(res.keys()))
                    wins[h] += 1.0
                else:
                    for h, s in res.items():
                        ties[h] += s
                continue

            if len(winners) == 1:
                wins[winners[0]] += 1.0
            else:
                share = 1.0 / len(winners)
                for h in winners:
                    ties[h] += share

        total = float(iters)
        return {hand: (wins[hand] + ties[hand]) / total for hand in hands}


# ==========================================================
# Simple Card & Action Abstractions for CFR(+/DCFR)
# ==========================================================
class CardBucketer:
    """
    Range→bucket abstraction using current-board 5-7 card strength.
    Buckets are quantiles of evaluator value among candidate combos.
    """
    def __init__(self, evaluator: HandEvaluator, num_buckets: int = 6):  # Reduced from 9
        self.evaluator = evaluator
        self.num_buckets = max(3, int(num_buckets))

    def _score_combo_on_board(self, combo: str, board: List[str]) -> int:
        cards = [combo[:2], combo[2:]] + board
        val, _ = self.evaluator.evaluate_hand(cards)
        return val

    def bucketize(self, combos: List[str], board: List[str]) -> Tuple[Dict[str, int], List[List[str]]]:
        if not combos:
            return {}, [[] for _ in range(self.num_buckets)]
        scores = [(c, self._score_combo_on_board(c, board)) for c in combos]
        scores.sort(key=lambda x: x[1])  # ascending
        buckets: List[List[str]] = [[] for _ in range(self.num_buckets)]
        mapping: Dict[str, int] = {}
        n = len(scores)
        for i, (c, _) in enumerate(scores):
            b = min(self.num_buckets - 1, int(i * self.num_buckets / n))
            buckets[b].append(c)
            mapping[c] = b
        return mapping, buckets


class ActionAbstraction:
    """
    Discrete size sets per decision node.
    For simplicity, we allow at most 1 raise round (bet/raise + call/fold).
    """
    def __init__(self, pot_fractions_bet: List[float] = None, pot_fractions_raise: List[float] = None):
        self.bet_fracs = pot_fractions_bet or [0.33, 0.66, 1.00]
        self.raise_fracs = pot_fractions_raise or [2.2, 3.5]  # multiples of facing amount

    def bet_sizes(self, pot: int, min_bet: int, max_bet: int, eff: int) -> List[int]:
        out = set()
        for f in self.bet_fracs:
            s = int(round(pot * f))
            s = max(min_bet, min(s, max_bet, eff))
            if s > 0:
                out.add(s)
        if min_bet > 0:
            out.add(min_bet)
        if max_bet > 0:
            out.add(min(max_bet, eff))
        return sorted(set([x for x in out if x > 0]))

    def raise_to_sizes(self, to_call: int, pot: int, min_raise: int, max_raise: int, eff: int, current_bet_high: int) -> List[int]:
        """
        Return TOTAL raise-to amounts (not increments). Use both multiples and pot-based raises.
        """
        out = set()
        # Multiples of facing amount (sizing norms)
        for m in self.raise_fracs:
            Y = int(round(m * to_call))
            Y = max(min_raise, min(Y, max_raise, eff + current_bet_high))
            out.add(Y)
        # Pot raises heuristic: raise to current_bet_high + pot + 2*to_call
        pot_raise = current_bet_high + pot + 2 * to_call
        out.add(max(min_raise, min(pot_raise, max_raise, eff + current_bet_high)))
        out.add(max(min_raise, min(max_raise, eff + current_bet_high)))
        return sorted(set([x for x in out if x > current_bet_high]))


# ==========================================================
# CFR(+/DCFR) Spot Solver (Heads-Up, one street with one raise)
# ==========================================================
@dataclass
class CFRConfig:
    iterations: int = 200  # Reduced from 400
    variant: str = "dcfr"  # "cfr+", "dcfr"
    regret_discount: float = 0.98  # DCFR regret decay per iter
    strat_discount: float = 0.99   # DCFR avg strategy decay per iter


class CFRNode:
    def __init__(self, key: str, actions: List[str]):
        self.key = key
        self.actions = actions[:]  # e.g., ["check","bet:40"]
        self.nA = len(actions)
        self.regrets = np.zeros(self.nA, dtype=np.float64)
        self.strategy_sum = np.zeros(self.nA, dtype=np.float64)
        self.strategy = np.full(self.nA, 1.0 / self.nA, dtype=np.float64)

    def get_strategy(self) -> np.ndarray:
        # Regret-matching+
        pos = np.maximum(self.regrets, 0.0)
        total = pos.sum()
        if total > 1e-12:
            self.strategy = pos / total
        else:
            self.strategy = np.full(self.nA, 1.0 / self.nA)
        return self.strategy

    def get_avg_strategy(self) -> np.ndarray:
        s = self.strategy_sum
        tot = s.sum()
        if tot > 1e-12:
            return s / tot
        return np.full(self.nA, 1.0 / self.nA)


class CFRSpotSolver:
    """
    Heads-up per-street CFR(+/DCFR) using action abstraction and showdown rollouts for called pots.
    Limits: 1 bet/raise round (i.e., bet, raise, call). Check-bet allowed. Single re-raise is capped.
    Utilities measured in chips relative to current state (fold=0 when facing a bet; bet&fold win current pot).
    """
    def __init__(self, equity_calc: EquityCalculator, action_abs: ActionAbstraction, cfg: CFRConfig = CFRConfig()):
        self.eq = equity_calc
        self.abs = action_abs
        self.cfg = cfg
        self.nodes: Dict[str, CFRNode] = {}
        self.range_parser = self.eq.range_parser

    # --------- Helpers ---------
    @staticmethod
    def _key(player: str, history: str, tag: str = "") -> str:
        return f"{player}|{history}|{tag}"

    @staticmethod
    def _parse_size(action: str) -> int:
        # e.g., "bet:40" or "raise:220"
        try:
            return int(action.split(":")[1])
        except Exception:
            return 0

    def _terminal_showdown_ev(self, hero_hand: str, villain_range: str, board: List[str], pot: int) -> float:
        """
        Checkdown to river from current board. EV in chips for Hero.
        """
        bstr = "".join(board)
        eq = self.eq.calculate_vs_range(hero_hand, villain_range, bstr)
        return eq * pot

    def _terminal_call_ev(self, hero_hand: str, villain_range: str, board: List[str], pot: int, facing: int) -> float:
        """
        Call a bet of size 'facing'. Called pot size = pot + facing + facing.
        Hero invests 'facing'. EV = eq*(pot+2*facing) - facing
        """
        bstr = "".join(board)
        eq = self.eq.calculate_vs_range(hero_hand, villain_range, bstr)
        return eq * (pot + 2 * facing) - facing

    def _terminal_call_after_raise_ev(self, hero_hand: str, villain_range: str, board: List[str], pot: int, hero_add: int, vill_face: int) -> float:
        """
        After we raise to R and they call: pot becomes pot + hero_add + vill_face; hero invested hero_add.
        """
        bstr = "".join(board)
        eq = self.eq.calculate_vs_range(hero_hand, villain_range, bstr)
        return eq * (pot + hero_add + vill_face) - hero_add

    def _node(self, key: str, actions: List[str]) -> CFRNode:
        if key not in self.nodes:
            self.nodes[key] = CFRNode(key, actions)
        return self.nodes[key]

    # --------- Public interface ---------
    def solve(
            self,
            hero_hand: str,
            villain_range: str,
            board: List[str],
            pot: int,
            to_call: int,
            hero_stack: int,
            villain_stack: int,
            min_bet: int,
            max_bet: int,
            min_raise_to: int,
            max_raise_to: int,
            current_bet_high: int,
            stage: str,
            villain_profile_hint: str = "tag") -> Dict[str, Any]:
        """
        Return dict with hero root average strategy and mapped concrete actions.
        """
        eff = max(0, min(hero_stack, villain_stack))
        # Build root actions
        if to_call <= 0:
            root_actions = ["check"]
            bet_sizes = self.abs.bet_sizes(pot, min_bet, max_bet, eff)
            for b in bet_sizes:
                if b > 0:
                    root_actions.append(f"bet:{b}")
        else:
            root_actions = ["fold", "call"]
            raise_tos = self.abs.raise_to_sizes(to_call, pot, min_raise_to, max_raise_to, eff, current_bet_high)
            for r in raise_tos:
                if r > current_bet_high:
                    root_actions.append(f"raise:{r}")

        # Optional: nudge initial regrets by villain profile (nit defends less, station defends more)
        profile_bias = {'nit': -0.20, 'tag': 0.0, 'lag': +0.05, 'station': +0.12}.get(villain_profile_hint, 0.0)

        # CFR recursion
        def cfr(history: str, player: int, pot_now: int, to_call_now: int,
                hero_reach: float, vill_reach: float) -> float:
            """
            Returns utility for Hero at this node. player=0 (Hero), 1 (Villain).
            """
            # Build actions for this node (respecting one betting round with 1 raise)
            if player == 0:  # Hero to act
                if len(history) == 0:
                    actions = root_actions
                elif history.endswith("c|"):  # after villain check, hero acts
                    acts = ["check"]
                    bs = self.abs.bet_sizes(pot_now, min_bet, max_bet, eff)
                    for b in bs:
                        acts.append(f"bet:{b}")
                    actions = acts
                elif history.endswith("b|") or "|b" in history:
                    if "|r" in history:
                        actions = ["fold", "call"]
                    else:
                        acts = ["fold", "call"]
                        last_size = int(history.split("#")[-1])
                        raise_tos_local = self.abs.raise_to_sizes(last_size, pot_now, min_raise_to, max_raise_to, eff, current_bet_high)
                        for r in raise_tos_local:
                            acts.append(f"raise:{r}")
                        actions = acts
                else:
                    actions = ["check"]
            else:  # Villain to act
                if history.endswith("c|"):  # after hero check, villain acts
                    acts = ["check"]
                    bs = self.abs.bet_sizes(pot_now, min_bet, max_bet, eff)
                    for b in bs:
                        acts.append(f"bet:{b}")
                    actions = acts
                elif history.endswith("b|") or "|b" in history:
                    if "|r" in history:
                        actions = ["fold", "call"]
                    else:
                        acts = ["fold", "call"]
                        last_size = int(history.split("#")[-1])
                        raise_tos_local = self.abs.raise_to_sizes(last_size, pot_now, min_raise_to, max_raise_to, eff, current_bet_high)
                        for r in raise_tos_local:
                            acts.append(f"raise:{r}")
                        actions = acts
                else:
                    acts = ["fold", "call"]
                    last_size = int(history.split("#")[-1]) if "#" in history else to_call_now
                    raise_tos_local = self.abs.raise_to_sizes(last_size, pot_now, min_raise_to, max_raise_to, eff, current_bet_high)
                    for r in raise_tos_local:
                        acts.append(f"raise:{r}")
                    actions = acts

            # Infoset key
            who = "H" if player == 0 else "V"
            key = self._key(who, history if history else "root", tag=str(to_call_now))
            node = self._node(key, actions)

            # Strategy with optional profile bias at Villain nodes (initialization effect)
            strat = node.get_strategy().copy()
            if player == 1 and profile_bias != 0.0:
                bias = np.ones_like(strat)
                for i, a in enumerate(actions):
                    if a == "call":
                        bias[i] *= (1.0 + profile_bias)
                    elif a.startswith("fold"):
                        bias[i] *= (1.0 - profile_bias * 0.6)
                bias = np.maximum(bias, 1e-6)
                strat = bias * strat
                ssum = strat.sum()
                if ssum > 0:
                    strat /= ssum

            # Recurse for utilities
            util = np.zeros(len(actions), dtype=np.float64)

            for i, a in enumerate(actions):
                if a == "check":
                    if player == 0:
                        util[i] = cfr(history + "c|", 1, pot_now, 0, hero_reach * strat[i], vill_reach)
                    else:
                        util[i] = self._terminal_showdown_ev(hero_hand, villain_range, board, pot_now)
                elif a == "fold":
                    if player == 0:
                        util[i] = 0.0
                    else:
                        util[i] = pot_now
                elif a == "call":
                    facing = to_call_now if to_call_now > 0 else int(history.split("#")[-1]) if "#" in history else 0
                    util[i] = self._terminal_call_ev(hero_hand, villain_range, board, pot_now, facing)
                elif a.startswith("bet:"):
                    size = self._parse_size(a)
                    next_hist = history + f"b|#{size}"
                    if player == 0:
                        util[i] = cfr(next_hist, 1, pot_now, size, hero_reach * strat[i], vill_reach)
                    else:
                        util[i] = cfr(next_hist, 0, pot_now, size, hero_reach, vill_reach * strat[i])
                elif a.startswith("raise:"):
                    to = self._parse_size(a)
                    last_size = to_call_now if to_call_now > 0 else int(history.split("#")[-1]) if "#" in history else 0
                    hero_add = max(0, to - (0 if player == 0 else 0))
                    vill_face = max(0, to - last_size)
                    if player == 0:
                        ev_fold = pot_now
                        ev_call = self._terminal_call_after_raise_ev(hero_hand, villain_range, board, pot_now, hero_add, vill_face)
                        opp_key = self._key("V", history + f"r|#{to}", tag=str(vill_face))
                        opp_node = self.nodes.get(opp_key)
                        if opp_node:
                            opp_strat = opp_node.get_avg_strategy()
                            p_fold = 0.0
                            p_call = 0.0
                            for j, name in enumerate(opp_node.actions):
                                if name == "fold":
                                    p_fold += opp_strat[j]
                                if name == "call":
                                    p_call += opp_strat[j]
                            norm = p_fold + p_call
                            if norm > 1e-9:
                                p_fold /= norm
                                p_call /= norm
                            else:
                                mdf = GTOUtils.mdf(vill_face, pot_now)
                                p_call = mdf
                                p_fold = 1 - mdf
                        else:
                            mdf = GTOUtils.mdf(vill_face, pot_now)
                            p_call = mdf
                            p_fold = 1 - mdf
                        util[i] = p_fold * ev_fold + p_call * ev_call
                    else:
                        ev_fold = 0.0
                        ev_call = self._terminal_call_after_raise_ev(hero_hand, villain_range, board, pot_now, vill_face, hero_add)
                        opp_key = self._key("H", history + f"r|#{to}", tag=str(vill_face))
                        opp_node = self.nodes.get(opp_key)
                        if opp_node:
                            opp_strat = opp_node.get_avg_strategy()
                            p_fold = 0.0
                            p_call = 0.0
                            for j, name in enumerate(opp_node.actions):
                                if name == "fold":
                                    p_fold += opp_strat[j]
                                if name == "call":
                                    p_call += opp_strat[j]
                            norm = p_fold + p_call
                            if norm > 1e-9:
                                p_fold /= norm
                                p_call /= norm
                            else:
                                po = GTOUtils.caller_pot_odds_from_size(vill_face, pot_now)
                                p_call = 1.0 if self.eq.calculate_vs_range(hero_hand, villain_range, "".join(board)) > po else 0.0
                                p_fold = 1.0 - p_call
                        else:
                            po = GTOUtils.caller_pot_odds_from_size(vill_face, pot_now)
                            p_call = 1.0 if self.eq.calculate_vs_range(hero_hand, villain_range, "".join(board)) > po else 0.0
                            p_fold = 1.0 - p_call
                        util[i] = p_fold * ev_fold + p_call * ev_call
                else:
                    util[i] = 0.0

            node_util = float(np.dot(strat, util))

            # Regret & strategy updates
            if player == 0:
                regrets = util - node_util
                node.regrets = np.maximum(0.0, self.cfg.regret_discount * node.regrets + vill_reach * regrets) \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    np.maximum(0.0, node.regrets + vill_reach * regrets)
                node.strategy_sum = self.cfg.strat_discount * node.strategy_sum + hero_reach * strat \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    node.strategy_sum + hero_reach * strat
            else:
                regrets = -(util - node_util)  # utility is for Hero; Villain regrets inverted
                node.regrets = np.maximum(0.0, self.cfg.regret_discount * node.regrets + hero_reach * regrets) \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    np.maximum(0.0, node.regrets + hero_reach * regrets)
                node.strategy_sum = self.cfg.strat_discount * node.strategy_sum + vill_reach * strat \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    node.strategy_sum + vill_reach * strat

            return node_util

        # Run iterations
        for _ in range(max(1, self.cfg.iterations)):
            cfr(history="", player=0, pot_now=pot, to_call_now=to_call, hero_reach=1.0, vill_reach=1.0)

        # Root strategy
        root_key = self._key("H", "root", tag=str(to_call))
        root_node = self.nodes.get(root_key)
        if not root_node:
            # Fallback: trivial check or call if no node created
            if to_call <= 0:
                return {"strategy": {"check": 1.0}, "actions": ["check"], "probs": [1.0]}
            else:
                return {"strategy": {"call": 1.0}, "actions": ["call"], "probs": [1.0]}

        avg = root_node.get_avg_strategy()
        actions = root_node.actions
        strat_map = {a: float(avg[i]) for i, a in enumerate(actions)}
        return {"strategy": strat_map, "actions": actions, "probs": [float(x) for x in avg]}


# ==========================================================
# Full GTOSolver: Range-vs-Range DCFR with bucket abstraction & chance sampling
# ==========================================================
class GTOSolver:
    """
    Multi-street, heads-up DCFR solver with:
      - Card bucketing (range→buckets) per street
      - Action abstraction (bet sizes, raise-to sizes; 1 raise per street)
      - Chance sampling across future streets (flop/turn/river)
      - Opponent modeling bias on defense frequencies
    Notes:
      - Uses "frozen ranges": actions do not update distributions; this is a common
        tractable simplification for on-the-fly guidance.
      - Private information sets are conditioned on each player's bucket.
      - Equity between buckets is estimated via sampling (cached).
    """

    def __init__(
        self,
        equity_calc: EquityCalculator,
        action_abs: ActionAbstraction,
        bucketer: CardBucketer = None,
        cfg: CFRConfig = CFRConfig(iterations=150, variant="dcfr"),  # Reduced iterations
        chance_samples_per_street: int = 4,  # Reduced from 6
        max_bucket_pairs_eval: int = 16,  # Reduced from 24
    ):
        self.eq = equity_calc
        self.abs = action_abs
        self.bucketer = bucketer or CardBucketer(HandEvaluator(), num_buckets=6)  # Reduced buckets
        self.cfg = cfg
        self.chance_samples = max(1, int(chance_samples_per_street))
        self.max_bucket_pairs_eval = max(8, int(max_bucket_pairs_eval))
        self.nodes: Dict[str, CFRNode] = {}
        self.range_parser = self.eq.range_parser
        self._rng = random.Random(0xC0FFEE)

        # Cache for bucket pair equity on a given board & street
        self._eq_cache: Dict[Tuple[str, str, int, int, str], float] = {}  # (stage, boardstr, hB, vB, keytag)->eq

    # ---------- Utility helpers ----------
    @staticmethod
    def _stage_order(stage: str) -> int:
        order = {"preflop": 0, "flop": 1, "turn": 2, "river": 3}
        return order.get(stage, 3)

    @staticmethod
    def _next_stage(stage: str) -> str:
        return { "preflop": "flop", "flop": "turn", "turn": "river", "river": "river" }[stage]

    def _board_needed(self, stage: str) -> int:
        return {"preflop": 3, "flop": 1, "turn": 1, "river": 0}[stage]

    @staticmethod
    def _board_to_str(board: List[str]) -> str:
        return "".join(board)

    # ---------- Range preparation ----------
    def _filter_and_cap(self, combos: List[str], dead: Set[str], cap: int = 200) -> List[str]:  # Reduced cap
        filtered = [c for c in combos if ({c[:2], c[2:]} & dead) == set()]
        if len(filtered) <= cap:
            return filtered
        return random.sample(filtered, cap)

    def _villain_bucket_data(self, vill_range_str: str, board: List[str], dead: Set[str]) -> Tuple[Dict[int, List[str]], Dict[int, float], List[List[str]]]:
        combos = list(self.range_parser.parse_range(vill_range_str))
        combos = self._filter_and_cap(combos, dead, cap=200)  # Reduced cap
        mapping, buckets = self.bucketer.bucketize(combos, board)
        bucket_lists: Dict[int, List[str]] = defaultdict(list)
        for c, b in mapping.items():
            bucket_lists[b].append(c)
        total = sum(len(v) for v in bucket_lists.values()) or 1
        probs = {b: len(v) / total for b, v in bucket_lists.items()}
        return bucket_lists, probs, buckets

    def _hero_bucket(self, hero_hand: str, board: List[str]) -> int:
        mapping, _ = self.bucketer.bucketize([hero_hand], board)
        return mapping.get(hero_hand, 0)

    # ---------- Bucket equity estimation ----------
    def _bucket_equity(self, stage: str, board: List[str], hero_bucket: int, vill_bucket: int,
                       hero_combo: str, vill_bucket_lists: Dict[int, List[str]]) -> float:
        key = (stage, self._board_to_str(board), hero_bucket, vill_bucket, hero_combo)
        if key in self._eq_cache:
            return self._eq_cache[key]
        # Sample up to max_bucket_pairs_eval villain combos from the bucket, avoid card collisions
        vlist = vill_bucket_lists.get(vill_bucket, [])
        if not vlist:
            eq = 0.5
            self._eq_cache[key] = eq
            return eq
        samples = min(self.max_bucket_pairs_eval, len(vlist))
        picks = random.sample(vlist, samples)
        used_board = set(board + [hero_combo[:2], hero_combo[2:]])
        total = 0.0
        cnt = 0
        bstr = self._board_to_str(board)
        for vc in picks:
            if {vc[:2], vc[2:]} & used_board:
                continue
            eq_map = self.eq.calculate_equity([hero_combo, vc], bstr, n_iterations=128 if stage != "river" else None)  # Reduced iterations
            total += eq_map.get(hero_combo, 0.5)
            cnt += 1
        eq = (total / cnt) if cnt else 0.5
        self._eq_cache[key] = eq
        return eq

    # ---------- Node management ----------
    @staticmethod
    def _node_key(player: int, stage: str, history: str, my_bucket: int, to_call: int) -> str:
        # player: 0 hero, 1 villain
        return f"{player}|{stage}|{history}|b{my_bucket}|tc{to_call}"

    def _node(self, key: str, actions: List[str]) -> CFRNode:
        if key not in self.nodes:
            self.nodes[key] = CFRNode(key, actions)
        return self.nodes[key]

    # ---------- Chance: roll next street board ----------
    def _sample_next_board(self, stage: str, board: List[str], hero_combo: str, vill_combo_sample: str = "") -> List[str]:
        need = self._board_needed(stage)
        if need <= 0:
            return board
        used = set(board + [hero_combo[:2], hero_combo[2:]])
        if vill_combo_sample:
            used |= {vill_combo_sample[:2], vill_combo_sample[2:]}
        deck = [r + s for r in 'AKQJT98765432' for s in 'shdc']
        avail = [c for c in deck if c not in used]
        return board + random.sample(avail, need)

    # ---------- DCFR solve ----------
    def solve(
        self,
        hero_hand: str,
        villain_range: str,
        board: List[str],
        pot: int,
        to_call: int,
        hero_stack: int,
        villain_stack: int,
        min_bet: int,
        max_bet: int,
        min_raise_to: int,
        max_raise_to: int,
        current_bet_high: int,
        stage: str,
        villain_profile_hint: str = "tag",
    ) -> Dict[str, Any]:
        """
        Returns a dict:
          {
            "bucket": hero_bucket_index,
            "actions": [...],
            "probs": [...],
            "strategy_by_bucket": {bucket_index: {action: prob, ...}, ...}
          }
        """
        self.nodes.clear()
        self._eq_cache.clear()
        eff = max(0, min(hero_stack, villain_stack))
        stage = stage.lower()
        if stage == "preflop":
            # We support multi-street from flop forward for tractability.
            stage = "flop"  # Treat as flop solve with empty board and chance sampling 3 cards
            board = []

        # Prepare ranges & buckets
        dead = set(board + [hero_hand[:2], hero_hand[2:]])
        v_bucket_lists, v_bucket_probs, _ = self._villain_bucket_data(villain_range, board, dead)
        if not v_bucket_lists:
            return {"bucket": 0, "actions": ["check" if to_call == 0 else "call"], "probs": [1.0], "strategy_by_bucket": {}}
        h_bucket = self._hero_bucket(hero_hand, board)

        # Opponent defense bias
        prof_bias = {'nit': -0.15, 'tag': 0.0, 'lag': +0.05, 'station': +0.12}.get(villain_profile_hint, 0.0)

        # Action lists at root depend on to_call
        def root_actions_for_state(pot_now: int, to_call_now: int) -> List[str]:
            if to_call_now <= 0:
                acts = ["check"]
                for b in self.abs.bet_sizes(pot_now, min_bet, max_bet, eff):
                    acts.append(f"bet:{b}")
                return acts
            else:
                acts = ["fold", "call"]
                for r in self.abs.raise_to_sizes(max(1, to_call_now), pot_now, min_raise_to, max_raise_to, eff, current_bet_high):
                    acts.append(f"raise:{r}")
                return acts

        # Recursive DCFR with chance sampling to river; at terminal (river showdown) uses bucket equity
        def cfr(player: int, stage_now: str, history: str, pot_now: int, to_call_now: int,
                raised_this_street: bool, last_was_check: bool,
                hero_bucket_now: int, vill_bucket_now: int,
                hero_reach: float, vill_reach: float) -> float:
            # Terminal if river and both checked
            if stage_now == "river" and to_call_now == 0 and last_was_check and history.endswith("|c"):
                eq = self._bucket_equity(stage_now, board, hero_bucket_now, vill_bucket_now, hero_hand, v_bucket_lists)
                return eq * pot_now

            # Build actions
            if to_call_now <= 0:
                actions = ["check"]
                for b in self.abs.bet_sizes(pot_now, min_bet, max_bet, eff):
                    actions.append(f"bet:{b}")
            else:
                if raised_this_street:
                    actions = ["fold", "call"]
                else:
                    actions = ["fold", "call"]
                    for r in self.abs.raise_to_sizes(max(1, to_call_now), pot_now, min_raise_to, max_raise_to, eff, current_bet_high):
                        actions.append(f"raise:{r}")

            # Node key is private-bucket aware
            my_bucket = hero_bucket_now if player == 0 else vill_bucket_now
            key = self._node_key(player, stage_now, history if history else "root", my_bucket, to_call_now)
            node = self._node(key, actions)

            # Strategy
            strat = node.get_strategy().copy()
            if player == 1 and prof_bias != 0.0:
                # Villain call-fold tilt
                bias = np.ones_like(strat)
                for i, a in enumerate(actions):
                    if a == "call":
                        bias[i] *= (1.0 + prof_bias)
                    elif a == "fold":
                        bias[i] *= (1.0 - 0.6 * prof_bias)
                bias = np.maximum(bias, 1e-6)
                strat *= bias
                ssum = strat.sum()
                if ssum > 0:
                    strat /= ssum

            util = np.zeros(len(actions), dtype=np.float64)

            for i, a in enumerate(actions):
                if a == "check":
                    # If both checked in a row -> next street (chance sampling)
                    if last_was_check:
                        # Sample next street board once
                        next_stage = self._next_stage(stage_now)
                        if next_stage == "river":
                            # Sample final card if any needed
                            next_board = self._sample_next_board(stage_now, board, hero_hand)
                            # Estimate river checkdown equity
                            eq_key = (next_stage, self._board_to_str(next_board), hero_bucket_now, vill_bucket_now, hero_hand)
                            if eq_key in self._eq_cache:
                                eq = self._eq_cache[eq_key]
                            else:
                                bstr = self._board_to_str(next_board)
                                total = 0.0
                                cnt = 0
                                v_candidates = []
                                for b_idx, combos in v_bucket_lists.items():
                                    if not combos:
                                        continue
                                    w = v_bucket_probs.get(b_idx, 0.0)
                                    if w <= 0:
                                        continue
                                    take = max(1, int(self.max_bucket_pairs_eval * w))
                                    v_candidates.extend(random.sample(combos, min(take, len(combos))))
                                if not v_candidates:
                                    eq = 0.5
                                else:
                                    for vc in random.sample(v_candidates, min(len(v_candidates), self.max_bucket_pairs_eval)):
                                        if {vc[:2], vc[2:]} & set(next_board + [hero_hand[:2], hero_hand[2:]]):
                                            continue
                                        eq_map = self.eq.calculate_equity([hero_hand, vc], bstr, n_iterations=None)
                                        total += eq_map.get(hero_hand, 0.5)
                                        cnt += 1
                                    eq = (total / cnt) if cnt else 0.5
                                self._eq_cache[eq_key] = eq
                            util[i] = eq * pot_now
                        else:
                            # Advance to next street with chance sampling multiple times and average
                            saved_board = list(board)
                            vals = 0.0
                            n = 0
                            for _ in range(self.chance_samples):
                                next_board = self._sample_next_board(stage_now, board, hero_hand)
                                board[:] = next_board
                                # Re-bucket hero for updated board
                                next_h_bucket = self._hero_bucket(hero_hand, board)
                                v_b = vill_bucket_now  # frozen bucket id
                                vals += cfr(1 - player, next_stage, "", pot_now, 0, False, False,
                                            next_h_bucket, v_b,
                                            hero_reach * strat[i] if player == 0 else hero_reach,
                                            vill_reach * strat[i] if player == 1 else vill_reach)
                                n += 1
                            util[i] = (vals / max(1, n))
                            board[:] = saved_board
                    else:
                        # Pass action to opponent on same street
                        util[i] = cfr(1 - player, stage_now, (history + "|c")[-40:], pot_now, 0,
                                      raised_this_street, True,
                                      hero_bucket_now, vill_bucket_now,
                                      hero_reach * strat[i] if player == 0 else hero_reach,
                                      vill_reach * strat[i] if player == 1 else vill_reach)
                elif a.startswith("bet:"):
                    size = int(a.split(":")[1])
                    # Move to opponent, facing 'size'
                    util[i] = cfr(1 - player, stage_now, (history + f"|b{size}")[-40:], pot_now, size,
                                  raised_this_street, False,
                                  hero_bucket_now, vill_bucket_now,
                                  hero_reach * strat[i] if player == 0 else hero_reach,
                                  vill_reach * strat[i] if player == 1 else vill_reach)
                elif a == "fold":
                    # Fold vs bet/raise: bettor (opponent) wins current pot
                    if player == 0:
                        util[i] = 0.0  # hero folds -> 0 (baseline at node)
                    else:
                        util[i] = pot_now  # villain folds -> hero wins pot
                elif a == "call":
                    # Resolve call: if river → showdown; else chance to next street with updated pot
                    facing = to_call_now
                    if facing <= 0 and "#" in history:
                        try:
                            facing = int(''.join(ch for ch in history.split("|")[-1] if ch.isdigit()))
                        except Exception:
                            facing = to_call_now
                    new_pot = pot_now + 2 * facing
                    if stage_now == "river":
                        eq = self._bucket_equity(stage_now, board, hero_bucket_now, vill_bucket_now, hero_hand, v_bucket_lists)
                        util[i] = eq * new_pot - (facing if player == 0 else 0)
                    else:
                        # Chance sample next street
                        next_stage = self._next_stage(stage_now)
                        saved_board = list(board)
                        vals = 0.0
                        n = 0
                        for _ in range(self.chance_samples):
                            next_board = self._sample_next_board(stage_now, board, hero_hand)
                            board[:] = next_board
                            next_h_bucket = self._hero_bucket(hero_hand, board)
                            v_b = vill_bucket_now
                            vals += cfr(1 - player, next_stage, "", new_pot, 0, False, False,
                                        next_h_bucket, v_b,
                                        hero_reach * strat[i] if player == 0 else hero_reach,
                                        vill_reach * strat[i] if player == 1 else vill_reach)
                            n += 1
                        util[i] = (vals / max(1, n)) - (facing if player == 0 else 0)
                        board[:] = saved_board
                elif a.startswith("raise:"):
                    if raised_this_street:
                        util[i] = 0.0
                    else:
                        to_amt = int(a.split(":")[1])
                        util[i] = cfr(1 - player, stage_now, (history + f"|r{to_amt}")[-40:], pot_now, to_amt,
                                      True, False,
                                      hero_bucket_now, vill_bucket_now,
                                      hero_reach * strat[i] if player == 0 else hero_reach,
                                      vill_reach * strat[i] if player == 1 else vill_reach)
                else:
                    util[i] = 0.0

            node_util = float(np.dot(strat, util))

            # Regret & strategy updates
            if player == 0:
                regrets = util - node_util
                node.regrets = np.maximum(0.0, self.cfg.regret_discount * node.regrets + vill_reach * regrets) \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    np.maximum(0.0, node.regrets + vill_reach * regrets)
                node.strategy_sum = self.cfg.strat_discount * node.strategy_sum + hero_reach * strat \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    node.strategy_sum + hero_reach * strat
            else:
                regrets = -(util - node_util)
                node.regrets = np.maximum(0.0, self.cfg.regret_discount * node.regrets + hero_reach * regrets) \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    np.maximum(0.0, node.regrets + hero_reach * regrets)
                node.strategy_sum = self.cfg.strat_discount * node.strategy_sum + vill_reach * strat \
                    if self.cfg.variant.lower().startswith("dcfr") else \
                    node.strategy_sum + vill_reach * strat

            return node_util

        # Run iterations of DCFR from root
        root_actions = root_actions_for_state(pot, to_call)
        root_key = self._node_key(0, stage, "root", h_bucket, to_call)
        self._node(root_key, root_actions)
        for _ in range(max(1, self.cfg.iterations)):
            # Sample villain bucket each iter (external sampling)
            vb_choices = [(b, p) for b, p in v_bucket_probs.items() if p > 0]
            if not vb_choices:
                sampled_vb = 0
            else:
                r = random.random()
                acc = 0.0
                sampled_vb = vb_choices[-1][0]
                for b, p in vb_choices:
                    acc += p
                    if r <= acc:
                        sampled_vb = b
                        break
            cfr(0, stage, "root", pot, to_call, False, False, h_bucket, sampled_vb, 1.0, 1.0)

        # Extract hero root avg strategy for hero bucket
        node = self.nodes.get(root_key)
        if not node:
            return {"bucket": h_bucket, "actions": ["check" if to_call == 0 else "call"], "probs": [1.0], "strategy_by_bucket": {}}
        avg = node.get_avg_strategy()
        actions = node.actions
        strat_map = {actions[i]: float(avg[i]) for i in range(len(actions))}

        # Strategies for all hero buckets at root
        strategy_by_bucket: Dict[int, Dict[str, float]] = {h_bucket: strat_map}
        for key, n in self.nodes.items():
            try:
                pid, stg, hist, btag, tc = key.split("|")
                if pid == '0' and stg == stage and hist == "root":
                    b = int(btag[1:])
                    av = n.get_avg_strategy()
                    strategy_by_bucket[b] = {n.actions[i]: float(av[i]) for i in range(len(n.actions))}
            except Exception:
                continue

        return {"bucket": h_bucket, "actions": actions, "probs": [float(x) for x in avg], "strategy_by_bucket": strategy_by_bucket}


# ==========================================================
# MonteCarloSimulator (multiway range equity; exact board when cheap)
#    eval7-accelerated for showdowns
# ==========================================================
class MonteCarloSimulator:
    def __init__(self, n_jobs: int = -1):
        self.hand_evaluator = HandEvaluator()
        self.range_parser = RangeParser()
        self.deck = [r + s for r in RangeParser.RANKS for s in RangeParser.SUITS]
        self.equity_calc = EquityCalculator()

    def _parse_range(self, range_str: str) -> List[str]:
        return list(self.range_parser.parse_range(range_str)) if range_str else []

    def simulate_multiway_range_equity(self, hero: str, villain_ranges: List[str], board: str = "", n_iterations: int = 300) -> float:  # Reduced iterations
        """
        Approximate hero equity vs multiple opponents' ranges (2–8 villains).
        Uses exact rollouts on turn/river, MC otherwise. eval7 accelerates the showdown evals.
        """
        all_ranges = [self._parse_range(r) for r in villain_ranges]
        board_cards = {board[i:i + 2] for i in range(0, len(board), 2)}
        hero_cards = {hero[:2], hero[2:]}
        all_ranges = [{h for h in hands if not ({h[:2], h[2:]} & (board_cards | hero_cards))} for hands in all_ranges]
        pools = [list(h) for h in all_ranges if h]
        if not pools:
            return 0.5

        seed = hash(("MW2", hero, tuple(villain_ranges), board)) & 0xFFFFFFFF
        rng = random.Random(seed)

        # Exact on flop/turn
        board_list = [board[i:i + 2] for i in range(0, len(board), 2)]
        missing = 5 - len(board_list)
        if missing <= 2:
            tot = 0.0
            cnt = 0
            for _ in range(n_iterations):
                used = set(hero_cards) | set(board_list)
                drawn = []
                ok = True
                for pool in pools:
                    for _try in range(80):
                        pick = rng.choice(pool)
                        c1, c2 = pick[:2], pick[2:]
                        if (c1 not in used) and (c2 not in used):
                            drawn.append(pick)
                            used.add(c1)
                            used.add(c2)
                            break
                    else:
                        ok = False
                        break
                if not ok:
                    continue
                eq = self.equity_calc.calculate_equity([hero] + drawn, board, n_iterations=None)
                tot += eq.get(hero, 0.5)
                cnt += 1
            return (tot / cnt) if cnt else 0.5

        # Otherwise MC
        total = 0.0
        samples = 0
        for _ in range(n_iterations):
            used = set(hero_cards) | board_cards
            drawn = []
            ok = True
            for pool in pools:
                for _try in range(80):
                    pick = rng.choice(pool)
                    c1, c2 = pick[:2], pick[2:]
                    if (c1 not in used) and (c2 not in used):
                        drawn.append(pick)
                        used.add(c1)
                        used.add(c2)
                        break
                else:
                    ok = False
                    break
            if not ok:
                continue
            eq = self.equity_calc.calculate_equity([hero] + drawn, board, n_iterations=32)  # Reduced iterations
            total += eq.get(hero, 0.5)
            samples += 1
        return total / samples if samples else 0.5


# ==========================================================
# ICM
# ==========================================================
@dataclass
class GameState:
    player: int
    pot: float
    board: str
    actions: List[str]
    hero_range: Dict[str, float]
    villain_range: Dict[str, float]


from functools import lru_cache


class ICMCalculator:
    def __init__(self, cache_size: int = 128):  # Reduced cache
        self._calculate_icm_cached = lru_cache(maxsize=cache_size)(self._calculate_icm)

    def calculate_equity(self, stacks: List[int], payouts: List[float]) -> List[float]:
        if len(stacks) < len(payouts):
            return [0.0] * len(stacks)
        total = sum(stacks) or 1.0
        ns = [s / total for s in stacks]
        return [sum(self._calculate_icm_cached(tuple(ns), i, j) * p for j, p in enumerate(payouts)) for i in range(len(stacks))]

    def _calculate_icm(self, stacks: Tuple[float, ...], player: int, position: int) -> float:
        # Simple recursive ICM approximator
        n = len(stacks)
        if n == 1:
            return 1.0 if (player == 0 and position == 0) else 0.0
        if position >= n or stacks[player] <= 0:
            return 0.0
        tot = sum(stacks)
        if position == n - 1:
            return stacks[player] / tot
        # Probability that someone else busts first
        prob = 0.0
        for i in range(n):
            if i != player and stacks[i] > 0:
                p_elim = stacks[i] / tot
                ns = list(stacks)
                ns[i] = 0.0
                prob += p_elim * self._calculate_icm_cached(tuple(ns), player, position)
        return prob


# ==========================================================
# Preflop Ranges (RFI priors) — extended to 9-max positions
# ==========================================================
class PreflopRangeModel:
    # Baseline open ranges by position label. These are reasonable priors; adjust via opponent modeling.
    RFI: Dict[str, str] = {
        # Core positions
        "UTG":  "77+,AJo+,KQo,ATs+,KTs+,QTs+,JTs,T9s,98s,A5s-A2s",
        "UTG1": "77+,ATo+,KQo,ATs+,KTs+,QTs+,JTs,T9s,98s,A5s-A2s",
        "UTG2": "66+,ATo+,KQo,ATs+,KTs+,QTs+,JTs,T9s,98s,87s,A5s-A2s",
        "LJ":   "66+,ATo+,KJo+,A9s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,A5s-A2s",
        "HJ":   "55+,A9o+,KJo+,QJo,A8s+,KTs+,QTs+,J9s+,T9s,98s,87s,76s,65s,A5s-A2s",
        "MP":   "66+,ATo+,KQo,A9s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,A5s-A2s",
        "MP1":  "66+,ATo+,KQo,A9s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,A5s-A2s",
        "MP2":  "66+,ATo+,KQo,A9s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,A5s-A2s",
        "CO":   "55+,A8o+,KJo+,QJo,A2s+,K9s+,Q9s+,J9s+,T9s,T8s,98s,87s,76s,65s,54s,A5s-A2s",
        "BTN":  "22+,A2o+,K7o+,Q9o+,J9o+,T9o,A2s+,K2s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s",
        "SB":   "22+,A7o+,KTo+,QTo+,JTo,A2s+,K6s+,Q8s+,J8s+,T8s+,98s,87s,76s,65s,54s",
        "BB":   "22+,A2o+,K7o+,Q9o+,J9o+,T9o,A2s+,K2s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s",
    }

    @classmethod
    def by_position(cls, pos: str) -> str:
        return cls.RFI.get(pos, cls.RFI.get("MP", "66+,ATo+,A9s+,KTs+,QTs+,JTs,T9s,98s"))

    @classmethod
    def by_position_table_size(cls, pos: str, n_players: int) -> str:
        # Slight tightening for earlier positions at 9-max, relaxing at short-handed
        base = cls.by_position(pos)
        if n_players >= 9:
            if pos in ("UTG", "UTG1", "UTG2"):
                return "88+,AQo+,AQs+,KQs,A5s-A2s"
        elif n_players == 8:
            if pos in ("UTG", "UTG1"):
                return "77+,AQo+,AJs+,KQs,A5s-A2s"
        elif n_players == 7:
            if pos in ("UTG", "LJ"):
                return "66+,AJo+,ATs+,KQs,QJs,JTs,T9s,98s,A5s-A2s"
        elif n_players == 6:
            if pos == "UTG":
                return "66+,AJo+,ATs+,KQs,QJs,JTs,T9s,98s,A5s-A2s"
            if pos == "MP":
                return "55+,AJo+,ATs+,KQs,QJs,JTs,T9s,98s,87s,A5s-A2s"
        return base


# ==========================================================
# Decision Engine — integrates GTOSolver & CFR spot solver & multiway EV model
# ==========================================================
class PokerAIDecisionEngine:
    RANK_ORDER = {r: i for i, r in enumerate('23456789TJQKA')}

    def __init__(self):
        self.equity_calc = EquityCalculator()
        self.monte_carlo = MonteCarloSimulator()
        # Full GTO solver
        self.gto_solver = GTOSolver(self.equity_calc, ActionAbstraction(), CardBucketer(HandEvaluator(), 6),
                                    cfg=CFRConfig(iterations=150, variant="dcfr"),
                                    chance_samples_per_street=4, max_bucket_pairs_eval=16)
        # Spot solver (fallback / quick)
        self.spot_solver = CFRSpotSolver(self.equity_calc, ActionAbstraction(), CFRConfig(iterations=200, variant="dcfr"))
        self.icm_calc = ICMCalculator()
        self.hand_eval = HandEvaluator()
        self.range_parser = RangeParser()
        self.action_abs = ActionAbstraction()
        self._rng = random.Random(0xA17E5)

    # ---------- Threshold: legacy print (kept for UI) ----------
    def printed_pot_odds(self, to_call: int, pot_size: int, hero_eq: float, num_players: int) -> float:
        he = max(0.0, min(1.0, hero_eq))
        v = 1.0 - he
        return max(0.0, min(1.0, v / max(1, int(num_players))))

    def compare_pot_odds(self, to_call: int, pot_size: int, hero_eq: float, num_players: int) -> float:
        he = max(0.0, min(1.0, hero_eq))
        v = 1.0 - he
        return max(0.0, min(1.0, v / max(1, int(num_players))))

    # ---------- Helpers ----------
    @staticmethod
    def _bubble_pressure_factor(players_remaining: int, places_paid: int, hero_stack_bb: float) -> float:
        base = 1.0
        if players_remaining and places_paid:
            ratio = players_remaining / max(1, places_paid)
            if ratio <= 1.05:
                base *= 0.90
            elif ratio <= 1.20:
                base *= 0.95
        if hero_stack_bb < 12:
            base *= 0.95
        return max(0.85, min(1.0, base))

    def _opponent_profiles(self, stats: Dict[str, Dict[str, int]], actives: List[str], hero_seat: Optional[str]) -> Dict[str, str]:
        prof = {}
        for seat in actives:
            if seat == hero_seat:
                continue
            s = stats.get(seat, {})
            vpip = s.get('vpip', 0)
            pfr = s.get('preflop_raises', 0)
            hands = max(1, s.get('hands_played', 1))
            v = 100.0 * vpip / hands
            p = 100.0 * pfr / hands
            if v < 18 and p < 10:
                prof[seat] = 'nit'
            elif v > 28 and p < 12:
                prof[seat] = 'station'
            elif v > 28 and p > 20:
                prof[seat] = 'lag'
            else:
                prof[seat] = 'tag'
        return prof

    def _eq_called_mult(self, profiles: List[str]) -> float:
        if not profiles:
            return 1.0
        mult = 1.0
        for p in profiles:
            mult *= {'nit': 0.985, 'tag': 0.995, 'lag': 1.005, 'station': 1.015}.get(p, 1.0)
        return max(0.97, min(1.03, mult))

    def _board_texture(self, board: List[str]) -> float:
        if len(board) < 3:
            return 0.0
        suits = [c[1] for c in board]
        ranks = [self.RANK_ORDER[c[0]] for c in board]
        max_suit = max([suits.count(s) for s in 'shdc']) if suits else 0
        span = (max(ranks) - min(ranks)) if ranks else 0
        wet = 0.0
        if max_suit >= 2:
            wet += 0.25
        if max_suit >= 3:
            wet += 0.40
        if span <= 4:
            wet += 0.25
        if any([c[0] for c in board].count(x) >= 2 for x in set([c[0] for c in board])):
            wet += 0.15
        return max(0.0, min(1.0, wet))

    def _draw_profile(self, hero_cards: List[str], board: List[str]) -> Dict[str, bool]:
        cards = hero_cards + board
        if not cards:
            return {"flush_draw": False, "straight_draw": False, "nutted": False}
        suit_counts = {s: 0 for s in 'shdc'}
        for c in cards:
            suit_counts[c[1]] += 1
        flush_draw = any(v >= 4 for v in suit_counts.values()) and not any(v >= 5 for v in suit_counts.values())

        ranks = set(self.RANK_ORDER[c[0]] for c in cards)

        def window4(start):
            window = {(start + i) % 13 for i in range(5)}
            k = len(window & ranks)
            return (k >= 4) and (k < 5)

        straight_draw = any(window4(s) for s in range(9)) or window4(12)

        nutted = False
        if len(board) >= 3:
            try:
                val, cat = self.hand_eval.evaluate_hand(hero_cards + board)
                nutted = (val >> 24) >= 6 or (cat in ("Straight", "Flush") and len(board) >= 4)
            except Exception:
                pass
        return {"flush_draw": flush_draw, "straight_draw": straight_draw, "nutted": nutted}

    # ---------- Villain ranges ----------
    def _default_pos_range(self, pos: str, n_players: int) -> str:
        return PreflopRangeModel.by_position_table_size(pos, n_players)

    def _adapt_villain_ranges(self, positions: Dict[str, str], actives: List[str], hero_seat: str,
                              stats: Dict[str, Dict[str, int]], n_players: int) -> List[str]:
        out = []
        for seat, pos in positions.items():
            if seat == hero_seat or seat not in actives:
                continue
            base = self._default_pos_range(pos, n_players)
            s = stats.get(seat, {})
            hands = max(1, s.get('hands_played', 1))
            vpip = 100.0 * s.get('vpip', 0) / hands
            pfr = 100.0 * s.get('preflop_raises', 0) / hands
            if vpip > 30 and pfr < 12:
                base += ",A2o-A5o,86s,75s,64s,53s"
            elif pfr > 20:
                base += ",KTs,QTs,J9s,T9s,98s,87s,76s,A5o"
            out.append(base)
        if not out:
            out = [self._default_pos_range('MP', n_players)]
        return out

    # ---------- Equity estimation ----------
    def _preflop_quick_strength(self, hero_cards: List[str]) -> float:
        if len(hero_cards) < 2:
            return 0.5
        r = '23456789TJQKA'
        a, b = hero_cards[0][0], hero_cards[1][0]
        s = (hero_cards[0][1] == hero_cards[1][1])
        ia, ib = r.index(a), r.index(b)
        hi = max(ia, ib)
        pair = (ia == ib)
        base = 0.50
        if pair:
            base = [0.52, 0.55, 0.57, 0.59, 0.61, 0.63, 0.65, 0.68, 0.71, 0.75, 0.80, 0.85, 0.85][ia]
        else:
            base += (hi >= r.index('T')) * 0.04
            base += s * 0.02
            base += (abs(ia - ib) == 1) * 0.02
            base += (a == 'A' or b == 'A') * 0.02
        return max(0.35, min(0.85, base))

    def _estimate_equity(self, hero_cards: List[str], board: List[str], num_players: int,
                         hero_position: str, villain_ranges: List[str]) -> float:
        try:
            if not hero_cards or len(hero_cards) < 2:
                return 0.5
            hero = hero_cards[0] + hero_cards[1]
            bstr = "".join(board)
            if len(board) == 5:
                if villain_ranges:
                    eq = self.monte_carlo.simulate_multiway_range_equity(hero, villain_ranges, bstr, n_iterations=400)  # Reduced
                else:
                    eq = 0.5
            else:
                if num_players > 2 and villain_ranges:
                    eq = self.monte_carlo.simulate_multiway_range_equity(hero, villain_ranges, bstr, n_iterations=300)  # Reduced
                else:
                    solo = villain_ranges[0] if villain_ranges else ""
                    eq = self.equity_calc.calculate_vs_range(hero, solo, bstr)
        except Exception:
            eq = self._preflop_quick_strength(hero_cards)

        # Position & multiway penalty
        pos_factor = {"BTN": 1.04, "CO": 1.02, "HJ": 1.01, "MP": 1.0, "LJ": 0.99, "UTG2": 0.985, "UTG1": 0.98, "UTG": 0.975, "SB": 0.95, "BB": 1.0}.get(hero_position, 1.0)
        eq = max(0.05, min(0.95, eq * pos_factor))
        if num_players > 2:
            # Mild equity realization discount as more players remain
            eq *= (1.0 - min(0.20, 0.03 * (num_players - 2)))
        return eq

    # ---------- Main ----------
    def get_optimal_action(self, game_state: Dict[str, Any]) -> Dict[str, Any]:
        hero_info = game_state['hero']
        hand_info = game_state['hand']
        table_info = game_state['table']
        action_info = game_state['action']
        tinfo = game_state.get('tournament', {})

        hero_cards = hand_info['hero_cards'] or []
        board_cards = hand_info['community_cards'] or []
        stage = hand_info['stage']
        hero_stack = max(0, hero_info['stack'])
        positions = table_info['positions']
        hero_position = positions.get(str(hand_info['hero_seat']), "MP")
        num_players = max(2, len([s for s in table_info['active_players'] if s not in table_info.get('folded_players', [])]))
        big_blind = max(1, hand_info.get('big_blind_amount', 1))

        to_call = max(0, hero_info.get('to_call', max(0, int(hand_info['current_bet']) - int(hero_info.get('current_bet', 0)))))
        pot_size = max(0, hand_info['pot_amount'])
        current_bet_high = int(hand_info.get('current_bet', 0))

        profiles_map = self._opponent_profiles(table_info.get('player_actions', {}), table_info.get('active_players', []), str(hand_info['hero_seat']))
        profiles_to_act = [profiles_map.get(seat, 'tag') for seat in table_info.get('active_players', []) if seat != str(hand_info['hero_seat'])]
        villain_profile_hint = profiles_to_act[0] if profiles_to_act else "tag"

        villain_ranges = self._adapt_villain_ranges(positions, table_info.get('active_players', []), str(hand_info['hero_seat']), table_info.get('player_actions', {}), num_players)

        equity = self._estimate_equity(hero_cards, board_cards, num_players, hero_position, villain_ranges)

        threshold_print = self.printed_pot_odds(to_call, pot_size, equity, num_players)
        threshold_cmp = self.compare_pot_odds(to_call, pot_size, equity, num_players)

        eq_called_mult = self._eq_called_mult(profiles_to_act)
        bubble_factor = self._bubble_pressure_factor(tinfo.get('players_remaining', 0), tinfo.get('places_paid', 0), hero_stack / max(1, big_blind))

        avail = [a.lower() for a in action_info['available_actions']]
        min_raise = int(hand_info.get('min_raise', 0))
        max_raise = int(hand_info.get('max_raise', hero_stack))
        min_bet = int(hand_info.get('min_bet', max(1, min_raise)))
        max_bet = int(hand_info.get('max_bet', max_raise))

        # Determine villain effective stack (pick largest among not-hero as rough eff)
        villain_stack = 0
        for seat, stack in table_info.get('player_stacks', {}).items():
            if seat != str(hand_info['hero_seat']):
                villain_stack = max(villain_stack, stack)

        # --- Primary path: GTOSolver for heads-up postflop
        cfr_result = None
        active_not_folded = [s for s in table_info.get('active_players', []) if s not in table_info.get('folded_players', [])]
        if len(active_not_folded) == 2 and len(hero_cards) >= 2 and stage != 'preflop':
            hero = hero_cards[0] + hero_cards[1]
            opp_seat = next(s for s in active_not_folded if s != str(hand_info['hero_seat']))
            opp_pos = positions.get(opp_seat, "MP")
            opp_range = PreflopRangeModel.by_position_table_size(opp_pos, num_players)
            try:
                cfr_result = self.gto_solver.solve(
                    hero_hand=hero,
                    villain_range=opp_range,
                    board=board_cards[:],  # pass a copy
                    pot=pot_size,
                    to_call=to_call,
                    hero_stack=hero_stack,
                    villain_stack=villain_stack,
                    min_bet=min_bet,
                    max_bet=max_bet,
                    min_raise_to=min_raise,
                    max_raise_to=max_raise,
                    current_bet_high=current_bet_high,
                    stage=stage,
                    villain_profile_hint=villain_profile_hint
                )
            except Exception:
                cfr_result = None

        # Fallback: spot solver (single street)
        if not cfr_result and len(active_not_folded) == 2 and len(hero_cards) >= 2 and stage != 'preflop':
            hero = hero_cards[0] + hero_cards[1]
            opp_seat = next(s for s in active_not_folded if s != str(hand_info['hero_seat']))
            opp_pos = positions.get(opp_seat, "MP")
            opp_range = PreflopRangeModel.by_position_table_size(opp_pos, num_players)
            try:
                spot = self.spot_solver.solve(
                    hero_hand=hero, villain_range=opp_range,
                    board=board_cards, pot=pot_size, to_call=to_call,
                    hero_stack=hero_stack, villain_stack=villain_stack,
                    min_bet=min_bet, max_bet=max_bet,
                    min_raise_to=min_raise, max_raise_to=max_raise,
                    current_bet_high=current_bet_high,
                    stage=stage, villain_profile_hint=villain_profile_hint
                )
                # Convert to same shape
                cfr_result = {"actions": spot["actions"], "probs": spot["probs"], "bucket": 0, "strategy_by_bucket": {0: spot["strategy"]}}
            except Exception:
                cfr_result = None

        # EV scoring for non-solved modes (and for reporting)
        action_ev_raw: Dict[str, float] = {}
        action_ev: Dict[str, float] = {}
        breakdown: Dict[str, Any] = {}

        if 'fold' in avail:
            action_ev['fold'] = 0.0
            action_ev_raw['fold'] = 0.0
        if 'check' in avail:
            # Realization factor: IP > OOP; multiway discount
            is_ip = hero_position in ("BTN", "CO", "HJ")
            R = (0.96 if is_ip else 0.92) * (1.0 - 0.02 * max(0, num_players - 2))
            ev = (equity * R) * pot_size
            action_ev['check'] = ev / max(1.0, big_blind)
            action_ev_raw['check'] = ev
        if 'call' in avail:
            is_ip = hero_position in ("BTN", "CO", "HJ")
            R = (0.96 if is_ip else 0.92) * (1.0 - 0.02 * max(0, num_players - 2))
            ev_call = (equity * R) * (pot_size + to_call) - to_call if to_call > 0 else equity * R * pot_size
            ev_call *= eq_called_mult * bubble_factor
            action_ev['call'] = ev_call / max(1.0, big_blind)
            action_ev_raw['call'] = ev_call

        # Extract GTOSolver suggestion if available
        best_action = None
        amount = 0

        if cfr_result:
            actions = cfr_result["actions"]
            probs = cfr_result["probs"]
            idx = int(np.argmax(probs))
            pick = actions[idx]
            if pick.startswith("bet:"):
                best_action = 'bet'
                amount = int(pick.split(":")[1])
            elif pick.startswith("raise:"):
                best_action = 'raise'
                amount = int(pick.split(":")[1])
            else:
                best_action = pick  # 'check', 'call', or 'fold'
            breakdown['gto'] = {"actions": actions, "probs": probs, "bucket": cfr_result.get("bucket", 0), "picked": pick}

        # If GTOSolver not available (multiway or preflop), pick best EV among allowed,
        # and propose a practical size using action abstraction grids.
        if not best_action:
            if 'bet' in avail and stage != 'preflop':
                # try abstraction bet sizes and pick max EV (called-branch only, conservative)
                eff = hero_stack
                size_grid = self.action_abs.bet_sizes(pot_size, min_bet, max_bet, eff)
                best_b, best_b_ev = 0, float("-inf")
                for B in size_grid:
                    # called pot (no FE) multiway conservative
                    ev_raw = ((equity * (pot_size + 2 * B)) - B) * eq_called_mult * bubble_factor
                    ev_bb = ev_raw / max(1.0, big_blind)
                    if ev_bb > best_b_ev and equity >= threshold_cmp - 1e-9:
                        best_b_ev, best_b = ev_bb, B
                if best_b > 0:
                    best_action = 'bet'
                    amount = best_b

            if not best_action and 'raise' in avail:
                eff = hero_stack
                to_c = max(1, to_call) if to_call > 0 else 1
                size_grid = self.action_abs.raise_to_sizes(to_c, pot_size, min_raise, max_raise, eff, current_bet_high)
                best_r, best_r_ev = 0, float("-inf")
                for R in size_grid:
                    hero_add = max(0, R - 0)  # total we put in on this street as raise-to
                    vill_face = max(0, R - current_bet_high)
                    ev_raw = ((equity * (pot_size + hero_add + vill_face)) - hero_add) * eq_called_mult * bubble_factor
                    ev_bb = ev_raw / max(1.0, big_blind)
                    if ev_bb > best_r_ev and equity >= threshold_cmp - 1e-9:
                        best_r_ev, best_r = ev_bb, R
                if best_r > 0:
                    best_action = 'raise'
                    amount = best_r

            # Otherwise default to call/check/fold by EV
            if not best_action:
                candidates = [(k, v) for k, v in action_ev.items()]
                if candidates:
                    best_action = max(candidates, key=lambda kv: kv[1])[0]
                    if best_action == 'call':
                        amount = to_call
                    elif best_action == 'check':
                        amount = 0
                    elif best_action == 'fold':
                        amount = 0

        # Safety: ensure amount within bounds
        if best_action == 'bet':
            amount = int(max(min_bet, min(amount, max_bet, hero_stack)))
        elif best_action == 'raise':
            amount = int(max(min_raise, min(amount, max_raise, hero_stack)))
        elif best_action == 'call':
            amount = int(max(0, min(to_call, hero_stack)))
        elif best_action == 'check':
            amount = 0
        elif best_action == 'fold':
            amount = 0
        else:
            # last resort
            best_action = 'check' if 'check' in avail else ('call' if 'call' in avail else 'fold')
            amount = 0 if best_action != 'call' else int(max(0, min(to_call, hero_stack)))

        return {
            "action": best_action,
            "amount": amount,
            "reasoning": (
                "Heads-up streets solved via DCFR with bucketed range-vs-range and chance-sampled lookahead. "
                "Multiway fallback uses equity realization model and opponent-profile multipliers."
            ),
            "equity": equity,
            "pot_odds": threshold_print,
            "pot_odds_compare": threshold_cmp,
            "ev_raw": {k: v for k, v in action_ev_raw.items()},
            "breakdown": breakdown if breakdown else {}
        }


# ==========================================================
# Async worker
# ==========================================================
class AsyncDecisionWorker:
    def __init__(self):
        self.exec = ThreadPoolExecutor(max_workers=1, thread_name_prefix="poker-ai")
        self._pending = None
        self._last_key = None
        self._lock = threading.Lock()
        self.engine = PokerAIDecisionEngine()

    def submit_if_needed(self, key, snapshot, print_fn):
        with self._lock:
            if self._pending and not self._pending.done():
                return
            if key == self._last_key:
                return
            self._last_key = key
            self._pending = self.exec.submit(self._run, snapshot, print_fn)

    def _run(self, snapshot, print_fn):
        try:
            decision = self.engine.get_optimal_action(snapshot)
            print_fn(snapshot, decision)
        except Exception:
            # Suppress all non-equity/game-state prints
            pass


# ==========================================================
# Table position utilities (2-max to 9-max)
# ==========================================================
def position_labels_for_n(n: int) -> List[str]:
    """
    Returns position labels in order of distance from BTN (0=BTN, then clockwise).
    Supports 2 to 9 players.
    """
    n = max(2, min(9, int(n)))
    if n == 2:
        return ["BTN", "BB"]
    if n == 3:
        return ["BTN", "SB", "BB"]
    if n == 4:
        return ["BTN", "SB", "BB", "UTG"]
    if n == 5:
        return ["BTN", "SB", "BB", "UTG", "CO"]
    if n == 6:
        return ["BTN", "SB", "BB", "UTG", "MP", "CO"]
    if n == 7:
        return ["BTN", "SB", "BB", "UTG", "LJ", "HJ", "CO"]
    if n == 8:
        return ["BTN", "SB", "BB", "UTG", "UTG1", "LJ", "HJ", "CO"]
    # 9-max
    return ["BTN", "SB", "BB", "UTG", "UTG1", "UTG2", "LJ", "HJ", "CO"]


# ==========================================================
# Tournament tracker & mitmproxy hooks
# ==========================================================
class PokerTournamentTracker:
    def __init__(self):
        self.tournament_id = None
        self.tournament_name = ""
        self.buy_in = 0.0
        self.fee = 0.0
        self.knockout_fee = 0.0
        self.max_entries = 0
        self.prize_pool = 0.0
        self.gtd_amount = 0.0
        self.places_paid = 0
        self.payout_structure = {}
        self.total_contrib = defaultdict(int)
        self.street_bets = defaultdict(int)
        self.call_amount = 0
        self._debounce_sec = 0.20
        self._last_key = None
        self._last_compute_ts = 0.0
        self.worker = AsyncDecisionWorker()

        self._saw_pots_change_in_frame = False
        self._local_pot_delta = 0

        self.table_id = None
        # Default to 9-max support
        self.max_seats = 9
        self.table_name = ""
        self.table_type = ""
        self.seats = {}

        self.hand_number = None
        self.hand_history_id = None
        self.dealer_seat = None
        self.small_blind_seat = None
        self.big_blind_seat = None
        self.hero_seat = None
        self.hero_uuid = None
        self.hero_nickname = ""
        self.hero_cards: List[str] = []
        self.community_cards: List[str] = []
        self.visible_cards: Dict[str, List[str]] = {}

        self.current_stage = "preflop"
        self.pot_amount = 0
        self.side_pots = []
        self.current_bet = 0
        self.min_raise = 0
        self.max_raise = 0
        self.min_bet = 0
        self.max_bet = 0

        self.player_stacks: Dict[str, int] = {}
        self.player_bets: Dict[str, int] = {}
        self.player_actions = defaultdict(lambda: {'vpip': 0, 'pfr': 0, 'aggression_factor': 0, 'hands_played': 0,
                                                   'preflop_raises': 0, 'calls': 0, 'folds': 0, 'three_bets': 0})
        self.active_players: Set[str] = set()
        self.folded_players: Set[str] = set()
        self.all_in_players: Set[str] = set()

        self.current_level = 0
        self.small_blind_amount = 0
        self.big_blind_amount = 0
        self.ante_amount = 0
        self.next_level_info = {}
        self.players_remaining = 0
        self.total_players = 0
        self.average_stack = 0
        self.largest_stack = 0
        self.lowest_stack = 0

        self.current_actor = None
        self.available_actions: List[str] = []
        self.last_action = None
        self.action_time_remaining = 0
        self.time_bank = 0

        self.hand_history = []
        self.ai_engine = PokerAIDecisionEngine()
        self.last_decision = None

    def reset_hand(self):
        self.hand_number = None
        self.hand_history_id = None
        self.dealer_seat = None
        self.hero_cards = []
        self.community_cards = []
        self.visible_cards.clear()
        self.current_stage = "preflop"
        self.pot_amount = 0
        self.side_pots = []
        self.current_bet = 0
        self.min_raise = 0
        self.max_raise = 0
        self.min_bet = 0
        self.max_bet = 0
        self.player_bets.clear()
        self.active_players.clear()
        self.folded_players.clear()
        self.all_in_players.clear()
        self.current_actor = None
        self.available_actions = []
        self.last_action = None
        self.total_contrib.clear()
        self.street_bets.clear()
        self.call_amount = 0

    def _pot_snapshot(self) -> int:
        base = int(self.pot_amount or 0)
        street = sum(int(v) for v in self.player_bets.values())
        return max(0, base + street)

    def maybe_compute_and_print(self):
        if self.current_actor != self.hero_seat:
            return
        key = (
            self.hand_number,
            self.current_stage,
            self.current_actor,
            tuple(self.available_actions),
            int(self.current_bet),
            int(self.pot_amount),
            tuple(self.community_cards),
        )
        now = time.monotonic()
        if key == self._last_key and (now - self._last_compute_ts) < self._debounce_sec:
            return
        self._last_key = key
        self._last_compute_ts = now
        self.get_ai_decision()
        self.print_ai_recommendation()

    def print_ai_recommendation_from_snapshot(self, state: Dict[str, Any], decision: Dict[str, Any]):
        # Only print game state + equity information
        pos = state['table']['positions'].get(str(state['hand']['hero_seat']), 'Unknown')
        print(f"Hand: #{state['hand']['number']} | Stage: {state['hand']['stage'].upper()}")
        print(f"Hero: {state['hand']['hero_cards']} | Board: {state['hand']['community_cards']}")
        print(f"Pot: {state['hand']['pot_amount']} | To call: {state['hero']['to_call']}")
        print(f"Stack: {state['hero']['stack']} | Position: {pos}")
        print(f"Equity: {decision['equity']:.1%} | Threshold: {decision['pot_odds']:.1%} (compare {decision.get('pot_odds_compare', 0):.1%})")

    def _decision_key(self) -> Tuple[Any, ...]:
        return (
            self.hand_number,
            self.current_stage,
            self.current_actor,
            tuple(self.available_actions),
            int(self.current_bet),
            int(self.pot_amount),
            tuple(self.community_cards),
            self.hero_seat,
            int(self.player_bets.get(self.hero_seat, 0)) if self.hero_seat else 0,
        )

    def process_message(self, message: str):
        try:
            if message.strip().startswith('<'):
                root = ET.fromstring(message)
                self._process_xml_element(root)
        except ET.ParseError:
            # Suppress non-essential prints
            pass
        except Exception:
            # Suppress non-essential prints
            pass

    def _process_xml_element(self, element):
        tag = element.tag
        if tag == 'EnterTable':
            self._handle_enter_table(element)
        elif tag == 'TableDetails':
            self._handle_table_details(element)
        elif tag == 'Message':
            self._handle_message(element)
        elif tag == 'PlayerInfo':
            self._handle_player_info(element)
        elif tag == 'TargetBalance':
            self._handle_balance(element)
        elif tag in ('Raise', 'Fold'):
            self._handle_player_action_response(element)
        elif tag == 'Ping':
            pass

    def _handle_enter_table(self, element):
        self.table_id = element.get('tableId')
        self.tournament_id = element.get('tournamentId')

    def _handle_table_details(self, element):
        table_type = element.get('type')
        if table_type == 'TOURNAMENT_TABLE':
            self._handle_tournament_table(element)
        elif table_type == 'SCHEDULED_TOURNAMENT':
            self._handle_tournament_info(element)

    def _handle_tournament_table(self, element):
        tournament_table = element.find('.//TournamentTable')
        if tournament_table is not None:
            self.buy_in = float(tournament_table.get('buyIn', 0) or 0)
            self.fee = float(tournament_table.get('fee', 0) or 0)
            self.knockout_fee = float(tournament_table.get('knockoutFee', 0) or 0)
            self.max_entries = int(tournament_table.get('maxEntries', 0) or 0)
            self.tournament_name = tournament_table.get('tournamentName', '')
        params = element.find('.//Parameters')
        if params is not None:
            self.ante_amount = int(params.get('ante', 0) or 0)
            self.small_blind_amount = int(params.get('lowStake', 0) or 0)
            self.big_blind_amount = int(params.get('highStake', 0) or 0)
        seats_elem = element.find('.//Seats')
        if seats_elem is not None:
            self._parse_seats(seats_elem)
            self.max_seats = max(2, min(9, len(seats_elem.findall('.//Seat'))))

    def _handle_tournament_info(self, element):
        tournament = element.find('.//ScheduledTournament')
        if tournament is not None:
            self.tournament_name = tournament.get('name', '')
            params = tournament.find('.//Parameters')
            if params is not None:
                self.buy_in = float(params.get('buyIn', 0) or 0)
                self.fee = float(params.get('fee', 0) or 0)
                self.knockout_fee = float(params.get('knockoutFee', 0) or 0)
                self.max_entries = int(params.get('maxEntries', 0) or 0)
            prize_info = tournament.find('.//PrizeInfo')
            if prize_info is not None:
                self.prize_pool = float(prize_info.get('prizePool', 0) or 0)
                self.gtd_amount = float(prize_info.get('gtd', 0) or 0)
                self.places_paid = int(prize_info.get('placesPaid', 0) or 0)
                self.payout_structure.clear()
                for payment in prize_info.findall('.//Payment'):
                    places = payment.get('place')
                    amount = float(payment.get('amount', 0) or 0)
                    if places:
                        self.payout_structure[places] = amount
            participants = tournament.find('.//Participants')
            if participants is not None:
                self.players_remaining = int(participants.get('remaining', 0) or 0)
                self.total_players = int(participants.get('total', 0) or 0)

    def _handle_message(self, element):
        changes = element.find('.//Changes')
        if changes is not None:
            self._process_changes(changes)
        game_state = element.find('.//GameState')
        if game_state is not None:
            self._process_game_state(game_state)

    def _process_game_state(self, game_state):
        self.hand_number = game_state.get('hand') or self.hand_number
        self.hand_history_id = game_state.get('historyId') or self.hand_history_id
        tourney_info = game_state.find('.//TournamentInfo')
        if tourney_info is not None:
            self._update_tournament_state(tourney_info)
        hand_info = game_state.find('.//HandInfo')
        if hand_info is not None:
            self.ante_amount = int(hand_info.get('ante', 0) or 0)
            self.small_blind_amount = int(hand_info.get('lowStake', 0) or 0)
            self.big_blind_amount = int(hand_info.get('highStake', 0) or 0)
        seats_elem = game_state.find('.//Seats')
        if seats_elem is not None:
            self.dealer_seat = seats_elem.get('dealer') or self.dealer_seat
            self.hero_seat = seats_elem.get('me') or self.hero_seat
            self._parse_seats(seats_elem)
            for seat_elem in seats_elem.findall('.//Seat'):
                sid = seat_elem.get('id')
                if sid:
                    self.active_players.add(sid)
                    if seat_elem.get('status') == 'all-in':
                        self.all_in_players.add(sid)
        board = game_state.find('.//Board')
        if board is not None:
            self.community_cards = []
            for card_elem in board.findall('.//Card'):
                c = card_elem.text
                if c and c != 'xx':
                    self.community_cards.append(c)
            if len(self.community_cards) >= 5:
                self.current_stage = "river"
            elif len(self.community_cards) >= 4:
                self.current_stage = "turn"
            elif len(self.community_cards) >= 3:
                self.current_stage = "flop"
            else:
                self.current_stage = "preflop"

    def _process_changes(self, changes):
        self._saw_pots_change_in_frame = False
        self._local_pot_delta = 0

        new_hand = changes.find('.//NewHand')
        if new_hand is not None:
            self.reset_hand()
            self.hand_number = new_hand.get('number') or self.hand_number
            self.dealer_seat = new_hand.get('dealer') or self.dealer_seat

        active_seats = changes.find('.//ActiveSeats')
        if active_seats is not None:
            self.active_players.clear()
            for seat_elem in active_seats.findall('.//Seat'):
                sid = seat_elem.get('id')
                if sid:
                    self.active_players.add(sid)

        for action_elem in changes.findall('.//PlayerAction'):
            self._process_player_action(action_elem)

        pots_change = changes.find('.//PotsChange')
        if pots_change is not None:
            self._update_pot(pots_change)

        dealing_cards = changes.find('.//DealingCards')
        if dealing_cards is not None:
            self._process_dealing_cards(dealing_cards)

        # Community cards by street
        dealing_flop = changes.find('.//DealingFlop')
        if dealing_flop is not None:
            self._process_community_cards(dealing_flop, 'flop')
        dealing_turn = changes.find('.//DealingTurn')
        if dealing_turn is not None:
            self._process_community_cards(dealing_turn, 'turn')
        dealing_river = changes.find('.//DealingRiver')
        if dealing_river is not None:
            self._process_community_cards(dealing_river, 'river')

        winners = changes.find('.//Winners')
        if winners is not None:
            self._process_winners(winners)
        end_hand = changes.find('.//EndHand')
        if end_hand is not None:
            self._process_end_hand(end_hand)

        tourney_change = changes.find('.//TournamentInfoChange')
        if tourney_change is not None:
            tinfo = tourney_change.find('.//TournamentInfo')
            if tinfo is not None:
                self._update_tournament_state(tinfo)

        active_change = changes.find('.//ActiveChange')
        if active_change is not None:
            self._process_active_change(active_change)

        if not self._saw_pots_change_in_frame and self._local_pot_delta:
            self.pot_amount += self._local_pot_delta
        self._local_pot_delta = 0

    def _parse_seats(self, seats_elem):
        for seat_elem in seats_elem.findall('.//Seat'):
            sid = seat_elem.get('id')
            if not sid:
                continue

            pinfo = seat_elem.find('.//PlayerInfo')
            if pinfo is not None:
                uuid = pinfo.get('uuid')
                nickname = pinfo.get('nickname', '')
                if uuid and uuid == self.hero_uuid:
                    self.hero_seat = sid
                    self.hero_nickname = nickname
                chips = pinfo.find('.//Chips')
                if chips is not None:
                    stack_size = int(chips.get('stack-size', 0) or 0)
                    self.player_stacks[sid] = stack_size
                    if chips.get('bet') is not None:
                        self.player_bets[sid] = int(chips.get('bet', 0) or 0)

            cards_elem = seat_elem.find('.//Cards')
            if cards_elem is not None:
                cards = []
                for c in cards_elem.findall('.//Card'):
                    t = c.text
                    if t and t != 'xx':
                        cards.append(t)
                if cards:
                    self.visible_cards[sid] = cards
                    if sid == self.hero_seat:
                        self.hero_cards = cards

    def _process_player_action(self, action_elem):
        sid = action_elem.get('seat')
        if sid is None:
            return
        node = action_elem[0] if len(action_elem) else None
        if node is None:
            return

        atype = node.tag
        amt_raw = node.get('amount', 0)
        amount = int(float(amt_raw or 0))

        # Stats
        self.player_actions[sid]['hands_played'] = max(1, self.player_actions[sid]['hands_played'] + (1 if self.current_stage == 'preflop' else 0))
        if self.current_stage == 'preflop':
            if atype in ('Call', 'Bet', 'Raise'):
                self.player_actions[sid]['vpip'] += 1
            if atype == 'Raise':
                self.player_actions[sid]['preflop_raises'] += 1
            if atype == 'Fold':
                self.player_actions[sid]['folds'] += 1

        prev = int(self.player_bets.get(sid, 0))

        def add_or_set_total(new_amount: int, is_total_hint: bool) -> int:
            if is_total_hint:
                return max(prev, new_amount)
            delta_total = prev + new_amount
            return max(delta_total, new_amount)

        if atype == 'Fold':
            self.folded_players.add(sid)
            self.active_players.discard(sid)

        elif atype == 'Check':
            pass

        elif atype == 'PostSmallBlind':
            self.player_bets[sid] = prev + amount
            if sid in self.player_stacks:
                self.player_stacks[sid] = max(0, self.player_stacks[sid] - amount)

        elif atype == 'PostBigBlind':
            self.player_bets[sid] = prev + amount
            self.current_bet = max(self.current_bet, self.player_bets[sid])
            if sid in self.player_stacks:
                self.player_stacks[sid] = max(0, self.player_stacks[sid] - amount)

        elif atype == 'PostAnte':
            self.player_bets[sid] = prev + amount
            if sid in self.player_stacks:
                self.player_stacks[sid] = max(0, self.player_stacks[sid] - amount)

        elif atype == 'Bet':
            total = add_or_set_total(amount, True)
            self.player_bets[sid] = total
            self.current_bet = max(self.current_bet, total)
            if sid in self.player_stacks:
                self.player_stacks[sid] = max(0, self.player_stacks[sid] - max(0, total - prev))

        elif atype == 'Raise':
            total = add_or_set_total(amount, True)
            self.player_bets[sid] = total
            self.current_bet = max(self.current_bet, total)
            if sid in self.player_stacks:
                self.player_stacks[sid] = max(0, self.player_stacks[sid] - max(0, total - prev))

        elif atype == 'Call':
            total = add_or_set_total(amount, False)
            if self.current_bet > 0:
                total = min(total, self.current_bet)
            self.player_bets[sid] = total
            if sid in self.player_stacks:
                self.player_stacks[sid] = max(0, self.player_stacks[sid] - max(0, total - prev))

        self.last_action = (sid, atype, amount)

    def _process_dealing_cards(self, dealing_cards):
        for seat_elem in dealing_cards.findall('.//Seat'):
            sid = seat_elem.get('id')
            cards_elem = seat_elem.find('.//Cards')
            if not sid or cards_elem is None:
                continue
            cards = []
            for c in cards_elem.findall('.//Card'):
                t = c.text
                if t and t != 'xx':
                    cards.append(t)
            if sid == self.hero_seat and cards:
                self.hero_cards = cards

    def _process_community_cards(self, dealing_elem, street: str):
        self.current_stage = street
        cards_elem = dealing_elem.find('.//Cards')
        new_cards = []
        if cards_elem is not None:
            for c in cards_elem.findall('.//Card'):
                t = c.text
                if t:
                    new_cards.append(t)
        if new_cards:
            self.community_cards.extend(new_cards)
        self.current_bet = 0
        self.street_bets.clear()
        for sid in list(self.player_bets.keys()):
            self.player_bets[sid] = 0

    def _process_active_change(self, active_change):
        self.current_actor = active_change.get('seat')
        self.action_time_remaining = int(active_change.get('time', 0) or 0)
        self.time_bank = int(active_change.get('timeBank', 0) or 0)

        self.available_actions = []
        self.min_raise = 0
        self.max_raise = 0
        self.min_bet = 0
        self.max_bet = 0
        self.call_amount = 0

        actions_elem = active_change.find('.//Actions')
        if actions_elem is not None:
            for a in actions_elem:
                tag = a.tag
                self.available_actions.append(tag)
                if tag == 'Raise':
                    self.min_raise = int(a.get('min', 0) or 0)
                    self.max_raise = int(a.get('max', 0) or 0)
                elif tag == 'Bet':
                    self.min_bet = int(a.get('min', 0) or 0)
                    self.max_bet = int(a.get('max', 0) or 0)
                elif tag == 'Call':
                    raw = a.get('amount') or a.get('to') or a.get('total') or '0'
                    try:
                        self.call_amount = int(float(raw))
                    except Exception:
                        self.call_amount = 0

        if self.current_actor == self.hero_seat:
            self.maybe_compute_and_print()

    def _update_pot(self, pots_change):
        total = 0
        saw_absolute = False
        for pot_elem in pots_change.findall('.//Pot'):
            if pot_elem.get('value') is not None:
                total += int(pot_elem.get('value', 0) or 0)
                saw_absolute = True
            elif pot_elem.get('amount') is not None:
                total += int(pot_elem.get('amount', 0) or 0)
                saw_absolute = True
        if saw_absolute:
            self.pot_amount = total
            return
        self._saw_pots_change_in_frame = True
        delta = 0
        for pot_elem in pots_change.findall('.//Pot'):
            change = int(pot_elem.get('change', 0) or 0)
            delta += change
        self.pot_amount += delta
        # Fallback sanity
        self.pot_amount = max(self.pot_amount, sum(self.total_contrib.values()))

    def _process_winners(self, winners):
        # Minimal: could parse payouts per seat if needed
        pass

    def _process_end_hand(self, end_hand):
        self.hand_history.append({
            'hand_number': self.hand_number,
            'hero_cards': list(self.hero_cards),
            'community_cards': list(self.community_cards),
            'pot_amount': self.pot_amount,
            'actions': {k: dict(v) for k, v in self.player_actions.items()},
            'winner_last_action': self.last_action
        })

    def _update_tournament_state(self, tinfo):
        parts = tinfo.find('.//Participants')
        if parts is not None:
            self.players_remaining = int(parts.get('remaining', 0) or 0)
            self.total_players = int(parts.get('total', 0) or 0)
        current_level = tinfo.find('.//CurrentLevel')
        if current_level is not None:
            self.current_level = int(current_level.get('number', 0) or 0)
            self.ante_amount = int(current_level.get('ante', 0) or 0)
            self.small_blind_amount = int(current_level.get('lowStake', 0) or 0)
            self.big_blind_amount = int(current_level.get('highStake', 0) or 0)
        stacks = tinfo.find('.//Stacks')
        if stacks is not None:
            self.average_stack = float(stacks.get('averageStack', 0) or 0)
            self.largest_stack = int(stacks.get('largestStack', 0) or 0)
            self.lowest_stack = int(stacks.get('lowestStack', 0) or 0)

    def _handle_player_info(self, element):
        self.hero_uuid = element.get('uuid') or self.hero_uuid
        self.hero_nickname = element.get('nickname', self.hero_nickname)
        entry = element.find('.//Entry')
        if entry is not None:
            chips = entry.find('.//Chips')
            if chips is not None and self.hero_seat is not None:
                self.player_stacks[self.hero_seat == self.hero_seat] = int(chips.get('stack-size', 0) or 0)

    def _handle_balance(self, element):
        return

    def _handle_player_action_response(self, element):
        return

    def _calculate_positions(self) -> Dict[str, str]:
        """
        Map active seat ids (strings of ints) to positions using dealer seat and table size (2–9).
        """
        if not self.dealer_seat or not self.active_players:
            return {}
        try:
            seats_sorted = sorted(int(s) for s in self.active_players)
            dealer = int(self.dealer_seat)
            if dealer not in seats_sorted:
                seats_sorted.append(dealer)
                seats_sorted = sorted(seats_sorted)
            # rotate so dealer is first (BTN)
            if dealer in seats_sorted:
                idx = seats_sorted.index(dealer)
                order = seats_sorted[idx:] + seats_sorted[:idx]
            else:
                order = seats_sorted
            n = len([s for s in order if str(s) in self.active_players])
            labels = position_labels_for_n(n)
            pos_map: Dict[str, str] = {}
            i_label = 0
            for sid in order:
                sid_str = str(sid)
                if sid_str in self.active_players:
                    pos_map[sid_str] = labels[i_label % len(labels)]
                    i_label += 1
            return pos_map
        except Exception:
            # Fallback simple mapping
            labels = position_labels_for_n(len(self.active_players))
            return {s: labels[i % len(labels)] for i, s in enumerate(sorted(self.active_players))}

    def get_game_state_summary(self) -> Dict[str, Any]:
        payouts: List[float] = []
        if self.prize_pool > 0 and self.places_paid > 0:
            for i in range(1, self.places_paid + 1):
                key = f"{i}"
                if key in self.payout_structure:
                    payouts.append(float(self.payout_structure[key]))

        positions = self._calculate_positions()

        hero_bet = 0
        if self.hero_seat is not None:
            hero_bet = int(self.player_bets.get(self.hero_seat, 0))
        computed_to_call = max(0, int(self.current_bet) - hero_bet)
        to_call = self.call_amount if self.call_amount > 0 else computed_to_call

        pot_effective = self._pot_snapshot()

        return {
            'tournament': {
                'id': self.tournament_id,
                'name': self.tournament_name,
                'players_remaining': self.players_remaining,
                'total_players': self.total_players,
                'current_level': self.current_level,
                'blinds': f"{self.small_blind_amount}/{self.big_blind_amount}",
                'ante': self.ante_amount,
                'average_stack': self.average_stack,
                'payouts': payouts,
                'prize_pool': self.prize_pool,
                'places_paid': self.places_paid
            },
            'hand': {
                'number': self.hand_number,
                'stage': self.current_stage,
                'dealer': self.dealer_seat,
                'hero_seat': self.hero_seat,
                'hero_cards': self.hero_cards,
                'community_cards': self.community_cards,
                'pot_amount': pot_effective,
                'current_bet': self.current_bet,
                'min_raise': self.min_raise,
                'max_raise': self.max_raise,
                'min_bet': self.min_bet,
                'max_bet': self.max_bet,
                'big_blind_amount': self.big_blind_amount
            },
            'hero': {
                'stack': self.player_stacks.get(self.hero_seat, 0) if self.hero_seat is not None else 0,
                'current_bet': hero_bet,
                'to_call': to_call
            },
            'table': {
                'active_players': list(self.active_players),
                'folded_players': list(self.folded_players),
                'all_in_players': list(self.all_in_players),
                'player_stacks': dict(self.player_stacks),
                'player_bets': dict(self.player_bets),
                'player_actions': {k: dict(v) for k, v in self.player_actions.items()},
                'positions': positions,
                'average_stack': self.average_stack
            },
            'action': {
                'current_actor': self.current_actor,
                'available_actions': self.available_actions,
                'time_remaining': self.action_time_remaining,
                'time_bank': self.time_bank
            }
        }

    def get_ai_decision(self) -> Dict[str, Any]:
        state = self.get_game_state_summary()
        decision = self.ai_engine.get_optimal_action(state)
        self.last_decision = decision
        return decision

    def print_ai_recommendation(self):
        if not self.last_decision or self.current_actor != self.hero_seat:
            return
        decision = self.last_decision
        state = self.get_game_state_summary()
        pos = state['table']['positions'].get(str(state['hand']['hero_seat']), 'Unknown')
        # Only print game state + equity information
        print(f"Hand: #{state['hand']['number']} | Stage: {state['hand']['stage'].upper()}")
        print(f"Hero: {state['hand']['hero_cards']} | Board: {state['hand']['community_cards']}")
        print(f"Pot: {state['hand']['pot_amount']} | To call: {state['hero']['to_call']}")
        print(f"Stack: {state['hero']['stack']} | Position: {pos}")
        print(f"Equity: {decision['equity']:.1%} | Threshold: {decision['pot_odds']:.1%} (compare {decision.get('pot_odds_compare', 0):.1%})")


# -------------- Global instance & mitmproxy hooks --------------
tracker = PokerTournamentTracker()


def websocket_message(flow):
    # mitmproxy websocket hook
    ws = getattr(flow, "websocket", None)
    if not ws or not getattr(ws, "messages", None):
        return
    msg = None
    for m in reversed(ws.messages):
        if not m.from_client:
            msg = m
            break
    if msg is None:
        return
    try:
        content = msg.content.decode('utf-8', errors='ignore')
        if not content.lstrip().startswith('<'):
            return
        tracker.process_message(content)
        if tracker.current_actor == tracker.hero_seat:
            key = tracker._decision_key()
            now = time.monotonic()
            if key != tracker._last_key or (now - tracker._last_compute_ts) >= tracker._debounce_sec:
                tracker._last_key = key
                tracker._last_compute_ts = now
                snapshot = copy.deepcopy(tracker.get_game_state_summary())
                tracker.worker.submit_if_needed(key, snapshot, tracker.print_ai_recommendation_from_snapshot)
        if len(ws.messages) > 2000:
            del ws.messages[:-1000]
    except Exception:
        # Suppress non-essential prints
        pass


def request(flow):
    return


def response(flow):
    if flow.response.headers.get("Upgrade", "").lower() == "websocket":
        pass


def websocket_start(flow):
    # Suppress connection prints
    pass


def websocket_end(flow):
    # Suppress connection prints
    pass


if __name__ == "__main__":
    # This script is intended to run as a mitmproxy addon; when run standalone it does nothing.
    print("Poker AI Assistant module loaded. Using eval7:", _EVAL7_AVAILABLE, "| Use within mitmproxy to attach to the poker client websocket.")
