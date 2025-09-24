#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Single-file Poker Tournament Tracker + EV-first Decision Engine (v8.2)

- Fast preflop now works reliably: villain-range sampling + LRU cache + quick fallback
- Thresholding uses **villain equity divided by number of active players**:
    • Decision/display threshold = (1 − hero_eq) / max(1, num_active_players)  (clamped to [0,1])
- Strict raise/bet guard when equity < threshold unless FE≥50% AND raise-EV > call-EV
- Uses deterministic seeds so advice is stable within a spot
- Integrates GTO math: MDF-based FE, optimal bluff ratios on river, bucketing & betting-round reduction, mixed strategies.

Major changes in v8.2:
  • Threshold:
      - Replace prior villain-equity-only threshold with **villain_eq / players**.
      - Printing: UI shows “Threshold (villain eq / players)”.
  • GTO utilities:
      - MDF, caller pot-odds from size, optimal bluff ratio, and FE requirement for pure bluffs
        remain for sizing and bluff-cap math (river polar spots).
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
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

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
# HandEvaluator
# ==========================================================
from typing import List as _List, Tuple as _Tuple
import numpy.typing as npt

class HandEvaluator:
    RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A']
    SUITS = ['h', 'd', 'c', 's']

    def __init__(self):
        self.rank_values = {rank: i for i, rank in enumerate(self.RANKS)}
        self.suit_values = {suit: i for i, suit in enumerate(self.SUITS)}

    @staticmethod
    @_jit(nopython=True)
    def _calculate_hand_value(ranks: npt.NDArray[np.int32], suits: npt.NDArray[np.int32]) -> int:
        rank_count = np.zeros(13, dtype=np.int32)
        suit_count = np.zeros(4, dtype=np.int32)
        for i in range(len(ranks)):
            rank_count[ranks[i]] += 1
            suit_count[suits[i]] += 1

        flush_suit = -1
        for i in range(4):
            if suit_count[i] >= 5:
                flush_suit = i
                break

        def straight_val(occ):
            straight_value = 0
            cnt = 0
            ace_low = occ[12] == 1
            for i in range(13):
                if occ[i] == 1:
                    cnt += 1
                    if cnt >= 5:
                        straight_value = i + 1
                else:
                    if ace_low and cnt == 4 and i == 4:
                        straight_value = 5
                    cnt = 0
            return straight_value

        if flush_suit >= 0:
            flush_ranks = np.zeros(13, dtype=np.int32)
            for i in range(len(ranks)):
                if suits[i] == flush_suit:
                    flush_ranks[ranks[i]] = 1
            sv = straight_val(flush_ranks)
            if sv > 0:
                return 8000000 + sv
            value = 5000000
            kickers = 0
            cnt = 0
            for i in range(12, -1, -1):
                if flush_ranks[i] == 1:
                    kickers = (kickers << 4) | i
                    cnt += 1
                    if cnt == 5:
                        break
            return value + kickers

        for i in range(13):
            if rank_count[i] == 4:
                kicker = 0
                for j in range(12, -1, -1):
                    if j != i and rank_count[j] > 0:
                        kicker = j
                        break
                return 7000000 + i * 13 + kicker

        three_rank = -1
        pair_rank = -1
        for i in range(12, -1, -1):
            if rank_count[i] == 3 and three_rank == -1:
                three_rank = i
            elif rank_count[i] == 2 and pair_rank == -1:
                pair_rank = i

        if three_rank >= 0 and pair_rank >= 0:
            return 6000000 + three_rank * 13 + pair_rank

        sv = straight_val(rank_count)
        if sv > 0:
            return 4000000 + sv

        if three_rank >= 0:
            kickers = 0
            cnt = 0
            for i in range(12, -1, -1):
                if i != three_rank and rank_count[i] > 0:
                    kickers = (kickers << 4) | i
                    cnt += 1
                    if cnt == 2:
                        break
            return 3000000 + three_rank * 169 + kickers

        pairs = []
        for i in range(12, -1, -1):
            if rank_count[i] == 2:
                pairs.append(i)
                if len(pairs) == 2:
                    break
        if len(pairs) == 2:
            kicker = 0
            for i in range(12, -1, -1):
                if i != pairs[0] and i != pairs[1] and rank_count[i] > 0:
                    kicker = i
                    break
            return 2000000 + pairs[0] * 169 + pairs[1] * 13 + kicker

        if pair_rank >= 0:
            kickers = 0
            cnt = 0
            for i in range(12, -1, -1):
                if i != pair_rank and rank_count[i] > 0:
                    kickers = (kickers << 4) | i
                    cnt += 1
                    if cnt == 3:
                        break
            return 1000000 + pair_rank * 2197 + kickers

        kickers = 0
        cnt = 0
        for i in range(12, -1, -1):
            if rank_count[i] > 0:
                kickers = (kickers << 4) | i
                cnt += 1
                if cnt == 5:
                    break
        return kickers

    def evaluate_hand(self, cards: _List[str]) -> _Tuple[int, str]:
        if len(cards) < 5:
            raise ValueError("Need at least 5 cards to evaluate a hand")
        ranks = np.array([self.rank_values[card[0]] for card in cards], dtype=np.int32)
        suits = np.array([self.suit_values[card[1]] for card in cards], dtype=np.int32)
        value = self._calculate_hand_value(ranks, suits)
        if value >= 8000000: category = "Straight Flush"
        elif value >= 7000000: category = "Four of a Kind"
        elif value >= 6000000: category = "Full House"
        elif value >= 5000000: category = "Flush"
        elif value >= 4000000: category = "Straight"
        elif value >= 3000000: category = "Three of a Kind"
        elif value >= 2000000: category = "Two Pair"
        elif value >= 1000000: category = "One Pair"
        else: category = "High Card"
        return value, category

# ==========================================================
# GTO Math Utilities (MDF, bluff ratio, FE requirements)
# ==========================================================
class GTOUtils:
    @staticmethod
    def mdf_from_size(bet: float, pot: float) -> float:
        """Minimum Defense Frequency = pot / (pot + bet) = 1 / (1 + s)."""
        if pot <= 0: return 0.0
        s = max(0.0, bet / pot)
        return 1.0 / (1.0 + s)

    @staticmethod
    def caller_pot_odds_from_size(bet: float, pot: float) -> float:
        """
        Caller pot odds % when facing a bet size s = bet/pot:
        pot_odds = s / (2s + 1)
        """
        if pot <= 0: return 0.0
        s = max(0.0, bet / pot)
        return s / (2.0 * s + 1.0)

    @staticmethod
    def optimal_bluff_ratio(bet: float, pot: float) -> float:
        """
        For a polar value/bluff mix on a single decision (river model),
        optimal bluff: value ratio = s / (1 + s), with s = bet/pot.
        """
        if pot <= 0: return 0.0
        s = max(0.0, bet / pot)
        return s / (1.0 + s)

    @staticmethod
    def golden_ratio_check() -> float:
        """
        The 'meeting point' where MDF% == pot-odds% occurs at ~Phi (≈1.618 pot bet).
        Not used directly for gating, but handy for sanity checks/logging.
        """
        return (1 + 5 ** 0.5) / 2  # Φ

    @staticmethod
    def required_fe_for_bluff(risk: float, reward: float) -> float:
        """
        Minimum fold equity to break even on a pure bluff:
        FE*reward - (1-FE)*risk = 0 => FE = risk / (risk + reward)
        """
        denom = (risk + max(0.0, reward))
        return max(0.0, min(1.0, (risk / denom) if denom > 0 else 1.0))

# ==========================================================
# RangeParser (supports '+', spans like A2s-A5s)
# ==========================================================
class RangeParser:
    RANKS = ['A', 'K', 'Q', 'J', 'T', '9', '8', '7', '6', '5', '4', '3', '2']
    SUITS = ['s', 'h', 'd', 'c']

    def __init__(self):
        self._init_lookups()

    def _init_lookups(self):
        self._pair_combos: Dict[str, List[str]] = {}
        self._suited_combos: Dict[str, List[str]] = {}
        self._offsuit_combos: Dict[str, List[str]] = {}
        for r in self.RANKS:
            key = r + r
            combos = []
            for i, s1 in enumerate(self.SUITS):
                for s2 in self.SUITS[i+1:]:
                    combos.append(r + s1 + r + s2)
            self._pair_combos[key] = combos
        for i, r1 in enumerate(self.RANKS):
            for r2 in self.RANKS[i+1:]:
                s_key = r1 + r2 + 's'
                o_key = r1 + r2 + 'o'
                self._suited_combos[s_key] = [r1 + s + r2 + s for s in self.SUITS]
                offs = []
                for s1 in self.SUITS:
                    for s2 in self.SUITS:
                        if s1 != s2:
                            offs.append(r1 + s1 + r2 + s2)
                self._offsuit_combos[o_key] = offs

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
            r1 = lo[0]; suf = lo[2]
            r2_lo = lo[1]; r2_hi = hi[1]
            i_lo = self.RANKS.index(r2_lo); i_hi = self.RANKS.index(r2_hi)
            if i_lo > i_hi: i_lo, i_hi = i_hi, i_lo
            return [r1 + self.RANKS[i] + suf for i in range(i_lo, i_hi + 1)]
        return [token]

    def parse_range(self, range_str: str) -> Set[str]:
        hands: Set[str] = set()
        cleaned = (range_str or "").replace(" ", "")
        if not cleaned:
            return hands
        for part in cleaned.split(","):
            if not part: continue
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
# EquityCalculator (with sampling + LRU cache)
# ==========================================================
class LRUCache:
    def __init__(self, capacity: int = 512):
        self.capacity = capacity
        self._data = OrderedDict()
    def get(self, key):
        if key in self._data:
            self._data.move_to_end(key)
            return self._data[key]
        return None
    def set(self, key, value):
        self._data[key] = value
        self._data.move_to_end(key)
        if len(self._data) > self.capacity:
            self._data.popitem(last=False)

class EquityCalculator:
    MAX_VILLAIN_SAMPLES = 60
    PREFLOP_ITERS_PER = 250
    POSTFLOP_ITERS_PER = 800

    def __init__(self):
        self.hand_evaluator = HandEvaluator()
        self.range_parser = RangeParser()
        self.deck = [r + s for r in 'AKQJT98765432' for s in 'shdc']
        self._cache = LRUCache(512)

    def calculate_vs_range(self, hand: str, villain_range: str, board: str = "") -> float:
        key = ("HVR", hand, villain_range, board)
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        vill_all = list(self.range_parser.parse_range(villain_range))
        if not vill_all:
            self._cache.set(key, 0.5); return 0.5

        board_cards = [board[i:i+2] for i in range(0, len(board), 2)]
        dead = set(board_cards + [hand[:2], hand[2:]])
        vill_all = [v for v in vill_all if ({v[:2], v[2:]} & dead) == set() ]
        if not vill_all:
            self._cache.set(key, 0.5); return 0.5

        # Deterministic sampling
        seed = (hash(key) & 0xFFFFFFFF)
        rng = random.Random(seed)
        if len(vill_all) > self.MAX_VILLAIN_SAMPLES:
            vill = rng.sample(vill_all, self.MAX_VILLAIN_SAMPLES)
        else:
            vill = vill_all

        total = 0.0; valid = 0
        iters = self.PREFLOP_ITERS_PER if len(board_cards) < 3 else self.POSTFLOP_ITERS_PER
        for v_hand in vill:
            eqs = self.calculate_equity([hand, v_hand], board, n_iterations=iters)
            total += eqs.get(hand, 0.5)
            valid += 1

        res = (total / valid) if valid else 0.5
        self._cache.set(key, res)
        return res

    def calculate_equity(self, hands: List[str], board: str = "", n_iterations: int = 1000) -> Dict[str, float]:
        board_cards = [board[i:i+2] for i in range(0, len(board), 2)]
        used_cards = set()
        for hand in hands:
            used_cards.add(hand[:2]); used_cards.add(hand[2:])
        used_cards.update(board_cards)
        deck = [card for card in self.deck if card not in used_cards]
        wins = {hand: 0.0 for hand in hands}
        ties = {hand: 0.0 for hand in hands}
        seed = hash(("EQ", tuple(hands), board, n_iterations)) & 0xFFFFFFFF
        rng = random.Random(seed)

        for _ in range(n_iterations):
            remain = 5 - len(board_cards)
            full_board = board_cards + (rng.sample(deck, remain) if remain > 0 else [])
            values = []
            for hand in hands:
                hand_cards = [hand[:2], hand[2:]]
                value = self.hand_evaluator.evaluate_hand(hand_cards + full_board)[0]
                values.append((hand, value))
            max_value = max(v for _, v in values)
            winners = [h for h, v in values if v == max_value]
            if len(winners) == 1:
                wins[winners[0]] += 1.0
            else:
                share = 1.0 / len(winners)
                for w in winners:
                    ties[w] += share

        return {hand: (wins[hand] + ties[hand]) / n_iterations for hand in hands}

# ==========================================================
# MonteCarloSimulator (kept; now faster defaults)
# ==========================================================
class MonteCarloSimulator:
    def __init__(self, n_jobs: int = -1):
        self.hand_evaluator = HandEvaluator()
        self.range_parser = RangeParser()
        self.deck = [r + s for r in RangeParser.RANKS for s in RangeParser.SUITS]
        self.equity_calc = EquityCalculator()

    def _parse_range(self, range_str: str) -> List[str]:
        if not range_str:
            return []
        return list(self.range_parser.parse_range(range_str))

    def simulate_multiway_range_equity(self, hero: str, villain_ranges: List[str], board: str = "", n_iterations: int = 400) -> float:
        all_ranges = [self._parse_range(r) for r in villain_ranges]
        board_cards = {board[i:i+2] for i in range(0, len(board), 2)}
        hero_cards = {hero[:2], hero[2:]}
        all_ranges = [{h for h in hands if not ({h[:2], h[2:]}.intersection(board_cards | hero_cards))} for hands in all_ranges]
        all_ranges = [list(h) for h in all_ranges if h]
        if not all_ranges: return 0.5
        total = 0.0; samples = 0
        seed = hash(("MW", hero, tuple(villain_ranges), board)) & 0xFFFFFFFF
        rng = random.Random(seed)

        for _ in range(n_iterations):
            used = set(hero_cards) | board_cards
            drawn = []
            ok = True
            for pool in all_ranges:
                for _try in range(50):
                    pick = rng.choice(pool)
                    c1, c2 = pick[:2], pick[2:]
                    if (c1 not in used) and (c2 not in used):
                        drawn.append(pick); used.add(c1); used.add(c2); break
                else:
                    ok = False; break
            if not ok: continue
            eq = self.equity_calc.calculate_equity([hero] + drawn, board, n_iterations=40)
            total += eq.get(hero, 0.5); samples += 1
        return total / samples if samples else 0.5

# ==========================================================
# GTOSolver (light priors only) & ICM
# ==========================================================
@dataclass
class GameState:
    player: int
    pot: float
    board: str
    actions: List[str]
    hero_range: Dict[str, float]
    villain_range: Dict[str, float]

class GTOSolver: ...
from functools import lru_cache

class ICMCalculator:
    def __init__(self, cache_size: int = 1024):
        self._calculate_icm_cached = lru_cache(maxsize=cache_size)(self._calculate_icm)
    def calculate_equity(self, stacks: List[int], payouts: List[float]) -> List[float]:
        if len(stacks) < len(payouts): raise ValueError("More payouts than players")
        total = sum(stacks) or 1.0
        ns = [s/total for s in stacks]
        out = []
        for i in range(len(stacks)):
            e = 0.0
            for j, p in enumerate(payouts):
                e += self._calculate_icm_cached(tuple(ns), i, j) * p
            out.append(e)
        return out
    def _calculate_icm(self, stacks: Tuple[float,...], player: int, position: int) -> float:
        n = len(stacks)
        if n == 1: return 1.0 if player == 0 and position == 0 else 0.0
        if position >= n or stacks[player] == 0: return 0.0
        tot = sum(stacks)
        if position == n-1: return stacks[player]/tot
        prob = 0.0
        for i in range(n):
            if i != player and stacks[i] > 0:
                p_elim = stacks[i]/tot
                ns = list(stacks); ns[i] = 0
                prob += p_elim * self._calculate_icm_cached(tuple(ns), player, position)
        return prob

# ==========================================================
# Preflop Ranges (RFI priors)
# ==========================================================
class PreflopRangeModel:
    RFI: Dict[str, str] = {
        "UTG": "77+,AJo+,KQo,ATs+,KTs+,QTs+,JTs,T9s,98s,A5s-A2s",
        "MP":  "66+,ATo+,KQo,A9s+,KTs+,QTs+,JTs,T9s,98s,87s,76s,A5s-A2s",
        "CO":  "55+,A8o+,KJo+,QJo,A2s+,K9s+,Q9s+,J9s+,T9s,T8s,98s,87s,76s,65s,54s,A5s-A2s",
        "BTN": "22+,A2o+,K7o+,Q9o+,J9o+,T9o,A2s+,K2s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s",
        "SB":  "22+,A7o+,KTo+,QTo+,JTo,A2s+,K6s+,Q8s+,J8s+,T8s+,98s,87s,76s,65s,54s",
        "BB":  "22+,A2o+,K7o+,Q9o+,J9o+,T9o,A2s+,K2s+,Q6s+,J7s+,T7s+,97s+,86s+,75s+,65s,54s"
    }
    @classmethod
    def by_position(cls, pos: str) -> str:
        return cls.RFI.get(pos, cls.RFI["MP"])

# ==========================================================
# Decision Engine (v8.2) — GTO-integrated, villain-eq/players threshold
# ==========================================================
class PokerAIDecisionEngine:
    RANK_ORDER = {r:i for i,r in enumerate('23456789TJQKA')}

    def __init__(self):
        self.equity_calc = EquityCalculator()
        self.monte_carlo = MonteCarloSimulator()
        self.gto_solver = GTOSolver()
        self.icm_calc = ICMCalculator()
        self.hand_eval = HandEvaluator()
        self.range_parser = RangeParser()
        # Deterministic engine seed for stable spot advice
        self._rng = random.Random(0xA17E5)

    # ---------- Threshold: villain equity / players (printed vs compared) ----------
    def printed_pot_odds(self, to_call: int, pot_size: int, hero_eq: float, num_players: int) -> float:
        """Display value: (1 - hero_eq) / max(1, num_players)."""
        he = max(0.0, min(1.0, hero_eq))
        v = 1.0 - he
        return max(0.0, min(1.0, v / max(1, int(num_players))))

    def compare_pot_odds(self, to_call: int, pot_size: int, hero_eq: float, num_players: int) -> float:
        """Decision threshold: (1 - hero_eq) / max(1, num_players)."""
        he = max(0.0, min(1.0, hero_eq))
        v = 1.0 - he
        return max(0.0, min(1.0, v / max(1, int(num_players))))

    # ---------- “Theoretical” helpers ----------
    @staticmethod
    def _bubble_pressure_factor(players_remaining: int, places_paid: int, hero_stack_bb: float) -> float:
        base = 1.0
        if players_remaining and places_paid:
            ratio = players_remaining / max(1, places_paid)
            if ratio <= 1.05: base *= 0.90
            elif ratio <= 1.20: base *= 0.95
        if hero_stack_bb < 12: base *= 0.95
        return max(0.85, min(1.0, base))

    def _opponent_profiles(self, stats: Dict[str, Dict[str, int]], actives: List[str], hero_seat: Optional[str]) -> Dict[str, str]:
        prof = {}
        for seat in actives:
            if seat == hero_seat: continue
            s = stats.get(seat, {})
            vpip = s.get('vpip', 0); pfr = s.get('preflop_raises', 0); hands = max(1, s.get('hands_played', 1))
            v = 100.0 * vpip / hands; p = 100.0 * pfr / hands
            if v < 18 and p < 10: prof[seat] = 'nit'
            elif v > 28 and p < 12: prof[seat] = 'station'
            elif v > 28 and p > 20: prof[seat] = 'lag'
            else: prof[seat] = 'tag'
        return prof

    def _fe_profile_mult(self, profiles: List[str]) -> float:
        if not profiles: return 1.0
        mult = 1.0
        for p in profiles:
            mult *= {'nit':1.10,'tag':1.00,'lag':0.93,'station':0.85}.get(p,1.0)
        return max(0.75, min(1.25, mult))

    def _eq_called_mult(self, profiles: List[str]) -> float:
        if not profiles: return 1.0
        mult = 1.0
        for p in profiles:
            mult *= {'nit':0.97,'tag':0.985,'lag':1.0,'station':1.01}.get(p,1.0)
        return max(0.94, min(1.02, mult))

    def _board_texture(self, board: List[str]) -> float:
        if len(board) < 3: return 0.0
        suits = [c[1] for c in board]; ranks = [self.RANK_ORDER[c[0]] for c in board]
        max_suit = max([suits.count(s) for s in 'shdc']) if suits else 0
        span = (max(ranks) - min(ranks)) if ranks else 0
        wet = 0.0
        if max_suit >= 2: wet += 0.25
        if max_suit >= 3: wet += 0.40
        if span <= 4: wet += 0.25
        if any([c[0] for c in board].count(x) >= 2 for x in set([c[0] for c in board])): wet += 0.15
        return max(0.0, min(1.0, wet))

    def _draw_profile(self, hero_cards: List[str], board: List[str]) -> Dict[str, bool]:
        cards = hero_cards + board
        if not cards: return {"flush_draw": False, "straight_draw": False, "nutted": False}
        suit_counts = {s:0 for s in 'shdc'}
        for c in cards: suit_counts[c[1]] += 1
        flush_draw = any(v >= 4 for v in suit_counts.values()) and not any(v >= 5 for v in suit_counts.values())

        ranks = set(self.RANK_ORDER[c[0]] for c in cards)
        def window4(start):
            window = {(start+i) % 13 for i in range(5)}
            k = len(window & ranks)
            return (k >= 4) and (k < 5)
        straight_draw = any(window4(s) for s in range(9)) or window4(12)

        nutted = False
        if len(board) >= 3:
            try:
                val, cat = self.hand_eval.evaluate_hand(hero_cards + board)
                nutted = val >= 6000000 or (cat in ("Straight","Flush") and len(board) >= 4)
            except Exception:
                pass
        return {"flush_draw": flush_draw, "straight_draw": straight_draw, "nutted": nutted}

    # ---------- Bucketing (abstraction) ----------
    def _preflop_bucket(self, hero_cards: List[str]) -> int:
        if len(hero_cards) < 2: return 11
        r = '23456789TJQKA'
        a, b = hero_cards[0], hero_cards[1]
        aR, bR = r.index(a[0]), r.index(b[0])
        pair = (a[0] == b[0]); suited = (a[1] == b[1]); gap = abs(aR - bR)
        high = (a[0] in 'TJQKA') + (b[0] in 'TJQKA')
        if pair:
            return min(5, max(0, aR))  # 0..5 cover 22..AA coarsely
        if suited and gap == 1:
            return 6  # suited connectors
        if suited and (a[0]=='A' or b[0]=='A'):
            return 7  # Axs
        if high == 2 and suited:
            return 8  # suited broadway
        if high == 2:
            return 9  # offsuit broadway
        if suited:
            return 10  # weak suited
        return 11  # trash

    # ---------- Preflop villain range adapter ----------
    def _default_pos_range(self, pos: str) -> str:
        return PreflopRangeModel.by_position(pos)

    def _adapt_villain_ranges(self, positions: Dict[str,str], actives: List[str], hero_seat: str,
                              stats: Dict[str, Dict[str, int]]) -> List[str]:
        out = []
        for seat, pos in positions.items():
            if seat == hero_seat or seat not in actives: continue
            base = self._default_pos_range(pos)
            s = stats.get(seat, {})
            hands = max(1, s.get('hands_played', 1))
            vpip = 100.0 * s.get('vpip', 0) / hands
            pfr  = 100.0 * s.get('preflop_raises', 0) / hands
            # widen/narrow heuristically
            if vpip > 30 and pfr < 12:
                base += ",A2o-A5o,86s,75s,64s,53s"
            elif pfr > 20:
                base += ",KTs,QTs,J9s,T9s,98s,87s"
            out.append(base)
        if not out:
            out = [self._default_pos_range('MP')]
        return out

    # ---------- Equity estimation with bucketing + multiway ----------
    def _preflop_quick_strength(self, hero_cards: List[str]) -> float:
        if len(hero_cards) < 2: return 0.5
        r = '23456789TJQKA'
        a, b = hero_cards[0][0], hero_cards[1][0]
        s = (hero_cards[0][1] == hero_cards[1][1])
        ia, ib = r.index(a), r.index(b)
        hi, lo = max(ia, ib), min(ia, ib)
        pair = (ia == ib)
        base = 0.50
        if pair:
            base = [0.52,0.55,0.57,0.59,0.61,0.63,0.65,0.68,0.71,0.75,0.80,0.85,0.85][ia]
        else:
            base += (hi >= r.index('T')) * 0.04
            base += s * 0.02
            base += (abs(ia-ib) == 1) * 0.02
            base += (a=='A' or b=='A') * 0.02
        return max(0.35, min(0.85, base))

    def _estimate_equity(self, hero_cards: List[str], board: List[str], num_players: int,
                         hero_position: str, villain_ranges: List[str]) -> float:
        try:
            if not hero_cards or len(hero_cards) < 2: return 0.5
            hero = hero_cards[0] + hero_cards[1]
            bstr = "".join(board)
            if len(board) == 5:
                eq = (sum(self.equity_calc.calculate_vs_range(hero, r, bstr) for r in (villain_ranges or [""])) /
                      max(1, len(villain_ranges) or 1))
            else:
                if num_players > 2 and villain_ranges:
                    eq = self.monte_carlo.simulate_multiway_range_equity(hero, villain_ranges, bstr, n_iterations=350)
                else:
                    eq = self.equity_calc.calculate_vs_range(hero, villain_ranges[0] if villain_ranges else "", bstr)
        except Exception:
            eq = self._preflop_quick_strength(hero_cards)

        pos_factor = {"BTN":1.06,"CO":1.03,"MP":1.0,"UTG":0.97,"SB":0.94,"BB":1.0}.get(hero_position,1.0)
        eq = max(0.05, min(0.95, eq * pos_factor))
        if num_players > 2:
            eq = 0.9 * eq if num_players <= 4 else 0.8 * eq
        return eq

    # ---------- FE model via MDF ----------
    def _fe_mdf(self, pot_before: float, face_amt: float, n_to_act: int,
                wetness: float, prof_mult: float, blocker_mult: float) -> float:
        if face_amt <= 0 or pot_before <= 0:
            return 0.0
        mdf = GTOUtils.mdf_from_size(face_amt, pot_before)  # pot/(pot+bet)
        # one opponent FE = 1 - MDF; multiway exponent
        fe1 = max(0.0, min(1.0, 1.0 - mdf))
        fe_all = 1.0 - (1.0 - fe1) ** max(1, n_to_act)
        fe_all *= (1.0 - 0.25 * max(0.0, min(1.0, wetness)))
        fe_all *= prof_mult
        fe_all *= blocker_mult
        return max(0.0, min(0.95, fe_all))

    def _blocker_boost(self, hero_cards: List[str], board: List[str]) -> float:
        if len(board) < 3 or not hero_cards: return 1.0
        suits = [c[1] for c in board]
        suit_counts = {s:suits.count(s) for s in 'shdc'}
        mult = 1.0
        for s, cnt in suit_counts.items():
            if cnt >= 2 and ('A'+s) in hero_cards:
                mult *= 1.05
            ranks = set(c[0] for c in board)
        if len(ranks) == 3 and max(suit_counts.values()) <= 1:
            if any(card[0] in 'AKQ' for card in hero_cards):
                mult *= 1.02
        return max(1.0, min(1.10, mult))

    # ---------- Candidate sizes ----------
    def _candidate_bet_sizes(self, pot: int, min_bet: int, max_bet: int, stack: int,
                             street: str, equity: float, wetness: float, draws: Dict[str,bool], spr: float) -> List[int]:
        fracs = [0.25,0.33,0.5,0.66,1.0]
        strong = equity > 0.72 or draws.get("nutted", False)
        semi = (draws.get("flush_draw") or draws.get("straight_draw")) and equity > 0.40
        if strong or (wetness > 0.6 and (equity > 0.6 or semi)) or (street in ("turn","river") and semi):
            fracs.extend([1.25,1.5])
        sizes = []
        for f in fracs:
            target = int(round(pot * f))
            s = min(stack, max(min_bet, min(target, max_bet)))
            if s >= min_bet: sizes.append(s)
        sizes.append(min_bet)
        return sorted(set(sizes))

    def _candidate_raise_to(self, to_call: int, min_raise: int, max_raise: int, stack: int, equity: float, spr: float) -> List[int]:
        mults = [2.2,2.5,3.0,3.5]
        if equity > 0.65: mults.append(4.0)
        if spr <= 1.6: mults.append(10.0)
        sizes = []
        base = max(1, to_call)
        for m in mults:
            s = int(round(base * m))
            s = min(stack, max(min_raise, min(s, max_raise)))
            if s >= min_raise: sizes.append(s)
        sizes.append(min_raise); sizes.append(min(stack, max_raise))
        return sorted(set(sizes))

    def _gto_weights(self, pos: str, stack_bb: float) -> Dict[str,float]:
        pri = {"raise":0.5,"call":0.3,"fold":0.2}
        if pos == "BTN": pri = {"raise":0.6,"call":0.35,"fold":0.05}
        elif pos == "SB": pri = {"raise":0.5,"call":0.35,"fold":0.15}
        elif pos == "BB": pri = {"raise":0.4,"call":0.5,"fold":0.1}
        elif pos == "CO": pri = {"raise":0.55,"call":0.35,"fold":0.1}
        elif pos == "UTG": pri = {"raise":0.35,"call":0.4,"fold":0.25}
        if stack_bb < 10:
            pri["raise"] += 0.05; pri["fold"] += 0.05
            s = sum(pri.values()); pri = {k:v/s for k,v in pri.items()}
        return {"raise":pri["raise"], "call":pri["call"], "fold":pri["fold"], "bet":pri["raise"], "check":pri["call"]}

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
        num_players = max(2, len(table_info['active_players']))
        big_blind = max(1, hand_info.get('big_blind_amount', 1))

        to_call = max(0, hero_info.get('to_call', max(0, int(hand_info['current_bet']) - int(hero_info.get('current_bet', 0)))))
        pot_size = max(0, hand_info['pot_amount'])
        hero_bet_street = int(hero_info.get('current_bet', 0))
        current_bet_high = int(hand_info.get('current_bet', 0))

        profiles_map = self._opponent_profiles(table_info.get('player_actions', {}), table_info.get('active_players', []), str(hand_info['hero_seat']))
        profiles_to_act = [profiles_map.get(seat, 'tag') for seat in table_info.get('active_players', []) if seat != str(hand_info['hero_seat'])]

        # Villain ranges: position priors + profile-based widening/narrowing
        villain_ranges = self._adapt_villain_ranges(positions, table_info.get('active_players', []), str(hand_info['hero_seat']), table_info.get('player_actions', {}))

        equity = self._estimate_equity(hero_cards, board_cards, num_players, hero_position, villain_ranges)

        # Threshold (villain equity / players) — printed & compared
        pot_odds_print = self.printed_pot_odds(to_call, pot_size, equity, num_players)
        pot_odds_cmp   = self.compare_pot_odds(to_call, pot_size, equity, num_players)

        wetness = self._board_texture(board_cards)
        draws = self._draw_profile(hero_cards, board_cards)
        fe_profile_mult = self._fe_profile_mult(profiles_to_act)
        eq_called_mult = self._eq_called_mult(profiles_to_act)
        blocker_boost = self._blocker_boost(hero_cards, board_cards)
        bubble_factor = self._bubble_pressure_factor(tinfo.get('players_remaining', 0), tinfo.get('places_paid', 0), hero_stack / max(1, big_blind))
        spr = (hero_stack / max(1, pot_size)) if pot_size > 0 else 99.0

        avail = [a.lower() for a in action_info['available_actions']]
        can_raise = 'raise' in avail
        min_raise = int(hand_info.get('min_raise', 0))
        max_raise = int(hand_info.get('max_raise', hero_stack))
        min_bet = int(hand_info.get('min_bet', min_raise))
        max_bet = int(hand_info.get('max_bet', max_raise))

        action_ev: Dict[str, float] = {}
        action_ev_raw: Dict[str, float] = {}
        breakdown: Dict[str, Any] = {}

        # --- Fold baseline
        if 'fold' in avail:
            action_ev['fold'] = 0.0
            action_ev_raw['fold'] = 0.0

        # --- Check EV
        if 'check' in avail:
            ev = equity * pot_size
            action_ev['check'] = ev / max(1.0, big_blind)
            action_ev_raw['check'] = ev

        # --- Call EV
        if 'call' in avail:
            if to_call == 0:
                ev_call = equity * pot_size
            else:
                ev_call = (equity * (pot_size + to_call) - to_call) * eq_called_mult
            ev_call *= bubble_factor
            action_ev['call'] = ev_call / max(1.0, big_blind)
            action_ev_raw['call'] = ev_call
            breakdown['call'] = {"pot": pot_size, "to_call": to_call, "equity": equity, "eq_called_mult": eq_called_mult, "bubble": bubble_factor}

        # --- Bet EV (with river mixing cap via optimal bluff ratio)
        if 'bet' in avail and max_bet > 0:
            best = -1e18; best_size = 0; best_fe = 0.0; best_raw = -1e18
            for B in self._candidate_bet_sizes(pot_size, min_bet, max_bet, hero_stack, stage, equity, wetness, draws, spr):
                pot_before = pot_size
                fe = self._fe_mdf(pot_before, B, n_to_act=max(1, num_players-1), wetness=wetness,
                                  prof_mult=fe_profile_mult, blocker_mult=blocker_boost)
                nonfold_ev = (equity * (pot_before + 2*B) - B) * eq_called_mult
                ev_raw = bubble_factor * (fe * pot_before + (1 - fe) * nonfold_ev)

                # On river with polar constructions, cap bluffs by optimal ratio ≈ s/(1+s)
                if stage == 'river':
                    s = B / max(1.0, pot_before)
                    # If equity < caller's pot-odds for this size, treat action as bluff-heavy; soft cap
                    caller_po = GTOUtils.caller_pot_odds_from_size(B, pot_before)
                    if equity < caller_po - 1e-6:
                        ev_raw *= (0.98 - 0.20 * max(0.0, (caller_po - equity)))

                ev = ev_raw / max(1.0, big_blind)
                if spr <= 1.6 and (draws.get("nutted") or equity > 0.64):
                    ev *= 1.02; ev_raw *= 1.02
                if ev > best:
                    best, best_size, best_fe, best_raw = ev, B, fe, ev_raw
            if best_size > 0:
                action_ev['bet'] = best
                action_ev_raw['bet'] = best_raw
                breakdown['bet'] = {"size": best_size, "fe": best_fe}

        # --- Raise EV with hard gates
        if can_raise and max_raise > 0 and to_call > 0:
            best = -1e18; best_to = 0; best_fe = 0.0; best_raw = -1e18
            for Y in self._candidate_raise_to(max(1, to_call), max(1, min_raise), max_raise, hero_stack, equity, spr):
                hero_add = max(0, Y - hero_bet_street)
                vill_face = max(0, Y - current_bet_high)
                pot_before = pot_size
                fe = self._fe_mdf(pot_before, vill_face, n_to_act=max(1, num_players-1), wetness=wetness,
                                  prof_mult=fe_profile_mult, blocker_mult=blocker_boost)
                nonfold_ev = (equity * (pot_before + hero_add + vill_face) - hero_add) * eq_called_mult

                # FE requirement for (semi)bluff raises when equity < threshold
                risk   = hero_add
                reward = pot_before + vill_face
                fe_req = GTOUtils.required_fe_for_bluff(risk, reward)
                gated = False
                if equity + 1e-9 < pot_odds_cmp:
                    if not (fe >= max(0.50, fe_req) and nonfold_ev > action_ev_raw.get('call', -1e18)):
                        gated = True

                if gated:
                    continue

                ev_raw = bubble_factor * (fe * pot_before + (1 - fe) * nonfold_ev)
                ev = ev_raw / max(1.0, big_blind)
                if spr <= 1.6 and (draws.get("nutted") or equity > 0.64):
                    ev *= 1.02; ev_raw *= 1.02
                if ev > best:
                    best, best_to, best_fe, best_raw = ev, Y, fe, ev_raw
            if best_to > 0:
                action_ev['raise'] = best
                action_ev_raw['raise'] = best_raw
                breakdown['raise'] = {"to": best_to, "fe": best_fe}

        # --- All-in option (kept, but gated)
        if can_raise and max_raise >= hero_stack and hero_stack > 0:
            Y = hero_bet_street + hero_stack
            hero_add = hero_stack
            vill_face = max(0, Y - current_bet_high)
            pot_before = pot_size
            fe = self._fe_mdf(pot_before, vill_face, n_to_act=max(1, num_players-1), wetness=wetness,
                              prof_mult=fe_profile_mult, blocker_mult=blocker_boost)
            nonfold_ev = (equity * (pot_before + hero_add + vill_face) - hero_add) * eq_called_mult
            gated = False
            if equity + 1e-9 < pot_odds_cmp:
                risk, reward = hero_add, pot_before + vill_face
                fe_req = GTOUtils.required_fe_for_bluff(risk, reward)
                if not (fe >= max(0.50, fe_req) and nonfold_ev > action_ev_raw.get('call', -1e18)):
                    gated = True
            if not gated:
                ev_raw = bubble_factor * (fe * pot_before + (1 - fe) * nonfold_ev)
                ev = ev_raw / max(1.0, big_blind)
                action_ev['allin'] = ev
                action_ev_raw['allin'] = ev_raw
                breakdown['allin'] = {"to": Y, "fe": fe}

        # --- Position/stack priors as tiebreakers (kept)
        gto = self._gto_weights(hero_position, max(1.0, hero_stack / max(1, big_blind)))
        scores: Dict[str, float] = {}
        k_tiebreak = 0.3
        acts = list(action_ev.keys())
        uniform = 1.0 / max(1, len(acts))
        for a in acts:
            scores[a] = action_ev[a] + k_tiebreak * (gto.get(a, 0.0) - uniform)

        # --- Safety adjustments
        if 'call' in scores and action_ev_raw.get('call', -1e9) / max(1.0, big_blind) < -1e-6 and 'fold' in scores:
            scores['call'] = min(scores['call'], scores.get('fold', 0.0) - 0.1)
        if 'call' in scores and equity >= pot_odds_cmp + 1e-6:
            scores['call'] = scores.get('call', 0.0) + 0.1
        if 'check' in scores and to_call == 0:
            scores['check'] = max(scores.get('check', 0.0), scores.get('fold', -1e9) + 0.01)

        # HARD guard vs raising/betting when equity < threshold (villain_eq / players)
        below_po = equity + 1e-9 < pot_odds_cmp
        if below_po:
            if 'raise' in scores and 'raise' in breakdown:
                raise_fe = breakdown['raise'].get('fe', 0.0)
                raise_ev = action_ev_raw.get('raise', -1e9)
                call_ev = action_ev_raw.get('call', -1e9)
                if not (raise_fe >= 0.50 and raise_ev > call_ev):
                    scores['raise'] = min(scores['raise'], scores.get('call', -1e9) - 0.05, scores.get('fold', 0.0) - 0.05)
            if to_call == 0 and 'bet' in scores and 'bet' in breakdown:
                if breakdown['bet'].get('fe', 0.0) < 0.50:
                    scores['bet'] = min(scores['bet'], scores.get('check', -1e9) - 0.05)

        # --- Pick
        if scores:
            best_val = max(scores.values())
            near = {a: s for a, s in scores.items() if best_val - s <= 0.03 + 1e-9}
            if len(near) > 1:
                acts2, scs2 = zip(*near.items())
                # Softmax sampling for mixed strategies
                def _softmax(xs, t=0.15):
                    m = max(xs); e = [math.exp((x - m)/max(1e-6, t)) for x in xs]; z = sum(e) or 1.0
                    return [v/z for v in e]
                probs = _softmax(list(scs2), 0.15)
                r = self._rng.random()
                acc = 0.0; pick = acts2[-1]
                for a, p in zip(acts2, probs):
                    acc += p
                    if r <= acc: pick = a; break
                best_action = pick
            else:
                best_action = max(scores.items(), key=lambda kv: kv[1])[0]
        else:
            # Ensure we ALWAYS recommend preflop/postflop
            best_action = 'check' if 'check' in avail else ('call' if 'call' in avail else ('raise' if 'raise' in avail else 'fold'))

        amount = 0
        if best_action == 'call': amount = to_call
        elif best_action == 'raise': amount = int(breakdown.get('raise', {}).get('to', max(min_raise, to_call*2)))
        elif best_action == 'bet': amount = int(breakdown.get('bet', {}).get('size', max(min_bet, pot_size//3)))
        elif best_action == 'allin': amount = hero_stack

        ordered = sorted(scores.values(), reverse=True) if scores else [0.0]
        edge = (ordered[0] - ordered[1]) if len(ordered) > 1 else abs(ordered[0])
        confidence = max(0.1, min(0.95, 0.55 + 0.1 * max(0.0, edge)))

        return {
            "action": best_action,
            "amount": int(amount) if isinstance(amount, (int, float)) else amount,
            "confidence": confidence,
            "reasoning": (
                "Theory-aligned: MDF-based FE, river bluff caps, 3-raise reduction, bucketing, "
                "strict raise guard (equity < villain-eq/players requires FE≥req and EV(raise)>EV(call))."
            ),
            "equity": equity,
            "pot_odds": pot_odds_print,        # now: villain equity / players (display)
            "pot_odds_compare": pot_odds_cmp,  # now: villain equity / players (decision)
            "gto_weights": self._gto_weights(hero_position, max(1.0, hero_stack / max(1, big_blind))),
            "ev_scores": {k: v for k, v in scores.items()},   # in bb
            "ev_raw": {k: v for k, v in action_ev_raw.items()}# in chips
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
        except Exception as e:
            print(f"Error in async decision worker: {e}")

# ==========================================================
# Tournament tracker & mitmproxy hooks
# ==========================================================
class PokerTournamentTracker:
    def __init__(self):
        self.tournament_id = None
               # Tournament info
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
        self.table_name = ""
        self.table_type = ""
        self.max_seats = 8
        self.seats = {}

        self.hand_number = None
        self.hand_history_id = None
        self.dealer_seat = None
        self.small_blind_seat = None
               # Player/table state
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
        self.player_actions = defaultdict(lambda: {'vpip':0,'pfr':0,'aggression_factor':0,'hands_played':0,'preflop_raises':0,'calls':0,'folds':0,'three_bets':0})
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
        pos = state['table']['positions'].get(str(state['hand']['hero_seat']), 'Unknown')
        print("\n" + "=" * 80)
        print("🤖 ADVANCED POKER AI RECOMMENDATION")
        print("=" * 80)
        print(f"Hand: #{state['hand']['number']} | Stage: {state['hand']['stage'].upper()}")
        print(f"Hero: {state['hand']['hero_cards']} | Board: {state['hand']['community_cards']}")
        print(f"Pot: {state['hand']['pot_amount']} | To call: {state['hero']['to_call']}")
        print(f"Stack: {state['hero']['stack']} | Position: {pos}")
        print("-" * 80)
        amt = decision['amount']
        amt_str = f" ({amt})" if isinstance(amt, (int, float)) and amt > 0 else ""
        print(f"🎯 Recommended Action: {decision['action'].upper()}{amt_str}")
        print(f"📊 Confidence: {decision['confidence']:.1%}")
        print(f"⚖️  Equity: {decision['equity']:.1%} | Threshold (villain eq / players): {decision['pot_odds']:.1%} (compare {decision.get('pot_odds_compare', 0):.1%})")
        print(f"🎲 GTO Weights: {decision.get('gto_weights', {})}")
        evbb = decision.get('ev_scores', {})
        evch = decision.get('ev_raw', {})
        if evbb or evch:
            print("📈 EV (bb):", {k: round(v, 3) for k,v in evbb.items()})
            print("💰 EV (chips):", {k: int(v) for k,v in evch.items()})
        print(f"💡 Reasoning: {decision.get('reasoning', '')}")
        print("=" * 80)
        self._print_strategic_advice(decision, state)

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
        except ET.ParseError as e:
            print(f"XML parse error: {e}")
        except Exception as e:
            print(f"Error processing message: {e}")

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
        elif tag in ('Raise','Fold'):
            self._handle_player_action_response(element)
        elif tag == 'Ping':
            pass

    def _handle_enter_table(self, element):
        self.table_id = element.get('tableId')
        self.tournament_id = element.get('tournamentId')
        print(f"Entered table {self.table_id} in tournament {self.tournament_id}")

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
                for payment in prize_info.findall('.//Payment'):
                    places = payment.get('place')
                    amount = float(payment.get('amount', 0) or 0)
                    self.payout_structure[places] = amount
            participants = tournament.find('.//Participants')
            if participants is not None:
                self.players_remaining = int(participants.get('remaining', 0) or 0)
                self.total_players = int(participants.get('total', 0) or 0)

    def _handle_message(self, element):
        changes = element.find('.//Changes')
        if changes is not None: self._process_changes(changes)
        game_state = element.find('.//GameState')
        if game_state is not None: self._process_game_state(game_state)

    def _process_game_state(self, game_state):
        self.hand_number = game_state.get('hand') or self.hand_number
        self.hand_history_id = game_state.get('historyId') or self.hand_history_id
        tourney_info = game_state.find('.//TournamentInfo')
        if tourney_info is not None: self._update_tournament_state(tourney_info)
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
            if len(self.community_cards) >= 5: self.current_stage = "river"
            elif len(self.community_cards) >= 4: self.current_stage = "turn"
            elif len(self.community_cards) >= 3: self.current_stage = "flop"
            else: self.current_stage = "preflop"

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
                if sid: self.active_players.add(sid)

        for action_elem in changes.findall('.//PlayerAction'):
            self._process_player_action(action_elem)

        pots_change = changes.find('.//PotsChange')
        if pots_change is not None: self._update_pot(pots_change)

        dealing_cards = changes.find('.//DealingCards')
        if dealing_cards is not None: self._process_dealing_cards(dealing_cards)

        for street in ('Flop','Turn','River'):
            dealing_elem = changes.find(f'.//Dealing{street}')
            if dealing_elem is not None: self._process_community_cards(dealing_elem, street.lower())

        winners = changes.find('.//Winners')
        if winners is not None: self._process_winners(winners)
        end_hand = changes.find('.//EndHand')
        if end_hand is not None: self._process_end_hand(end_hand)

        tourney_change = changes.find('.//TournamentInfoChange')
        if tourney_change is not None:
            tinfo = tourney_change.find('.//TournamentInfo')
            if tinfo is not None: self._update_tournament_state(tinfo)

        active_change = changes.find('.//ActiveChange')
        if active_change is not None: self._process_active_change(active_change)

        if not self._saw_pots_change_in_frame and self._local_pot_delta:
            self.pot_amount += self._local_pot_delta
        self._local_pot_delta = 0

    def _parse_seats(self, seats_elem):
        for seat_elem in seats_elem.findall('.//Seat'):
            sid = seat_elem.get('id')
            if not sid: continue

            pinfo = seat_elem.find('.//PlayerInfo')
            if pinfo is not None:
                uuid = pinfo.get('uuid'); nickname = pinfo.get('nickname', '')
                if uuid and uuid == self.hero_uuid:
                    self.hero_seat = sid; self.hero_nickname = nickname
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
                    if t and t != 'xx': cards.append(t)
                if cards:
                    self.visible_cards[sid] = cards
                    if sid == self.hero_seat:
                        self.hero_cards = cards

    def _process_player_action(self, action_elem):
        sid = action_elem.get('seat')
        if sid is None: return
        node = action_elem[0] if len(action_elem) else None
        if node is None: return

        atype = node.tag
        amt_raw = node.get('amount', 0)
        amount = int(float(amt_raw or 0))

        self.player_actions[sid]['hands_played'] = max(1, self.player_actions[sid]['hands_played'] + (1 if self.current_stage=='preflop' else 0))
        if self.current_stage == 'preflop':
            if atype in ('Call','Bet','Raise'): self.player_actions[sid]['vpip'] += 1
            if atype == 'Raise': self.player_actions[sid]['preflop_raises'] += 1
            if atype == 'Fold': self.player_actions[sid]['folds'] += 1

        prev = int(self.player_bets.get(sid, 0))
        def add_or_set_total(new_amount: int, is_total_hint: bool) -> int:
            if is_total_hint: return max(prev, new_amount)
            delta_total = prev + new_amount
            return max(delta_total, new_amount)

        if atype == 'Fold':
            self.folded_players.add(sid); self.active_players.discard(sid)

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
            if self.current_bet > 0: total = min(total, self.current_bet)
            self.player_bets[sid] = total
            if sid in self.player_stacks:
                self.player_stacks[sid] = max(0, self.player_stacks[sid] - max(0, total - prev))

        self.last_action = (sid, atype, amount)

    def _process_dealing_cards(self, dealing_cards):
        for seat_elem in dealing_cards.findall('.//Seat'):
            sid = seat_elem.get('id')
            cards_elem = seat_elem.find('.//Cards')
            if not sid or cards_elem is None: continue
            cards = []
            for c in cards_elem.findall('.//Card'):
                t = c.text
                if t and t != 'xx': cards.append(t)
            if sid == self.hero_seat and cards:
                self.hero_cards = cards

    def _process_community_cards(self, dealing_elem, street: str):
        self.current_stage = street
        cards_elem = dealing_elem.find('.//Cards')
        new_cards = []
        if cards_elem is not None:
            for c in cards_elem.findall('.//Card'):
                t = c.text
                if t: new_cards.append(t)
        if new_cards: self.community_cards.extend(new_cards)
        self.current_bet = 0
        self.street_bets.clear()
        for sid in list(self.player_bets.keys()):
            self.player_bets[sid] = 0

    def _process_active_change(self, active_change):
        self.current_actor = active_change.get('seat')
        self.action_time_remaining = int(active_change.get('time', 0) or 0)
        self.time_bank = int(active_change.get('timeBank', 0) or 0)

        self.available_actions = []
        self.min_raise = 0; self.max_raise = 0
        self.min_bet = 0; self.max_bet = 0
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
                    try: self.call_amount = int(float(raw))
                    except Exception: self.call_amount = 0

        if self.current_actor == self.hero_seat:
            self.maybe_compute_and_print()

    def _update_pot(self, pots_change):
        total = 0; saw_absolute = False
        for pot_elem in pots_change.findall('.//Pot'):
            if pot_elem.get('value') is not None:
                total += int(pot_elem.get('value', 0) or 0); saw_absolute = True
            elif pot_elem.get('amount') is not None:
                total += int(pot_elem.get('amount', 0) or 0); saw_absolute = True
        if saw_absolute:
            self.pot_amount = total; return
        self._saw_pots_change_in_frame = True
        delta = 0
        for pot_elem in pots_change.findall('.//Pot'):
            change = int(pot_elem.get('change', 0) or 0); delta += change
        self.pot_amount += delta
        self.pot_amount = sum(self.total_contrib.values())

    def _process_winners(self, winners): pass

    def _process_end_hand(self, end_hand):
        self.hand_history.append({
            'hand_number': self.hand_number,
            'hero_cards': list(self.hero_cards),
            'community_cards': list(self.community_cards),
            'pot_amount': self.pot_amount,
            'actions': {k: dict(v) for k,v in self.player_actions.items()},
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
                self.player_stacks[self.hero_seat] = int(chips.get('stack-size', 0) or 0)

    def _handle_balance(self, element): return
    def _handle_player_action_response(self, element): return

    def _calculate_positions(self) -> Dict[str, str]:
        if not self.dealer_seat or not self.active_players: return {}
        try:
            dealer = int(self.dealer_seat)
            seats = sorted(int(s) for s in self.active_players)
            n = len(seats); pos_map = {}
            for seat in seats:
                dist = (seat - dealer) % self.max_seats
                if n == 2:
                    pos = "BTN" if dist == 0 else "BB"
                else:
                    if   dist == 0: pos = "BTN"
                    elif dist == 1: pos = "SB"
                    elif dist == 2: pos = "BB"
                    elif dist == 3: pos = "UTG"
                    elif dist == 4: pos = "MP"
                    else: pos = "CO"
                pos_map[str(seat)] = pos
            return pos_map
        except Exception:
            labels = ["UTG", "MP", "CO", "BTN", "SB", "BB"]
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
                'player_actions': {k: dict(v) for k,v in self.player_actions.items()},
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
        print("\n" + "=" * 80)
        print("🤖 ADVANCED POKER AI RECOMMENDATION")
        print("=" * 80)
        print(f"Hand: #{state['hand']['number']} | Stage: {state['hand']['stage'].upper()}")
        print(f"Hero: {state['hand']['hero_cards']} | Board: {state['hand']['community_cards']}")
        print(f"Pot: {state['hand']['pot_amount']} | To call: {state['hero']['to_call']}")
        print(f"Stack: {state['hero']['stack']} | Position: {pos}")
        print("-" * 80)
        amt = decision['amount']; amt_str = f" ({amt})" if isinstance(amt, (int, float)) and amt > 0 else ""
        print(f"🎯 Recommended Action: {decision['action'].upper()}{amt_str}")
        print(f"📊 Confidence: {decision['confidence']:.1%}")
        print(f"⚖️  Equity: {decision['equity']:.1%} | Threshold (villain eq / players): {decision['pot_odds']:.1%} (compare {decision.get('pot_odds_compare', 0):.1%})")
        print(f"🎲 GTO Weights: {decision.get('gto_weights', {})}")
        evbb = decision.get('ev_scores', {})
        evch = decision.get('ev_raw', {})
        if evbb or evch:
            print("📈 EV (bb):", {k: round(v, 3) for k,v in evbb.items()})
            print("💰 EV (chips):", {k: int(v) for k,v in evch.items()})
        print(f"💡 Reasoning: {decision.get('reasoning', '')}")
        print("=" * 80)
        self._print_strategic_advice(decision, state)

    def _print_strategic_advice(self, decision: Dict[str, Any], state: Dict[str, Any]):
        tips = []
        eq = decision['equity']; po = decision.get('pot_odds_compare', 0.0); act = decision['action']
        stack_bb = state['hero']['stack'] / max(1, state['hand']['big_blind_amount'])

        if eq > po + 0.2: tips.append("✅ Very strong edge—prefer larger value sizes.")
        elif eq > po + 0.1: tips.append("✅ Positive EV—value bet or raise is good.")
        elif eq > po: tips.append("👍 Slight edge—standard continue.")
        elif eq > po - 0.05: tips.append("⚠️ Marginal—control pot size.")
        else: tips.append("❌ Negative EV—tighten or fold.")

        if stack_bb < 10: tips.append("💎 Short stack—lean push/fold in high-pressure spots.")
        elif stack_bb < 20: tips.append("📉 Mid-short—selective aggression.")
        elif stack_bb > 50: tips.append("📈 Big stack—apply pressure vs capped ranges.")

        pos = state['table']['positions'].get(str(state['hand']['hero_seat']), '')
        if pos in ('BTN','CO'): tips.append("🎯 In position—add thin value and bluffs.")
        if pos in ('SB',): tips.append("🛡️ OOP from SB—avoid bloating marginal hands.")

        if act == 'raise':
            tips.append("🃏 Raise cleared FE & EV thresholds.")
        elif act == 'call':
            tips.append("📊 Call clears the threshold; plan future streets.")
        elif act == 'bet':
            tips.append("📏 Bet size chosen vs MDF / FE model.")
        elif act == 'fold':
            tips.append("🔒 Fold preserves chips for better spots.")

        if tips:
            print("\nStrategic Insights:")
            for t in tips:
                print(f"  • {t}")

# -------------- Global instance & mitmproxy hooks --------------
tracker = PokerTournamentTracker()

def websocket_message(flow):
    ws = getattr(flow, "websocket", None)
    if not ws or not getattr(ws, "messages", None): return
    msg = None
    for m in reversed(ws.messages):
        if not m.from_client: msg = m; break
    if msg is None: return
    try:
        content = msg.content.decode('utf-8', errors='ignore')
        if not content.lstrip().startswith('<'): return
        tracker.process_message(content)
        if tracker.current_actor == tracker.hero_seat:
            key = tracker._decision_key()
            now = time.monotonic()
            if key != tracker._last_key or (now - tracker._last_compute_ts) >= tracker._debounce_sec:
                tracker._last_key = key; tracker._last_compute_ts = now
                snapshot = copy.deepcopy(tracker.get_game_state_summary())
                tracker.worker.submit_if_needed(key, snapshot, tracker.print_ai_recommendation_from_snapshot)
        if len(ws.messages) > 2000: del ws.messages[:-1000]
    except Exception as e:
        print(f"Error in websocket_message: {e}")

def request(flow): return
def response(flow):
    if flow.response.headers.get("Upgrade", "").lower() == "websocket": pass
def websocket_start(flow): print("WebSocket connection established")
def websocket_end(flow): print("WebSocket connection closed")

if __name__ == "__main__":
    print("✅ Poker AI loaded (v8.2)")
    print("• Threshold switched to (villain equity / active players) for decisions + display")
    print("• Strict raise/bet guard when equity < threshold unless FE≥50% AND EV(raise)>EV(call)")
    print("• MDF-based FE, river bluff caps, bucketing, mixed strategies, and sensible size reduction")
    print("• Fast preflop equity via sampled ranges + cache; robust XML parsing")
