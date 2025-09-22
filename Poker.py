#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Poker action suggester (mitmproxy) using your gto_helper.py logic — fixed:
- uses authoritative stack from <ActiveChange><Actions><Raise max="...">
- corrects to-call from <ActiveChange><Actions><Call amount="...">
- stack-aware EV (min(to_call, stack)) so we don't fold when EV(raise) > 0
- supports 2–9 players, fast-fold safe, robust error guards

Run:
  mitmproxy -s this_script.py

Requires:
  pip install cython
  pip install eval7
"""

import xml.etree.ElementTree as ET
from collections import defaultdict, Counter, OrderedDict
from mitmproxy import http
import math
from pathlib import Path
from datetime import datetime

# =========================
# Runtime-safety settings
# =========================
MAX_ITERS_PREFLOP = 3000
MAX_ITERS_FLOP    = 4000
MAX_ITERS_TURN    = 5000
MAX_ITERS_RIVER   = 7000
VILLAINS_PENALTY  = 800
MIN_ITERS         = 1200
REUSE_DECISION_IF_SAME_STATE = True
EQ_CACHE_MAX_ENTRIES = 256

try:
    import numpy as np
except ImportError:
    np = None

try:
    import eval7
except ImportError:
    print("Please install eval7: pip install cython && pip install eval7")
    eval7 = None  # degrade gracefully


# ==============================
# ===  gto_helper.py  CORE   ===
# ==============================

SUITS, RANKS = "shdc", "23456789TJQKA"
STRICT_THRESH = {"raise": 0.65, "check": 0.4}
GAMES = ["Holdem", "Short Deck"]

class LRUCache:
    def __init__(self, capacity=EQ_CACHE_MAX_ENTRIES):
        self.capacity = capacity
        self.store = OrderedDict()
    def get(self, key):
        if key in self.store:
            val = self.store.pop(key)
            self.store[key] = val
            return val
        return None
    def put(self, key, value):
        if key in self.store:
            self.store.pop(key)
        elif len(self.store) >= self.capacity:
            self.store.popitem(last=False)
        self.store[key] = value

EQ_CACHE = LRUCache()

def card_rank_char(card: "eval7.Card") -> str:
    rc = getattr(card, "rank_char", str(card)[0])
    rc = rc.upper()
    return "T" if rc == "0" else rc

# 169 grid order
_order, ORDER = 169, {}
for r1 in RANKS[::-1]:
    for r2 in RANKS[::-1]:
        if r1 < r2:
            continue
        for suited in (True, False):
            ORDER[(r1, r2, suited)] = _order; _order -= 1

def top_range(p):
    keep = math.ceil(169 * p / 100)
    return set(k for k, _ in zip(sorted(ORDER, key=ORDER.get, reverse=True), range(keep)))

def deck_for_game(game: str) -> "eval7.Deck":
    deck = eval7.Deck()
    if game.lower().startswith("short"):
        for r in "2345":
            for s in SUITS:
                c = eval7.Card(r + s)
                try:
                    deck.cards.remove(c)
                except ValueError:
                    pass
    return deck

def _simulate(hero, board, villains, rng, weighted, iters, game):
    wins, buckets = 0.0, Counter()
    for _ in range(iters):
        d = deck_for_game(game)
        for c in hero + board:
            try:
                d.cards.remove(c)
            except ValueError:
                pass
        d.shuffle()
        opp = []
        while len(opp) < villains:
            a, b = d.deal(2)
            r1 = max(card_rank_char(a), card_rank_char(b))
            r2 = min(card_rank_char(a), card_rank_char(b))
            s = a.suit == b.suit
            if rng and (r1, r2, s) not in rng:
                d.cards.extend([a, b]); d.shuffle(); continue
            opp.append([a, b])
        sim_board = board + d.deal(max(0, 5 - len(board)))
        hero_sc = eval7.evaluate(hero + sim_board)
        best, ties, hero_best = hero_sc, 1, True
        for v in opp:
            vs = eval7.evaluate(v + sim_board)
            if vs > best:
                best, ties, hero_best = vs, 1, False
            elif vs == best:
                ties += 1; hero_best |= vs == hero_sc
        if hero_best and hero_sc == best:
            wins += 1 / max(1, ties)
            buckets[int((1 / max(1, ties)) * 10)] += 1
        else:
            buckets[0] += 1
    return wins, buckets

def equity(hero, board, villains, pct=None, custom=None, iters=25000, weighted=None, game="Holdem"):
    rng = custom if custom is not None else (top_range(pct) if (pct or 0) > 0 else None)
    key = (
        tuple(str(c) for c in hero),
        tuple(str(c) for c in board),
        int(villains),
        pct if custom is None else tuple(sorted(custom)) if custom else None,
        int(iters),
        game,
    )
    cached = EQ_CACHE.get(key)
    if cached is not None:
        return cached
    wins, buckets = _simulate(hero, board, villains, rng, weighted, iters, game)
    eq = wins / max(1, iters)
    if np:
        hist = (np.bincount([min(k, 9) for k in buckets.elements()], minlength=10) / max(1, iters)).tolist()
    else:
        hist = [buckets[i] / max(1, iters) for i in range(10)]
    result = (eq, hist)
    EQ_CACHE.put(key, result)
    return result

def strict_action(eq):
    return "RAISE" if eq >= STRICT_THRESH["raise"] else ("CHECK" if eq >= STRICT_THRESH["check"] else "FOLD")

def equilibrium_solver(hero, board, villains, pct=None, custom=None, iters=10000, game="Holdem"):
    eq, _ = equity(hero, board, villains, pct, custom, iters=iters, game=game)
    bet = round(min(max(eq, 0.0), 1.0), 2)
    check = round(1 - bet, 2)
    return {"bet_freq": bet, "check_freq": check}

RAISE_SIZES = {"0.5": .5, "1": 1.0, "2": 2.0, "shove": None}

def decide_bets(eq, pot, bet, stack, pref):
    """
    IMPORTANT: 'bet' is the *effective* to-call (capped by stack).
    """
    call_ev = eq * (pot + bet) - (1 - eq) * bet
    fold_ev = 0.0

    if pref == "shove" or stack <= bet:
        raise_total = stack
    else:
        raise_total = min(bet + (pot + 2*bet) * RAISE_SIZES.get(pref, 1.0), stack)

    raise_ev = eq * (pot + bet + raise_total) - (1 - eq) * raise_total
    best = max(fold_ev, call_ev, raise_ev)

    if best == raise_ev:
        act = "ALL_IN" if raise_total == stack else "RAISE"
        mv = {"call": bet, "raise": round(max(0.0, raise_total - bet),2), "total": round(raise_total,2),
              "breakdown": f"{bet} + {round(max(0.0, raise_total - bet),2)} = {round(raise_total,2)}"}
    elif best == call_ev:
        act, mv = "CALL", {"call": bet, "raise": 0, "total": bet, "breakdown": f"{bet} = {bet}"}
    else:
        act, mv = "FOLD", {}
    return act, fold_ev, call_ev, raise_ev, mv

def log_row(d):
    try:
        f = "gto_history.csv"
        head = ["ts","hand","board","villains","range","eq","act","ev_fold","ev_call","ev_raise"]
        need_header = not (Path(f).exists())
        with open(f,'a',newline='') as csvfile:
            import csv
            w = csv.writer(csvfile)
            if need_header:
                w.writerow(head)
            w.writerow([d.get(h) for h in head])
    except Exception:
        pass


# ==============================
# ===  Helper glue for live  ===
# ==============================

def parse_eval_cards(txt_list):
    if not eval7:
        return []
    out, seen = [], set()
    for t in txt_list or []:
        if not t or len(t) != 2:
            continue
        r, s = t[0].upper(), t[1].lower()
        if r == '0': r = 'T'
        if r not in RANKS or s not in SUITS:
            continue
        k = r + s
        if k in seen:
            continue
        seen.add(k)
        try:
            out.append(eval7.Card(k))
        except Exception:
            pass
    return out

def estimate_villain_range_pct(table_aggr: float, villains: int) -> float:
    base = 24.0 + (table_aggr - 0.5) * 20.0
    base += max(0, villains - 1) * 1.5
    return max(8.0, min(45.0, base))

def position_code_from_distance(distance: int, n_players: int) -> int:
    if distance == 0: return 3  # BTN
    if distance == 1: return 4  # SB
    if distance == 2: return 5  # BB
    k = max(0, n_players - 3)
    if k == 0:
        return 0
    i = distance - 3
    early = math.ceil(k/3.0)
    late  = math.ceil(k/3.0)
    middle = max(0, k - early - late)
    if i < early:
        return 0
    elif i < early + middle:
        return 1
    else:
        return 2

def street_max_iters(street: str) -> int:
    st = (street or "").lower()
    if st == "preflop": return MAX_ITERS_PREFLOP
    if st == "flop":    return MAX_ITERS_FLOP
    if st == "turn":    return MAX_ITERS_TURN
    if st == "river":   return MAX_ITERS_RIVER
    return MAX_ITERS_FLOP


# ==============================
# ===     MITM INTEGRATION   ===
# ==============================

class PokerGameTracker:
    def __init__(self):
        # Live state
        self.current_hand = None
        self.community_cards = []
        self.hero_cards = []
        self.player_actions = defaultdict(list)
        self.active_seats = []
        self.pot_amount = 0.0
        self.rake = 0.0
        self.jackpot_fee = 0.0
        self.bbj_amount = 0.0
        self.current_stage = "preflop"
        self.hero_stack = 0.0
        self.hero_nickname = ""
        self.hero_uuid = ""
        self.hero_seat = None
        self.player_stacks = defaultdict(float)
        self.player_names = defaultdict(str)
        self.new_street = True
        self.current_max_bet = 0.0
        self.street_bets = defaultdict(float)
        self.last_bet_size = 0.0
        self.num_players = 0
        # 0: UTG, 1: MP, 2: CO, 3: BTN, 4: SB, 5: BB
        self.position = 0
        self.available_actions = []
        self.big_blind = 2.0
        self.small_blind = 1.0
        self.dealer_seat = None
        self.preflop_raisers = []
        self.limpers = []
        self.table_size = 6
        self.game_type = "Holdem"

        # Decision memoization
        self._last_state_sig = None
        self._last_action = None

    # ---------- aggression helpers ----------
    def get_aggression_factor(self, seat):
        actions = self.player_actions.get(seat, [])
        if not actions:
            return 0.5
        aggression_count = 0.0
        total = 0.0
        for action_type, _ in actions:
            total += 1
            if action_type in ["Raise", "Bet"]:
                aggression_count += 1
            elif action_type == "Call":
                aggression_count += 0.5
        return aggression_count/total if total > 0 else 0.5

    def get_table_aggression(self):
        if not self.active_seats:
            return 0.5
        total = 0.0
        n = 0
        for seat in self.active_seats:
            if seat == self.hero_seat:
                continue
            total += self.get_aggression_factor(seat)
            n += 1
        return total/n if n > 0 else 0.5

    # ---------- position ----------
    def _update_position(self, hero_seat_id: str):
        try:
            n = int(self.num_players) if self.num_players else len(self.active_seats) or self.table_size
            n = max(2, min(9, n))
            dealer = self.dealer_seat if self.dealer_seat is not None else "0"
            distance = (int(hero_seat_id) - int(dealer)) % n
            self.position = position_code_from_distance(distance, n)
        except Exception:
            self.position = 2  # CO fallback

    # ---------- state signature ----------
    def _state_signature(self):
        try:
            sig = (
                self.current_hand,
                tuple(self.hero_cards or []),
                tuple(self.community_cards or []),
                self.current_stage,
                round(self.pot_amount, 2),
                round(self.current_max_bet, 2),
                round(self.player_stacks.get(self.hero_seat, 0.0), 2),
                tuple(sorted(self.active_seats or [])),
                tuple(sorted((s, round(self.street_bets.get(s, 0.0), 2)) for s in self.active_seats or [])),
            )
            return sig
        except Exception:
            return None

    # ---------- core decision using gto_helper ----------
    def compute_best_action(self):
        try:
            if not eval7:
                return "Install eval7 library"
            if not self.hero_cards:
                return "Unknown - no hole cards"

            if REUSE_DECISION_IF_SAME_STATE:
                sig = self._state_signature()
                if sig is not None and sig == self._last_state_sig and self._last_action:
                    return self._last_action

            to_call_raw = self.current_max_bet - self.street_bets.get(self.hero_seat, 0.0)
            to_call_raw = max(0.0, to_call_raw)
            stack = max(0.0, float(self.player_stacks.get(self.hero_seat, 0.0)))
            pot = max(0.0, float(self.pot_amount))

            hero = parse_eval_cards(self.hero_cards)
            board = parse_eval_cards(self.community_cards)

            villains = max(0, len([s for s in self.active_seats if s != self.hero_seat]))
            if villains == 0:
                action_out = ("Call" if "Call" in self.available_actions and to_call_raw > 0
                              else "Check" if "Check" in self.available_actions
                              else "Call" if "Call" in self.available_actions
                              else "Fold")
                self._remember_action(action_out)
                return action_out

            table_aggr = self.get_table_aggression()
            v_pct = estimate_villain_range_pct(table_aggr, villains)

            max_iters = street_max_iters(self.current_stage)
            iters = max(MIN_ITERS, max_iters - VILLAINS_PENALTY * max(0, villains - 1))

            try:
                eq, hist = equity(hero, board, villains, pct=v_pct, iters=int(iters), game=self.game_type)
            except Exception:
                eq, hist = 0.5, [0]*10

            # ======== DECISION ========
            if to_call_raw > 1e-9:
                # --- BETS MODE (facing a bet) ---
                bet_eff = min(to_call_raw, stack)  # stack-aware to-call
                pref = "shove" if stack <= max(self.small_blind*20, bet_eff * 2) else "1"
                act, evf, evc, evr, mv = decide_bets(eq, pot, bet_eff, stack, pref=pref)
                actions = set(self.available_actions)

                if act in ("RAISE", "ALL_IN"):
                    total = mv.get("total", bet_eff + max(self.big_blind * 2, self.last_bet_size * 2 or self.big_blind * 3))
                    total = min(total, stack)
                    if bet_eff <= 1e-9 and "Bet" in actions and "Raise" not in actions:
                        action_out = f"Bet {total:.2f}"
                    elif "Raise" in actions:
                        action_out = f"Raise {total:.2f}"
                    elif "Call" in actions:
                        action_out = "Call"
                    else:
                        action_out = "Check" if bet_eff <= 1e-9 and "Check" in actions else "Call"
                elif act == "CALL":
                    action_out = "Call" if "Call" in actions else ("Check" if bet_eff <= 1e-9 and "Check" in actions else "Call")
                else:
                    action_out = "FastFold" if "FastFold" in actions else ("Fold" if "Fold" in actions else ("Check" if bet_eff <= 1e-9 and "Check" in actions else "Call"))

                self._log_decision(eq, act, evf, evc, evr, villains, v_pct, hero, board)
                self._remember_action(action_out)
                return action_out

            else:
                # --- STRICT / OPENING MODE (no bet to face) ---
                actions = set(self.available_actions)

                if self.current_stage == "preflop":
                    freqs = equilibrium_solver(hero, board, villains, pct=v_pct, iters=int(max(MIN_ITERS, MAX_ITERS_PREFLOP)), game=self.game_type)
                    bet_freq = freqs.get("bet_freq", 0.0)
                    pos_nudge = {0: -0.05, 1: -0.02, 2: 0.0, 3: +0.05, 4: -0.03, 5: -0.04}.get(self.position, 0.0)
                    should_open = (bet_freq + pos_nudge) >= 0.5

                    if should_open and ("Bet" in actions or "Raise" in actions):
                        open_size = 2.5 * self.big_blind + len(self.limpers) * 1.0 * self.big_blind
                        open_size = max(open_size, 2.0 * self.big_blind)
                        open_size = min(open_size, max(0.0, self.player_stacks.get(self.hero_seat, 0.0)))
                        if "Bet" in actions and "Raise" not in actions:
                            action_out = f"Bet {open_size:.2f}"
                        else:
                            action_out = f"Raise {open_size:.2f}"
                    else:
                        action_out = "Check" if "Check" in actions else ("Fold" if "Fold" in actions else "Check")

                    self._log_decision(eq, "OPEN" if should_open else "CHECK", 0.0, 0.0, 0.0, villains, v_pct, hero, board)
                    self._remember_action(action_out)
                    return action_out

                # Postflop, no bet to face -> strict_action
                act = strict_action(eq)
                if act == "FOLD":
                    action_out = "Check" if "Check" in actions else ("Fold" if "Fold" in actions else "Check")
                    evf = evc = evr = 0.0
                elif act == "CHECK":
                    if "Check" in actions:
                        action_out = "Check"
                    elif "Bet" in actions:
                        bet_amt = min(stack, max(self.big_blind*1.0, pot*0.33))
                        action_out = f"Bet {bet_amt:.2f}"
                    else:
                        action_out = "Check"
                    evf = evc = evr = 0.0
                else:
                    bet_frac = 0.66 if eq >= 0.72 else 0.50
                    bet_amt = min(stack, max(self.big_blind*2.0, pot*bet_frac))
                    if "Bet" in actions:
                        action_out = f"Bet {bet_amt:.2f}"
                    elif "Raise" in actions:
                        action_out = f"Raise {bet_amt:.2f}"
                    else:
                        action_out = "Check" if "Check" in actions else "Call"
                    evf = evc = evr = 0.0

                self._log_decision(eq, act, evf, evc, evr, villains, v_pct, hero, board)
                self._remember_action(action_out)
                return action_out

        except Exception:
            if "Check" in (self.available_actions or []):
                fallback = "Check"
            elif "Call" in (self.available_actions or []):
                fallback = "Call"
            else:
                fallback = "Fold"
            self._remember_action(fallback)
            return fallback

    def _remember_action(self, action_out: str):
        if REUSE_DECISION_IF_SAME_STATE:
            self._last_state_sig = self._state_signature()
            self._last_action = action_out

    def _log_decision(self, eq, act, evf, evc, evr, villains, v_pct, hero_cards_eval, board_cards_eval):
        try:
            ts = datetime.now().isoformat()
            hand_txt = " ".join(str(c) for c in hero_cards_eval)
            board_txt = " ".join(str(c) for c in board_cards_eval)
            log_row({
                "ts": ts,
                "hand": hand_txt,
                "board": board_txt,
                "villains": villains,
                "range": round(v_pct, 2),
                "eq": round(eq, 3),
                "act": act,
                "ev_fold": round(evf, 3) if isinstance(evf, (int, float)) else 0,
                "ev_call": round(evc, 3) if isinstance(evc, (int, float)) else 0,
                "ev_raise": round(evr, 3) if isinstance(evr, (int, float)) else 0,
            })
        except Exception:
            pass

    # ---------------------------
    # XML parsing / state updates
    # ---------------------------
    def process_websocket_message(self, message: str):
        try:
            if not message or not message.strip().startswith('<'):
                return
            root = ET.fromstring(message)

            if root.tag == 'PlayerInfo':
                self.handle_player_info(root)
            elif root.tag == 'Message':
                self.process_game_updates(root)
            elif root.tag == 'TableDetails':
                self.handle_table_details(root)

        except ET.ParseError:
            pass
        except Exception:
            pass

    def handle_player_info(self, player_info):
        try:
            self.hero_nickname = player_info.get('nickname', '') or self.hero_nickname
            self.hero_uuid = player_info.get('uuid', '') or self.hero_uuid

            # External balances (can be useful, but we trust per-seat chips for in-hand stack)
            balance = player_info.find('.//Balance[@wallet="USD"]')
            if balance is not None:
                cash = balance.get('cash', None)
                if cash is not None:
                    self.hero_stack = float(cash)
        except Exception:
            pass

    def handle_table_details(self, table_details):
        try:
            blinds = table_details.find('.//Blinds')
            if blinds is not None:
                sb = blinds.find('.//SmallBlind')
                bb = blinds.find('.//BigBlind')
                if sb is not None:
                    self.small_blind = float(sb.get('amount', '1'))
                if bb is not None:
                    self.big_blind = float(bb.get('amount', '2'))

            seats = table_details.find('.//Seats')
            if seats is not None:
                self.table_size = len(seats.findall('.//Seat'))
                self.table_size = max(2, min(9, self.table_size))

            g = table_details.get('game')
            if g and 'short' in g.lower():
                self.game_type = "Short Deck"
        except Exception:
            pass

    def process_game_updates(self, root):
        try:
            game_state = root.find(".//GameState")
            if game_state is not None:
                self.handle_game_state(game_state)

            changes = root.find(".//Changes")
            if changes is None:
                return

            new_hand = changes.find(".//NewHand")
            if new_hand is not None:
                self.handle_new_hand(new_hand)

            active_seats = changes.find(".//ActiveSeats")
            if active_seats is not None:
                self.handle_active_seats(active_seats)

            for action in changes.findall(".//PlayerAction"):
                self.handle_player_action(action)

            pots_change = changes.find(".//PotsChange")
            if pots_change is not None:
                self.handle_pot_changes(pots_change)

            rake_change = changes.find(".//RakeChange")
            if rake_change is not None:
                self.handle_rake_change(rake_change)

            jackpot_fee_change = changes.find(".//JackpotFeeChange")
            if jackpot_fee_change is not None:
                self.handle_jackpot_fee_change(jackpot_fee_change)

            bbj = changes.find(".//BadBeatJackpot")
            if bbj is not None:
                try:
                    self.bbj_amount = float(bbj.get("amount", "0"))
                except Exception:
                    self.bbj_amount = 0.0

            dealing_cards = changes.find(".//DealingCards")
            if dealing_cards is not None:
                self.handle_dealing_cards(dealing_cards)
                self.reset_street_bets()

            dealing_flop = changes.find(".//DealingFlop")
            if dealing_flop is not None:
                self.handle_community_cards(dealing_flop, "flop")
                self.reset_street_bets()
                self.new_street = True

            dealing_turn = changes.find(".//DealingTurn")
            if dealing_turn is not None:
                self.handle_community_cards(dealing_turn, "turn")
                self.reset_street_bets()
                self.new_street = True

            dealing_river = changes.find(".//DealingRiver")
            if dealing_river is not None:
                self.handle_community_cards(dealing_river, "river")
                self.reset_street_bets()
                self.new_street = True

            active_change = changes.find(".//ActiveChange")
            if active_change is not None:
                self.handle_active_change(active_change)

            winners = changes.find(".//Winners")
            if winners is not None:
                self.handle_winners(winners)
        except Exception:
            pass

    def reset_street_bets(self):
        try:
            self.current_max_bet = 0.0
            self.street_bets = defaultdict(float)
            self.last_bet_size = 0.0
        except Exception:
            pass

    def handle_game_state(self, game_state):
        try:
            self.current_hand = game_state.get("hand")

            seats = game_state.find(".//Seats")
            if seats is not None:
                self.num_players = len(seats.findall(".//Seat"))
                self.num_players = max(2, min(9, self.num_players))
                self.dealer_seat = seats.get("dealer")
                for seat in seats.findall(".//Seat"):
                    sid = seat.get("id")
                    player_info = seat.find(".//PlayerInfo")
                    name = player_info.get("anonym") if player_info is not None else "Unknown"
                    self.player_names[sid] = name
                    chips = player_info.find(".//Chips") if player_info is not None else None
                    stack = chips.get("stack-size") if chips is not None else "0"
                    try:
                        self.player_stacks[sid] = float(stack)
                    except Exception:
                        self.player_stacks[sid] = 0.0

            bbj = game_state.find(".//BadBeatJackpot")
            if bbj is not None:
                try:
                    self.bbj_amount = float(bbj.get("amount", "0"))
                except Exception:
                    self.bbj_amount = 0.0
        except Exception:
            pass

    def handle_new_hand(self, new_hand):
        try:
            self.current_hand = new_hand.get("number")
            self.community_cards = []
            self.hero_cards = []
            self.player_actions.clear()
            self.pot_amount = 0.0
            self.rake = 0.0
            self.jackpot_fee = 0.0
            self.hero_seat = None
            self.new_street = True
            self.current_stage = "preflop"
            self.reset_street_bets()
            self.available_actions = []
            self.preflop_raisers = []
            self.limpers = []
            self._last_state_sig = None
            self._last_action = None

            d = new_hand.get("dealer")
            if d is not None:
                self.dealer_seat = d
        except Exception:
            pass

    def handle_active_seats(self, active_seats):
        try:
            self.active_seats = [seat.get("id") for seat in active_seats.findall(".//Seat")]
        except Exception:
            self.active_seats = []

    def handle_player_action(self, action):
        try:
            seat = action.get("seat")
            if len(action) == 0:
                return
            sub_action = action[0]
            action_type = sub_action.tag
            amount = sub_action.get("amount", "0")
            self.player_actions[seat].append((action_type, amount))

            # Track preflop limpers/raisers
            if self.current_stage == "preflop":
                if action_type == "Raise" and seat not in self.preflop_raisers:
                    self.preflop_raisers.append(seat)
                elif action_type == "Call":
                    try:
                        if float(amount) == self.big_blind and seat not in self.limpers and seat not in self.preflop_raisers:
                            self.limpers.append(seat)
                    except Exception:
                        pass

            try:
                amount_f = float(amount)
            except Exception:
                amount_f = 0.0

            if action_type in ["PostSmallBlind", "PostBigBlind", "Bet"]:
                previous_max = self.current_max_bet
                self.street_bets[seat] = amount_f
                self.current_max_bet = max(self.current_max_bet, amount_f)
                if action_type == "Bet":
                    self.last_bet_size = amount_f
            elif action_type == "Raise":
                previous_max = self.current_max_bet
                self.street_bets[seat] = amount_f
                self.current_max_bet = max(self.current_max_bet, amount_f)
                inc = max(0.0, amount_f - previous_max)
                self.last_bet_size = inc if inc > 0 else amount_f
            elif action_type == "Call":
                self.street_bets[seat] += amount_f
            elif action_type == "UncalledBet":
                try:
                    self.player_stacks[seat] += amount_f
                except Exception:
                    pass

            if action_type in ["Bet", "Raise", "Call", "PostSmallBlind", "PostBigBlind"]:
                try:
                    self.player_stacks[seat] -= amount_f
                except Exception:
                    pass

            if action_type in ["Fold", "FastFold"]:
                if seat in self.active_seats:
                    self.active_seats.remove(seat)
        except Exception:
            pass

    def handle_pot_changes(self, pots_change):
        try:
            total_change = 0.0
            for pot in pots_change.findall(".//Pot"):
                try:
                    change = float(pot.get("change", "0"))
                except Exception:
                    change = 0.0
                total_change += change
            self.pot_amount += total_change
        except Exception:
            pass

    def handle_rake_change(self, rake_change):
        try:
            node = rake_change.find(".//Rake")
            change = float(node.get("change", "0")) if node is not None else 0.0
            self.rake += change
        except Exception:
            pass

    def handle_jackpot_fee_change(self, jackpot_fee_change):
        try:
            node = jackpot_fee_change.find(".//JackpotFee")
            change = float(node.get("change", "0")) if node is not None else 0.0
            self.jackpot_fee += change
        except Exception:
            pass

    def handle_dealing_cards(self, dealing_cards):
        try:
            for seat in dealing_cards.findall(".//Seat"):
                seat_id = seat.get("id")
                cards_elem = seat.find(".//Cards")
                cards = [card.text for card in cards_elem.findall(".//Card") if card.text is not None] if cards_elem is not None else []
                if cards and cards[0] != 'xx':
                    self.hero_seat = seat_id
                    self.hero_cards = cards
                    self.player_names[seat_id] = self.hero_nickname
                    self._update_position(seat_id)
        except Exception:
            pass

    def handle_community_cards(self, dealing_element, stage):
        try:
            self.current_stage = stage
            cards_element = dealing_element.find(".//Cards")
            if cards_element is not None:
                new_cards = [card.text for card in cards_element.findall(".//Card") if card.text is not None]
                self.community_cards.extend(new_cards)
        except Exception:
            pass

    def handle_active_change(self, active_change):
        try:
            seat = active_change.get("seat")
            actions_element = active_change.find(".//Actions")
            self.available_actions = []

            # Capture authoritative values when it's our turn
            hero_raise_max = None
            hero_call_amt = None

            if actions_element is not None:
                for action in actions_element:
                    self.available_actions.append(action.tag)
                    if seat == self.hero_seat:
                        if action.tag == "Raise":
                            try:
                                hero_raise_max = float(action.get("max", "0"))
                            except Exception:
                                hero_raise_max = None
                        elif action.tag == "Call":
                            try:
                                hero_call_amt = float(action.get("amount", "0"))
                            except Exception:
                                hero_call_amt = None

            # Overwrite our stack and to-call using server authoritative values
            if self.hero_seat is not None and seat == self.hero_seat:
                if hero_raise_max is not None and hero_raise_max > 0:
                    self.player_stacks[self.hero_seat] = hero_raise_max  # TRUE spendable stack
                if hero_call_amt is not None:
                    already_put_in = self.street_bets.get(self.hero_seat, 0.0)
                    # Ensure: to_call (= hero_call_amt) = current_max_bet - already_put_in
                    self.current_max_bet = hero_call_amt + already_put_in

            if self.hero_seat is None or seat != self.hero_seat:
                return

            action = self.compute_best_action()
            if action.startswith("Fold") and "FastFold" in self.available_actions:
                action = "FastFold"
            print(action)
        except Exception:
            pass

    def handle_winners(self, winners):
        try:
            for winner in winners.findall(".//Winner"):
                seat = winner.get("seat")
                amount = winner.get("amount")
                try:
                    self.player_stacks[seat] += float(amount)
                except Exception:
                    pass
        except Exception:
            pass


# Global tracker instance
tracker = PokerGameTracker()

def websocket_message(flow):
    try:
        messages = flow.websocket.messages
        if not messages:
            return
        message = messages[-1]
        if message.from_client:
            return
        try:
            content = message.content.decode('utf-8', errors='ignore')
        except Exception:
            return
        tracker.process_websocket_message(content)
    except Exception:
        pass

def request(flow: http.HTTPFlow):
    pass

def response(flow: http.HTTPFlow):
    try:
        _ = flow.response.headers.get("Upgrade", "")
    except Exception:
        pass

def websocket_start(flow):
    pass

def websocket_end(flow):
    pass

if __name__ == "__main__":
    print("Poker action suggester script loaded. Start mitmproxy with this script to intercept WebSocket traffic.")
