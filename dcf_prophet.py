#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Dynamic FMP multi-phase DCF with Monte-Carlo & full year-by-year table
----------------------------------------------------------------------
User input  : 'all' for batch processing or a single Ticker symbol.
File output : yearly_<ticker>.csv  (reliably rounded to 2 decimal places).
Console     : Intrinsic value + upside/downside vs. market, with key assumptions.

v7.5 – 2025-08-04 (Stability Guardrails)
• BETA FLOOR: Enforces a minimum terminal beta of 0.70 to prevent unrealistically
  low WACC calculations and ensure model stability.
• WACC-g SPREAD: Implements a minimum 2% spread between Terminal WACC and Terminal
  Growth to prevent terminal value from exploding.
• CASH CALCULATION: Improved logic to correctly find cash on the balance sheet
  by checking multiple common API field names.
• SECTOR AWARENESS: Excludes 'Financial Services' companies, as standard FCFF DCF
  models are not appropriate for them. This prevents erroneous valuations.
"""

import warnings
import requests
import math
import sys
import io
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
from collections import defaultdict
import time
import traceback
import os

# ---------------- console UTF-8 fallback ------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                      encoding="utf-8",
                                      errors="replace",
                                      line_buffering=True)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------- CONFIG ----------------------------------------------------
FMP_API_KEY = "435820a51119704ed53f7e7fb8a0cfec" # Replace with your FMP API key
FMP_BASE = "https://financialmodelingprep.com/api/v3"
HEADERS = {"User-Agent": "DCF-FMP/7.5"}
BILLION = 1000000000

# --- Valuation Assumptions ---
HIGH_GROWTH_YEARS = 1        # Phase 1: High growth period
CONVERGENCE_YEARS = 9        # Phase 2: Transition period (total horizon = Phase 1 + 2)
STAT_TAX = 0.25              # Statutory Tax Rate for terminal value and levering beta
MC_SIMULATIONS = 1000        # Number of Monte Carlo simulations

# ---------------------------------------------------------------------------
def coverage_spread(ic):
    """ Assigns a default risk spread based on interest coverage ratio. """
    table = [(8.5, 0.006), (6.5, 0.008), (5, 0.010), (4, 0.012), (3, 0.015), (2.5, 0.020),
             (2, 0.025), (1.5, 0.035), (1.25, 0.045), (0.8, 0.060), (0.5, 0.085), (0.2, 0.10)]
    for cut, sp in table:
        if ic >= cut:
            return sp
    return 0.120

def cum(series):
    """ Calculates the cumulative product of a series of (1 + rate). """
    return (1 + series).cumprod()

# ---------------------------------------------------------------------------
# FinancialModelingPrep thin wrapper
# ---------------------------------------------------------------------------
class FMP:
    def __init__(self, key):
        self.k = key
        self.s = requests.Session()
        self.s.headers.update(HEADERS)

    def _j(self, ep, **p):
        p["apikey"] = self.k
        try:
            r = self.s.get(f"{FMP_BASE}/{ep}", params=p, timeout=20)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"API error ({ep}): {str(e)}")
            return None

    def quote(self, t): return (self._j(f"quote/{t}") or [{}])[0]
    def prof(self, t): return (self._j(f"profile/{t}") or [{}])[0]
    def rpt(self, t, period="annual", limit=10): return self._j(f"income-statement/{t}", period=period, limit=limit) or []
    def cf(self, t, period="annual", limit=10): return self._j(f"cash-flow-statement/{t}", period=period, limit=limit) or []
    def bs(self, t, period="annual", limit=10): return self._j(f"balance-sheet-statement/{t}", period=period, limit=limit) or []
    def est(self, t): return self._j(f"analyst-estimates/{t}", limit=10) or []
    def treas(self): return self._j("treasury") or []
    def screener(self, **p): return self._j("stock-screener", **p) or []
    def fx(self, pair): return self._j(f"fx/{pair}")

# ---------------------------------------------------------------------------
# Market-level helpers
# ---------------------------------------------------------------------------
class Framework:
    def __init__(self, api: FMP):
        self.api = api
        self.RFR = self._rf()
        self.ERP = 0.055
        self._fx_cache = {}

    def _rf(self):
        """Fetches the 10-year US Treasury yield from FMP."""
        try:
            t = self.api.treas()
            return t[0].get("year10", 0.035) / 100 if t and 'year10' in t[0] else 0.035
        except Exception as e:
            print(f"Could not fetch Risk-Free Rate, using default 3.5%. Error: {e}")
            return 0.035

    def data(self, tkr):
        """Gathers all necessary FMP data for a given ticker."""
        return dict(
            quote=self.api.quote(tkr), prof=self.api.prof(tkr),
            inc_q=self.api.rpt(tkr, period="quarter", limit=4),
            cf_q=self.api.cf(tkr, period="quarter", limit=4),
            bs_q=self.api.bs(tkr, period="quarter", limit=1),
            inc_a=self.api.rpt(tkr, period="annual", limit=10),
            est=self.api.est(tkr)
        )

    def usd_factor(self, currency: str) -> float:
        """Converts any currency to USD using the FMP API."""
        currency = (currency or "USD").upper()
        if currency == "USD": return 1.0
        if currency in self._fx_cache: return self._fx_cache[currency]
        rate = 1.0
        try:
            pair_data = self.api.fx(f"{currency}USD")
            if pair_data and isinstance(pair_data, list) and pair_data[0].get('bid', 0) > 0:
                rate = float(pair_data[0]['bid'])
            else:
                inv_pair_data = self.api.fx(f"USD{currency}")
                if inv_pair_data and isinstance(inv_pair_data, list) and inv_pair_data[0].get('bid', 0) > 0:
                    rate = 1 / float(inv_pair_data[0]['bid'])
            if rate > 0:
                self._fx_cache[currency] = rate
                return rate
        except Exception as e:
            print(f"FX API error for {currency}, using fallback: {e}")
        fallback = {"JPY": 0.0065, "GBP": 1.27, "EUR": 1.08, "CAD": 0.73, "CHF": 1.11, "CNY": 0.14, "INR": 0.012}
        rate = fallback.get(currency, 1.0)
        self._fx_cache[currency] = rate
        return rate

# ---------------------------------------------------------------------------
# ----------------------------  DCF ENGINE  ----------------------------------
# ---------------------------------------------------------------------------
class DCF:
    def __init__(self, fw: Framework):
        self.fw = fw
        self.horizon = HIGH_GROWTH_YEARS + CONVERGENCE_YEARS

    @staticmethod
    def _validate(d):
        """Ensures all necessary data from FMP is present before valuation."""
        if not all([d['quote'], d['prof'], d['inc_q'], d['cf_q'], d['bs_q']]):
            raise ValueError("Core financial data missing (quote, profile, or statements).")
        if not d['inc_q'] or not d['inc_q'][0].get('revenue'):
                raise ValueError("Invalid or missing TTM revenue data.")
        if not d['quote'].get('price') or d['quote']['price'] <= 0:
            raise ValueError("Invalid or missing market price.")

    @staticmethod
    def _sum_quarters(reports):
        """Sums the last 4 or 2 quarters to get TTM data."""
        if not reports: return {}
        num_reports = 4
        if len(reports) > 1:
            try:
                d1 = datetime.fromisoformat(reports[0]['date'])
                d2 = datetime.fromisoformat(reports[1]['date'])
                if (d1 - d2).days > 150:
                    num_reports = 2
            except (ValueError, TypeError): pass
        reports_to_sum = reports[:num_reports]
        if not reports_to_sum: return {}
        ttm = defaultdict(float)
        for report in reports_to_sum:
            for key, value in report.items():
                if isinstance(value, (int, float)):
                    ttm[key] += value
        return dict(ttm)

    @staticmethod
    def _print_debug_info(tkr, data_dict):
        """Prints a formatted block of key financial data for debugging."""
        print(f"\n{'~'*20} DEBUG: Key Financials for {tkr} {'~'*20}")
        print("--- Base Financials (TTM, in Billions USD) ---")
        print(f"  Currency: {data_dict['currency']} | FX to USD: {data_dict['usd_fx']:.4f}")
        print(f"  TTM Revenue: ${data_dict['sales0']:.3f}B")
        print(f"  TTM EBIT: ${data_dict['ebit0']:.3f}B")
        print(f"  Total Debt: ${data_dict['debt0']:.3f}B")
        print(f"  Cash & Equivalents: ${data_dict['cash0']:.3f}B")
        print(f"  Book Equity: ${data_dict['equity0']:.3f}B")
        print(f"  Market Cap: ${data_dict['market_cap0']:.3f}B")
        print(f"  Invested Capital (Book): ${data_dict['invested_capital0']:.3f}B")
        print(f"  Shares Outstanding: {data_dict['shares0']/1e6:.2f}M")
        print("\n--- Current Ratios & Growth ---")
        print(f"  Operating Margin: {data_dict['margin0']:.2%}")
        print(f"  Sales-to-Capital: {data_dict['s2c0']:.2f}x")
        print(f"  Debt-to-Capital: {data_dict['d2c0']:.2%}")
        print(f"  Effective Tax Rate: {data_dict['eff_tax_rate_0']:.2%}")
        print(f"  Company Beta: {data_dict['beta0']:.2f}")
        print(f"  Interest Coverage Ratio: {data_dict['int_cov0']:.2f}x")
        print(f"  Analyst Est. Growth (Y1): {data_dict['g_analyst_y1']:.2%}")
        print(f"  Hist. Share Change (CAGR): {data_dict['share_change_rate']:.2%}")
        print("\n--- Peer Medians & Terminal Targets ---")
        print(f"  Industry Median Margin: {data_dict['industry_margin']:.2%}")
        print(f"  Industry Median Sales-to-Capital: {data_dict['industry_s2c']:.2f}x")
        print(f"  Industry Median Debt-to-Capital: {data_dict['industry_d2c']:.2%}")
        print(f"  Industry Median Beta (Raw): {data_dict['raw_industry_beta']:.2f}")
        print("  --------------------------------------")
        print(f"  Terminal Margin Target: {data_dict['terminal_margin']:.2%}")
        print(f"  Terminal S2C Target: {data_dict['terminal_s2c']:.2f}x")
        print(f"  Terminal D2C Target: {data_dict['terminal_d2c']:.2%}")
        print(f"  Terminal Beta Target (Floored): {data_dict['terminal_beta']:.2f}")
        print(f"{'~'*(48 + len(tkr))}\n")

    def _get_peer_medians(self, ticker):
        """
        Calculates median financial ratios for peers, excluding the ticker itself.
        """
        defaults = {"beta": 1.0, "margin": 0.10, "s2c": 1.0, "d2c": 0.30}
        try:
            profile = self.fw.api.prof(ticker)
            if not profile:
                print(f"⚠️ Could not fetch profile for {ticker}. Using global defaults.")
                return defaults

            industry = profile.get('industry')
            sector = profile.get('sector')
            peers_data = []

            screener_args = {
                "marketCapMoreThan": 1000000000,
                "revenueMoreThan": 100000000,
                "isActivelyTrading": True,
                "limit": 25
            }

            if industry:
                print(f"Finding peers for industry: '{industry}'...")
                peers_data = self.fw.api.screener(industry=industry, **screener_args)

            if not peers_data and sector:
                print(f"⚠️ Could not find industry peers. Broadening search to sector: '{sector}'...")
                peers_data = self.fw.api.screener(sector=sector, **screener_args)

            if not peers_data:
                print("⚠️ Could not find any industry or sector peers. Using global defaults.")
                return defaults

            peer_info = {
                p['symbol']: p.get('beta')
                for p in peers_data if p.get('symbol') and p.get('symbol') != ticker
            }

            if not peer_info:
                print(f"⚠️ Found peers, but none other than {ticker} itself. Using global defaults.")
                return defaults

            print(f"Fetching TTM ratios for up to {len(peer_info)} peers...")
            peer_betas, peer_margins, peer_s2c, peer_d2c = [], [], [], []

            for symbol, beta in peer_info.items():
                try:
                    if beta is not None: peer_betas.append(float(beta))
                except (ValueError, TypeError):
                    pass

                ttm_ratios_list = self.fw.api._j(f"ratios-ttm/{symbol}")
                if not ttm_ratios_list or not isinstance(ttm_ratios_list, list):
                    continue

                peer_ratios = ttm_ratios_list[0]
                if not peer_ratios: continue

                try:
                    if peer_ratios.get('operatingProfitMarginTTM') is not None:
                        peer_margins.append(float(peer_ratios['operatingProfitMarginTTM']))
                    if peer_ratios.get('totalDebtToCapitalizationTTM') is not None:
                        peer_d2c.append(float(peer_ratios['totalDebtToCapitalizationTTM']))
                except (ValueError, TypeError, AttributeError):
                    pass

                try:
                    p_s = peer_ratios.get('priceToSalesRatioTTM')
                    d_e = peer_ratios.get('totalDebtToCapitalizationTTM')
                    if p_s is not None and d_e is not None:
                         peer_s2c.append(1 / (float(p_s) * (1 + float(d_e))))
                except (ValueError, TypeError, ZeroDivisionError, AttributeError):
                    pass

            return {
                "beta": np.median([b for b in peer_betas if b is not None]) if peer_betas else defaults['beta'],
                "margin": np.median([m for m in peer_margins if m is not None]) if peer_margins else defaults['margin'],
                "s2c": np.median([s for s in peer_s2c if s is not None]) if peer_s2c else defaults['s2c'],
                "d2c": np.median([d for d in peer_d2c if d is not None]) if peer_d2c else defaults['d2c']
            }

        except Exception as e:
            print(f"An unexpected error occurred in _get_peer_medians: {e}. Using global defaults.")
            traceback.print_exc()
            return defaults

    def _project(self, start, end, start_period, total_period):
        """Linearly projects a value from a start to an end value over a defined period."""
        path = [start] * start_period
        convergence_path = np.linspace(start, end, (total_period - start_period) + 1).tolist()
        path.extend(convergence_path[1:])
        return path

    def value(self, tkr, mc_inputs=None, cached_data=None):
        """Performs the full DCF valuation with stability guardrails."""
        mc_inputs = mc_inputs or {}
        if cached_data:
            d = cached_data['d']
            peer_medians = cached_data['peer_medians']
        else:
            d = self.fw.data(tkr)
            self._validate(d)
            peer_medians = self._get_peer_medians(tkr)

        ttm_inc = self._sum_quarters(d['inc_q'])
        ttm_cf = self._sum_quarters(d['cf_q'])
        latest_bs = d['bs_q'][0] if d['bs_q'] else {}
        currency = latest_bs.get("reportedCurrency", "USD")
        usd_fx = self.fw.usd_factor(currency)
        def fx(v): return (v or 0) * usd_fx
        
        q, p, est, inc_a = d['quote'], d['prof'], d['est'], d['inc_a']
        
        ttm_revenue_native = ttm_inc.get("revenue", 0)
        if ttm_revenue_native <= 0: raise ValueError("TTM Revenue is zero or negative.")
        
        analyst_est_native = est[0].get('estimatedRevenueAvg') if est and est[0].get('estimatedRevenueAvg') else None
        if analyst_est_native and ttm_revenue_native > 0:
            g_analyst_y1 = (analyst_est_native / ttm_revenue_native) - 1
        else:
            g_analyst_y1 = 0.10

        def to_b(val): return (val or 0) / BILLION
        
        sales0 = fx(to_b(ttm_revenue_native))
        ebit0 = fx(to_b(ttm_inc.get("operatingIncome")))
        shares0 = q.get("sharesOutstanding", 0)
        debt0 = fx(to_b(latest_bs.get("totalDebt")))
        
        cash_val = latest_bs.get("cashAndMarketableSecurities")
        if cash_val is None or cash_val == 0:
            cash_val = latest_bs.get("cashAndCashEquivalents", 0)
        cash0 = fx(to_b(cash_val))

        market_cap0 = fx(to_b(q.get('marketCap')))
        equity0 = fx(to_b(latest_bs.get("totalStockholdersEquity"))) if latest_bs.get("totalStockholdersEquity", 0) > 0 else market_cap0
        
        margin0 = ebit0 / sales0 if sales0 != 0 else 0
        invested_capital0 = debt0 + equity0
        d2c0 = debt0 / invested_capital0 if invested_capital0 > 0 else 0.5
        s2c0 = sales0 / invested_capital0 if invested_capital0 > 0 else 1.2
        pretax_income_ttm = fx(to_b(ttm_inc.get('pretaxIncome')))
        tax_expense_ttm = fx(to_b(ttm_inc.get('incomeTaxExpense')))
        eff_tax_rate_0 = abs(tax_expense_ttm / pretax_income_ttm) if pretax_income_ttm not in [0, None] else STAT_TAX
        interest_expense0 = abs(fx(to_b(ttm_cf.get("interestExpense"))))
        int_cov0 = ebit0 / interest_expense0 if interest_expense0 > 0 else 20
        beta0 = float(p.get("beta", 1.0))

        share_change_rate = 0.0
        if inc_a and len(inc_a) > 1:
            shares_start = inc_a[-1].get('weightedAverageShsOutDil')
            shares_end = inc_a[0].get('weightedAverageShsOutDil')
            if shares_start and shares_end and shares_start > 0:
                cagr = (shares_end / shares_start)**(1/(len(inc_a)-1)) - 1
                share_change_rate = max(-0.05, min(cagr, 0.05))

        raw_industry_beta = peer_medians.get('beta')
        terminal_beta = max(raw_industry_beta, 0.70)

        industry_margin = peer_medians.get('margin')
        industry_s2c = peer_medians.get('s2c')
        industry_d2c = peer_medians.get('d2c')
        
        simulated_margin = mc_inputs.get('margin_terminal')
        if simulated_margin is not None:
            terminal_margin = simulated_margin
        else:
            terminal_margin = (margin0 * 2 + industry_margin) / 3
        
        terminal_s2c = (s2c0 * 2 + industry_s2c) / 3 if industry_s2c > 0 else s2c0
        terminal_d2c = (d2c0 * 2 + industry_d2c) / 3

        if not mc_inputs:
            debug_data = {
                "currency": currency, "usd_fx": usd_fx, "sales0": sales0,
                "ebit0": ebit0, "debt0": debt0, "cash0": cash0, "equity0": equity0,
                "market_cap0": market_cap0, "invested_capital0": invested_capital0,
                "shares0": shares0, "margin0": margin0, "s2c0": s2c0, "d2c0": d2c0,
                "eff_tax_rate_0": eff_tax_rate_0, "beta0": beta0, "int_cov0": int_cov0,
                "g_analyst_y1": g_analyst_y1, "share_change_rate": share_change_rate,
                "industry_margin": industry_margin, "industry_s2c": industry_s2c,
                "industry_d2c": industry_d2c, "raw_industry_beta": raw_industry_beta,
                "terminal_margin": terminal_margin, "terminal_s2c": terminal_s2c,
                "terminal_d2c": terminal_d2c, "terminal_beta": terminal_beta
            }
            self._print_debug_info(tkr, debug_data)
        
        margin_path = self._project(margin0, terminal_margin, HIGH_GROWTH_YEARS, self.horizon)
        s2c_path = self._project(s2c0, terminal_s2c, HIGH_GROWTH_YEARS, self.horizon)
        tax_path = self._project(eff_tax_rate_0, STAT_TAX, HIGH_GROWTH_YEARS, self.horizon)
        d2c_path = self._project(d2c0, terminal_d2c, HIGH_GROWTH_YEARS, self.horizon)
        roic_path = [m * (1-t) * s for m, t, s in zip(margin_path, tax_path, s2c_path)]
        
        rr_y1 = roic_path[0] if roic_path[0] != 0 else 1.0
        rr_y1 = max(0, min(rr_y1, 1.2))

        g_terminal = mc_inputs.get('g_terminal', self.fw.RFR)
        roic_terminal = roic_path[-1]
        rr_terminal = g_terminal / roic_terminal if roic_terminal > 0 else 1.0
        rr_terminal = max(0, min(rr_terminal, 0.8))
        
        rr_path = self._project(rr_y1, rr_terminal, HIGH_GROWTH_YEARS, self.horizon)
        growth_path = [max(0, rr * roic) for rr, roic in zip(rr_path[0:], roic_path[0:])]
        
        years = np.arange(1, self.horizon + 1)
        df = pd.DataFrame(index=years)
        df['revenueGrowth'] = growth_path
        df['operatingMargin'] = margin_path
        df['taxRate'] = tax_path
        df['sales_to_capital'] = s2c_path
        df['ROIC'] = roic_path
        
        df['revenues'] = pd.Series(sales0 * (1 + df['revenueGrowth']).cumprod(), index=years)
        df['ebit'] = df['revenues'] * df['operatingMargin']
        df['ebit_after_tax'] = df['ebit'] * (1 - df['taxRate'])
        
        sales_change = df['revenues'].diff().fillna(df['revenues'].iloc[0] - sales0)
        df['reinvestment'] = (sales_change / df['sales_to_capital']).clip(lower=0)
        df['FCFF'] = df['ebit_after_tax'] - df['reinvestment']
        
        df['investedCapital'] = pd.Series(invested_capital0 + df['reinvestment'].cumsum(), index=years)

        cost_of_debt0 = self.fw.RFR + coverage_spread(int_cov0)
        cost_of_debt_terminal = self.fw.RFR + coverage_spread(6.5)

        df['beta'] = self._project(beta0, terminal_beta, HIGH_GROWTH_YEARS, self.horizon)
        df['costOfEquity'] = self.fw.RFR + (df['beta'] * self.fw.ERP)
        df['costOfDebt'] = self._project(cost_of_debt0, cost_of_debt_terminal, HIGH_GROWTH_YEARS, self.horizon)
        
        df['debt_to_capital'] = d2c_path
        df['debt'] = df['investedCapital'] * df['debt_to_capital']
        df['equity'] = df['investedCapital'] - df['debt']
        df['debt_weight'] = df['debt_to_capital']
        df['equity_weight'] = 1 - df['debt_to_capital']
        df['WACC'] = (df['equity_weight'] * df['costOfEquity']) + (df['debt_weight'] * df['costOfDebt'] * (1 - STAT_TAX))

        df['discountFactor'] = 1 / (1 + df['WACC']).cumprod()
        df['PV_FCFF'] = df['FCFF'] * df['discountFactor']
        
        wacc_terminal = df['WACC'].iloc[-1]
        
        capitalization_rate = max(0.02, wacc_terminal - g_terminal)
        
        fcff_terminal = df['ebit_after_tax'].iloc[-1] * (1 + g_terminal) * (1 - rr_terminal)
        tv = fcff_terminal / capitalization_rate
        pv_terminal = tv * df['discountFactor'].iloc[-1]

        final_debt = df['debt'].iloc[-1]
        value_of_operating_assets = df['PV_FCFF'].sum() + pv_terminal
        firm_value = value_of_operating_assets + cash0
        equity_value_b = firm_value - final_debt
        
        final_shares = shares0 * ((1 + share_change_rate) ** self.horizon)
        intrinsic_value = (equity_value_b * BILLION) / final_shares if final_shares > 0 else 0

        summary = {
            "Assumptions": {
                "Terminal Growth": g_terminal, "Terminal WACC": wacc_terminal,
                "Terminal ROIC": roic_terminal, "Current Margin": margin0,
                "Terminal Margin": terminal_margin, "Current S2C": s2c0,
                "Terminal S2C": terminal_s2c, "Current D2C": d2c0,
                "Terminal D2C": terminal_d2c, "Industry Beta": terminal_beta
            },
            "Valuation": {"Intrinsic Value": intrinsic_value, "Market Price": q.get('price', 0)}
        }
        data_to_cache = cached_data if cached_data else {'d': d, 'peer_medians': peer_medians}
        return intrinsic_value, q.get('price', 0), df, summary, data_to_cache

# ---------------------------------------------------------------------------
class MonteCarlo:
    def __init__(self, dcf_instance: DCF):
        self.dcf = dcf_instance

    def run(self, ticker, base_summary, cached_data):
        print(f"\nRunning {MC_SIMULATIONS} Monte Carlo simulations...")
        sim_results = []
        g_dist = np.random.normal(loc=self.dcf.fw.RFR, scale=0.005, size=MC_SIMULATIONS)
        current_m, terminal_m = base_summary['Assumptions']['Current Margin'], base_summary['Assumptions']['Terminal Margin']
        mode = terminal_m
        point1 = current_m
        point2 = terminal_m * 0.8
        point3 = terminal_m * 1.2
        left = min(point1, point2, point3, mode)
        right = max(point1, point2, point3, mode)

        if np.isclose(left, right):
            spread = 0.0001
            left -= spread
            right += spread
        
        margin_dist = np.random.triangular(left=left, mode=mode, right=right, size=MC_SIMULATIONS)

        for i in range(MC_SIMULATIONS):
            try:
                mc_inputs = {'g_terminal': g_dist[i], 'margin_terminal': margin_dist[i]}
                intrinsic_val, _, _, _, _ = self.dcf.value(ticker, mc_inputs, cached_data=cached_data)
                if intrinsic_val > 0: sim_results.append(intrinsic_val)
            except Exception:
                continue
        if not sim_results:
            print("Monte Carlo simulation failed to produce valid results.")
            return None
        return pd.DataFrame({"per_share": sim_results})

# ---------------------------------------------------------------------------
def process_ticker(tk, fw, dcf):
    """
    Processes a single ticker: DCF valuation, Monte Carlo, and output generation.
    Now includes a check to exclude Financial Services companies.
    """
    tk = tk.upper().strip()
    print(f"\n{'='*50}\nProcessing Ticker: {tk}\n{'='*50}")

    profile = fw.api.prof(tk)
    if not profile:
        print(f"⚠️ Could not fetch profile for {tk}. Skipping.")
        return None

    sector = profile.get('sector')
    if sector == 'Financial Services':
        print(f"⚠️ SKIPPING: {tk} is in the 'Financial Services' sector.")
        print("   Standard FCFF DCF models are not suitable for financial firms.")
        return "SKIPPED"

    base_intrinsic, mkt, df_yearly, summary, cached_data = dcf.value(tk)
    mc = MonteCarlo(dcf)
    sims = mc.run(tk, summary, cached_data)

    final_intrinsic_value = base_intrinsic
    if sims is not None and not sims.empty:
        final_intrinsic_value = sims['per_share'].median()
    summary['Valuation']['Intrinsic Value'] = final_intrinsic_value
    
    print("\n--- Valuation Assumptions ---")
    assumptions = summary['Assumptions']
    print(f"Risk-Free Rate: {dcf.fw.RFR:.2%}, Equity Risk Premium: {dcf.fw.ERP:.2%}")
    print(f"Convergence Horizon: {dcf.horizon} years ({HIGH_GROWTH_YEARS} high-growth + {CONVERGENCE_YEARS} transition)")
    print(f"Operating Margin: Converging from {assumptions['Current Margin']:.2%} to {assumptions['Terminal Margin']:.2%}")
    print(f"Sales-to-Capital: Converging from {assumptions['Current S2C']:.2f} to {assumptions['Terminal S2C']:.2f}")
    print(f"Debt-to-Capital: Converging from {assumptions['Current D2C']:.2%} to {assumptions['Terminal D2C']:.2%}")
    print(f"Beta: Converging from {df_yearly['beta'].iloc[0]:.2f} to {assumptions['Industry Beta']:.2f}")
    print(f"Terminal Growth: {assumptions['Terminal Growth']:.2%}, Terminal WACC: {assumptions['Terminal WACC']:.2%}, Terminal ROIC: {assumptions['Terminal ROIC']:.2%}")

    print("\n--- Valuation Summary ---")
    val_summary = summary['Valuation']
    upside = (val_summary['Intrinsic Value'] / val_summary['Market Price'] - 1) * 100 if val_summary['Market Price'] > 0 else 0
    s = "↑" if upside >= 0 else "↓"
    print(f"Market price : ${val_summary['Market Price']:,.2f}")
    print(f"Intrinsic Value (MC Avg): ${val_summary['Intrinsic Value']:,.2f}  ({s}{abs(upside):.1f}%)")

    cols_in_billions = ['revenues', 'ebit', 'ebit_after_tax', 'reinvestment', 'FCFF', 'investedCapital', 'debt', 'equity']
    df_yearly.rename(columns={c: f"{c}_B" for c in cols_in_billions}, inplace=True)

    output_filename = f"yearly_{tk.replace('.', '_')}.csv"
    df_yearly.reset_index().rename(columns={'index': 'Year'}).to_csv(output_filename, index=False, float_format='%.4f')
    print(f"\nYear-by-year table (in billions) saved to {output_filename}")

    if sims is not None:
        print("\n--- Monte Carlo Results ---")
        print(sims['per_share'].describe(percentiles=[0.1, 0.25, 0.5, 0.75, 0.9]).round(2))
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=sims["per_share"], nbinsx=70, name="Simulated Values", histnorm='probability density'))
        fig.add_vline(x=final_intrinsic_value, line_width=2, line_dash="dash", line_color="green", annotation_text="MC Avg. Value")
        fig.add_vline(x=base_intrinsic, line_width=1.5, line_dash="dot", line_color="yellow", annotation_text="Base Case Value")
        fig.add_vline(x=val_summary['Market Price'], line_width=2, line_dash="dash", line_color="red", annotation_text="Market Price")
        fig.update_layout(title_text=f"{tk} Monte Carlo Simulation of Intrinsic Value", xaxis_title="Value per Share ($)", yaxis_title="Probability Density", showlegend=False)
        fig.show()

    print(f"\nSuccessfully processed {tk}.")
    return upside

# ---------------------------------------------------------------------------
def main():
    api = FMP(FMP_API_KEY)
    fw = Framework(api)
    dcf = DCF(fw)
    results_filename = 'results.csv'
    
    try:
        ticker_df = pd.read_csv('beta.csv')
        tickers = ticker_df['Symbol'].drop_duplicates().tolist()
        print(f"Loaded {len(tickers)} unique tickers from beta.csv")
    except FileNotFoundError:
        print("Warning: 'beta.csv' not found. You can only process a single ticker.")
        tickers = []
    
    user_choice = input("Enter 'all' to process all tickers from beta.csv, or enter a single ticker symbol: ").strip()
    tickers_to_process = []
    if user_choice.lower() == 'all':
        if not tickers:
            print("Cannot process 'all' because 'beta.csv' was not found or is empty.")
            return
        tickers_to_process = tickers
        # For a full batch run, delete the old results file to start fresh
        if os.path.exists(results_filename):
            os.remove(results_filename)
            print(f"Removed old '{results_filename}' for a fresh batch run.")
    elif user_choice:
        tickers_to_process.append(user_choice)
    else:
        print("No ticker provided. Exiting.")
        return
        
    for i, ticker in enumerate(tickers_to_process):
        try:
            print(f"\n--- Processing ticker {i+1} of {len(tickers_to_process)} ---")
            result = process_ticker(ticker, fw, dcf)
            
            if result == "SKIPPED":
                result_df = pd.DataFrame([{'Ticker': ticker, 'Upside Potential %': 'SKIPPED (Financial Sector)'}])
            elif isinstance(result, (int, float)):
                upside = result
                result_df = pd.DataFrame([{'Ticker': ticker, 'Upside Potential %': f"{upside:.2f}%"}])
            else: # Handles None or other errors
                print(f"Could not process {ticker}, no result to save.")
                continue

            # Save the result
            write_header = not os.path.exists(results_filename)
            result_df.to_csv(results_filename, mode='a', header=write_header, index=False)
            print(f"✅ Result for {ticker} saved to {results_filename}")

            if len(tickers_to_process) > 1 and i < len(tickers_to_process) - 1:
                print("\nWaiting 5 seconds before next request...")
                time.sleep(5)
        except Exception as e:
            print(f"\n--- ERROR processing {ticker} ---")
            print(f"Error: {e}")
            traceback.print_exc()
            print("--- END ERROR ---")
            if len(tickers_to_process) > 1:
                print("Skipping to next ticker. Waiting 10 seconds...")
                time.sleep(10)
    
    if tickers_to_process:
        print(f"\n✅ All processing complete. Final results are in {results_filename}.")
    else:
        print("\nNo tickers were processed.")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nA fatal script error occurred: {exc}")
        traceback.print_exc()
