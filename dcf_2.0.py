#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Pure FMP multi‑phase DCF with Monte‑Carlo & full year‑by‑year table
------------------------------------------------------------------
User input  : 'all' for batch processing or a single Ticker symbol.
File output : yearly_<ticker>.csv  and interactive histograms.
Console     : intrinsic value + upside / downside vs. market.

v4.7 – 2024-07-30 (Updated with batch processing)
• Integrates batch processing from a CSV file.
• Refactored main logic for single and batch runs.
• Added timed delays and robust error handling for batch jobs.
"""

import warnings, requests, math, sys, io, pandas as pd, numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import re
import time
import traceback

# ---------------- console UTF‑8 fallback ------------------------------------
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except AttributeError:
    if hasattr(sys.stdout, "buffer"):
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer,
                                      encoding="utf-8",
                                      errors="replace",
                                      line_buffering=True)
warnings.filterwarnings("ignore")

# ---------------- CONFIG ----------------------------------------------------
FMP_API_KEY = "435820a51119704ed53f7e7fb8a0cfec"
FMP_BASE = "https://financialmodelingprep.com/api/v3"
HEADERS = {"User-Agent": "DCF-FMP/4.7"}

HORIZON = 10
STAT_TAX = 0.21

# ---------------------------------------------------------------------------
def coverage_spread(ic):
    table = [(8.5, 0.006), (6.5, 0.008), (5, 0.010), (4, 0.012), (3, 0.015), (2.5, 0.020),
             (2, 0.025), (1.5, 0.035), (1.25, 0.045), (0.8, 0.060), (0.5, 0.085), (0.2, 0.10)]
    for cut, sp in table:
        if ic >= cut:
            return sp
    return 0.120

def cum(series):
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
            r = self.s.get(f"{FMP_BASE}/{ep}", params=p, timeout=15)
            r.raise_for_status()
            return r.json()
        except requests.RequestException as e:
            print(f"API error ({ep}): {str(e)}")
            return None

    def quote(self, t):
        return (self._j(f"quote/{t}") or [{}])[0]

    def prof(self, t):
        return (self._j(f"profile/{t}") or [{}])[0]

    def inc(self, t):
        return self._j(f"income-statement/{t}", period="annual") or []

    def cf(self, t):
        return self._j(f"cash-flow-statement/{t}", period="annual") or []

    def bs(self, t):
        return self._j(f"balance-sheet-statement/{t}", period="annual") or []

    def est(self, t):
        return self._j(f"analyst-estimates/{t}")

    def treas(self):
        return self._j("treasury")

    def spy(self):
        return self._j("historical-price-full/SPY", serietype="line")

    def peers(self, t):
        sec = self.prof(t).get("sector", "Technology")
        return self._j("stock-screener", sector=sec, limit=100) or []

    def fx(self, pair):
        return self._j(f"fx/{pair}")

# ---------------------------------------------------------------------------
# Market‑level helpers  (unified FX)
# ---------------------------------------------------------------------------
class Framework:
    def __init__(self, api: FMP):
        self.api = api
        self.RFR = self._rf()
        self.ERP = self._erp()
        self._fx_cache = {}
        self._fx_last_update = {}

    def _rf(self):
        t = self.api.treas()
        return t[0]["year10"] / 100 if t and t[0].get("year10") else 0.028

    def _erp(self):
        h = self.api.spy()
        if not h or not h.get("historical"):
            return 0.055
        p = pd.DataFrame(h["historical"])
        if len(p) < 2:
            return 0.055
        yrs = max(1, len(p) / 252)
        start = p.iloc[0]["close"]
        end = p.iloc[-1]["close"]
        if start > 0 and end > 0:
            return (start / end) ** (1 / yrs) - 1 - self.RFR
        return 0.055

    def data(self, tkr):
        return dict(
            quote=self.api.quote(tkr),
            prof=self.api.prof(tkr),
            inc=self.api.inc(tkr),
            cf=self.api.cf(tkr),
            bs=self.api.bs(tkr),
            est=self.api.est(tkr),
            peers=self.api.peers(tkr)
        )

    # ------------------------------------------------------------------ #
    #                             UNIVERSAL FX  → USD                    #
    # ------------------------------------------------------------------ #
    def _fx_usd(self, cur: str) -> float:
        """Return 1 unit of `cur` in USD using multiple real-time sources"""
        cur = cur.upper()
        if cur == "USD":
            return 1.0

        # Check if we have a recent cached rate (<60 minutes)
        now = datetime.now()
        if cur in self._fx_cache and cur in self._fx_last_update:
            if (now - self._fx_last_update[cur]).seconds < 3600:
                return self._fx_cache[cur]

        sources = [
            self._get_fx_from_exchangerate,
            self._get_fx_from_fmp,
            self._get_fx_from_yahoo,
            self._get_fx_from_ecb,
            self._get_fx_from_xe
        ]

        rates = []
        for source in sources:
            try:
                rate = source(cur)
                if rate and rate > 0:
                    rates.append(rate)
                    print(f"FX source: {source.__name__} returned {cur}/USD = {rate:.6f}")
            except Exception as e:
                print(f"FX source error: {str(e)}")

        if not rates:
            raise ValueError(f"All FX sources failed for {cur}/USD")

        # Use median rate to avoid outliers
        median_rate = np.median(rates)
        print(f"Using median FX rate: {cur}/USD = {median_rate:.6f}")
        self._fx_cache[cur] = median_rate
        self._fx_last_update[cur] = now
        return median_rate

    def _get_fx_from_exchangerate(self, cur):
        """Source 1: exchangerate.host"""
        url = f"https://api.exchangerate.host/latest?base={cur}&symbols=USD"
        r = requests.get(url, timeout=10, headers=HEADERS)
        r.raise_for_status()
        data = r.json()
        if data.get("success") and data.get("rates", {}).get("USD"):
            rate = float(data["rates"]["USD"])
            if rate > 0:
                return rate
        return None

    def _get_fx_from_fmp(self, cur):
        """Source 2: Financial Modeling Prep"""
        data = self.api.fx(f"{cur}USD")
        if data and isinstance(data, list) and len(data) > 0:
            bid = data[0].get("bid")
            ask = data[0].get("ask")
            if bid and ask:
                rate = (float(bid) + float(ask)) / 2
                if rate > 0:
                    return rate
            elif bid:
                rate = float(bid)
                if rate > 0:
                    return rate
        
        data_inv = self.api.fx(f"USD{cur}")
        if data_inv and isinstance(data_inv, list) and len(data_inv) > 0:
            bid = data_inv[0].get("bid")
            ask = data_inv[0].get("ask")
            if bid and ask:
                inv_rate = (float(bid) + float(ask)) / 2
                if inv_rate > 0:
                    return 1 / inv_rate
            elif bid:
                inv_rate = float(bid)
                if inv_rate > 0:
                    return 1 / inv_rate
        return None

    def _get_fx_from_yahoo(self, cur):
        """Source 3: Yahoo Finance"""
        def fetch_yahoo(sym):
            url = f"https://query1.finance.yahoo.com/v7/finance/quote?symbols={sym}"
            r = requests.get(url, timeout=10, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            result = data.get("quoteResponse", {}).get("result", [])
            if result and result[0].get("regularMarketPrice") is not None:
                return float(result[0]["regularMarketPrice"])
            return None

        pairs = [
            (f"{cur}USD=X", False),
            (f"{cur}=X", False),
            (f"USD{cur}=X", True)
        ]

        for sym, invert in pairs:
            price = fetch_yahoo(sym)
            if price and price > 0:
                return 1 / price if invert else price
        return None

    def _get_fx_from_ecb(self, cur):
        """Source 4: European Central Bank (for EUR pairs)"""
        if cur == "EUR":
            url = "https://api.exchangerate.host/latest?base=EUR&symbols=USD"
            r = requests.get(url, timeout=10, headers=HEADERS)
            r.raise_for_status()
            data = r.json()
            if data.get("success") and data.get("rates", {}).get("USD"):
                return float(data["rates"]["USD"])
        return None

    def _get_fx_from_xe(self, cur):
        """Source 5: XE.com (fallback)"""
        if len(cur) != 3:
            return None
        
        url = f"https://www.xe.com/currencyconverter/convert/?Amount=1&From={cur}&To=USD"
        r = requests.get(url, timeout=10, headers=HEADERS)
        r.raise_for_status()
        
        pattern = r'<p class="result__BigRate-sc-1bsijpp-1 iGrAod">([\d,.]+)\s*USD</p>'
        match = re.search(pattern, r.text)
        if match:
            try:
                rate_text = match.group(1).replace(',', '')
                rate = float(rate_text)
                if rate > 0:
                    return rate
            except ValueError:
                pass
        return None

    def usd_factor(self, prof: dict, statement: list):
        currency = statement[0].get("reportedCurrency", "USD") or "USD"
        
        try:
            return self._fx_usd(currency.upper())
        except Exception as e:
            print(f"Warning: Using fallback FX for {currency}: {str(e)}")
            fallback_rates = {
                "JPY": 0.0091, "GBP": 1.25, "EUR": 1.08, "CAD": 0.73, 
                "AUD": 0.66, "CHF": 1.09, "CNY": 0.14, "INR": 0.012, 
                "MXN": 0.059
            }
            return fallback_rates.get(currency.upper(), 1.0)

# ---------------------------------------------------------------------------
# ----------------------------  DCF ENGINE  ----------------------------------
# ---------------------------------------------------------------------------
class DCF:
    def __init__(self, fw: Framework):
        self.fw = fw
        self.sigma = 0.20

    @staticmethod
    def _beta_u(beta_l, de):
        return beta_l / (1 + (1 - STAT_TAX) * de) if de != -1 else beta_l

    @staticmethod
    def _beta_l(beta_u, de):
        return beta_u * (1 + (1 - STAT_TAX) * de) if de != -1 else beta_u

    def _validate_financials(self, inc, cf, bs):
        if not inc or not cf or not bs:
            raise ValueError("Missing financial statements")
        
        if not inc[0].get("revenue") or not inc[0].get("operatingIncome"):
            raise ValueError("Incomplete income statement")
        
        if inc[0]["revenue"] <= 0:
            raise ValueError("Invalid revenue value")

    def _derived_ratios(self, inc, cf, bs):
        n = min(3, len(inc), len(cf), len(bs))
        rev = np.array([inc[i]["revenue"] for i in range(n)], float)
        cap = np.abs([cf[i].get("capitalExpenditure", 0) for i in range(n)])
        da = [cf[i].get("depreciationAndAmortization", 0) for i in range(n)]
        wc = [bs[i].get("workingCapital", 0) for i in range(n)]
        
        capx_ratio = np.mean(cap / rev) if rev.any() and rev.mean() > 0 else 0.06
        da_ratio = np.mean(np.array(da) / rev) if rev.any() and rev.mean() > 0 else 0.04
        wc_ratio = wc[0] / rev[0] if rev[0] and rev[0] > 0 else 0.10
        
        return capx_ratio, da_ratio, wc_ratio

    def _growth_path(self, inc, est):
        if not inc:
            return [0.05] * HORIZON
        
        if est:
            cur = inc[0]["revenue"]
            g = []
            for e in est[:HORIZON]:
                nxt = e.get("estimatedRevenueAvg", cur * 1.05) if cur else 0
                growth = (nxt - cur) / cur if cur and cur > 0 else 0.05
                g.append(max(min(growth, 2.0), -0.5))
                cur = nxt
            while len(g) < HORIZON:
                g.append(g[-1] * 0.5 if g else 0.05)
            return g
        
        if len(inc) < 2:
            return [0.05] * HORIZON
        
        r0, rn = inc[0]["revenue"], inc[-1]["revenue"]
        if not r0 or not rn or rn <= 0:
            return [0.05] * HORIZON
        
        cagr = (r0 / rn) ** (1 / len(inc)) - 1
        cagr = max(min(cagr, 2.0), -0.5)
        return [cagr] * HORIZON

    def _share_cagr(self, inc):
        if not inc or len(inc) < 2:
            return 0.0
        
        n = min(4, len(inc))
        dil = [inc[i].get("weightedAverageShsOutDil", 0) for i in range(n)]
        if dil[0] <= 0 or dil[-1] <= 0:
            return 0.0
        
        return (dil[0] / dil[-1]) ** (1 / (n - 1)) - 1

    def value(self, tkr):
        d = self.fw.data(tkr)
        if not d["quote"] or "price" not in d["quote"]:
            raise ValueError("Missing quote data")

        try:
            self._validate_financials(d["inc"], d["cf"], d["bs"])
            usd_fx = self.fw.usd_factor(d["prof"], d["inc"])
            print(f"Using FX rate: {usd_fx:.6f}")
            def fx(x): return x * usd_fx if x is not None and not np.isnan(x) else 0
        except Exception as e:
            raise ValueError(f"Data validation error: {str(e)}")

        q, inc, cf, bs, est, peers = (d["quote"], d["inc"], d["cf"], d["bs"], d["est"], d["peers"])

        shares0 = q.get("sharesOutstanding", 0)
        debt0 = fx(bs[0].get("totalDebt", 0)) if bs else 0
        cash0 = fx(bs[0].get("cashAndCashEquivalents", 0)) if bs else 0
        equity0 = fx(bs[0].get("totalStockholdersEquity", 0)) if bs else 0
        invested0 = max(0, equity0 + debt0 - cash0)
        sales0 = fx(inc[0].get("revenue", 0))
        EBIT0 = fx(inc[0].get("operatingIncome", 0))
        margin0 = EBIT0 / sales0 if sales0 > 0 else 0.10
        beta_l0 = float(d["prof"].get("beta", 1.0))
        de0 = debt0 / equity0 if equity0 > 0 else 0
        beta_u0 = self._beta_u(beta_l0, de0)
        capx_r, da_r, wc_r = self._derived_ratios(inc, cf, bs)
        growth = self._growth_path(inc, est)
        interest = abs(fx(cf[0].get("interestExpense", 0))) if cf else 0
        int_cov = EBIT0 / interest if interest > 0 else 15
        kd = self.fw.RFR + coverage_spread(int_cov)
        kd_at = kd * (1 - STAT_TAX)
        share_rate = self._share_cagr(inc)
        share_path = (1 + share_rate) ** np.arange(HORIZON + 1)

        if peers:
            ps = [p.get("priceToSalesRatio", 0) or p.get("peRatio", 0) for p in peers]
            ps = [v for v in ps if v and v > 0]
            if ps:
                self.sigma = min(0.5, max(0.1, np.std(ps) / np.mean(ps)))
                print(f"Using peer-based sigma: {self.sigma:.4f}")

        sales, ic, ebit, capex_l, daw_l, delta_wc_l, reinvest_l, ebit_at, fcff, beta_l, ceq_l, wacc_l, s2c, roic, reinv_rate = (
            [sales0], [invested0], [], [], [], [], [], [], [], [], [], [], [], [], []
        )

        for yr in range(1, HORIZON + 1):
            growth_rate = growth[yr - 1] if yr - 1 < len(growth) else growth[-1]
            sales.append(max(0, sales[-1] * (1 + growth_rate)))
            m = margin0 + (yr / HORIZON) * (0.10 - margin0)
            e = sales[-1] * m
            ebit.append(e)
            prev_sales = sales[-2]
            capex = capx_r * prev_sales
            da = da_r * prev_sales
            d_wc = wc_r * (sales[-1] - prev_sales)
            reinvest = capex + d_wc
            invested = max(0, ic[-1] + reinvest)
            at_ebit = e * (1 - STAT_TAX)
            fc = at_ebit + da - capex - d_wc
            equity_val = max(1, equity0 * (1 + growth_rate) ** yr)
            beta_l_now = self._beta_l(beta_u0, de0)
            ceq = self.fw.RFR + beta_l_now * self.fw.ERP
            wacc = (equity_val / (equity_val + debt0)) * ceq + (debt0 / (equity_val + debt0)) * kd_at

            capex_l.append(capex); daw_l.append(da); delta_wc_l.append(d_wc); reinvest_l.append(reinvest)
            ic.append(invested); ebit_at.append(at_ebit); fcff.append(fc); beta_l.append(beta_l_now)
            ceq_l.append(ceq); wacc_l.append(wacc);
            s2c.append(sales[-1] / invested if invested > 0 else np.nan)
            roic.append(at_ebit / invested if invested > 0 else np.nan)
            reinv_rate.append(reinvest / at_ebit if at_ebit > 0 else np.nan)

        wacc_series = pd.Series(wacc_l, index=range(1, HORIZON + 1))
        ceq_series = pd.Series(ceq_l, index=range(1, HORIZON + 1))
        disc_wacc = cum(wacc_series)
        disc_ceq = cum(ceq_series)
        fcff_series = pd.Series(fcff, index=range(1, HORIZON + 1))
        pv_fcff = (fcff_series / disc_wacc).sum()

        gT = max(min(growth[-1], 0.05), -0.02)
        waccT = wacc_l[-1]
        roicT = max(waccT + 0.02, 0.05)
        ebit_T = ebit[-1] * (1 + gT)
        ebit_AT = ebit_T * (1 - STAT_TAX)
        fcffT = ebit_AT * (1 - gT / roicT)
        term_val = fcffT / (waccT - gT) if waccT > gT else fcffT * 10
        pv_term = term_val / disc_wacc.iloc[-1]

        firm_val = pv_fcff + pv_term + cash0
        equity_val = max(0, firm_val - debt0)
        intrinsic = equity_val / (shares0 * share_path[-1]) if shares0 > 0 else 0
        mkt_price = q["price"]

        df = pd.DataFrame({
            "Year": [*range(1, HORIZON + 1)], "cumWACC": disc_wacc.values, "cumCostOfEquity": disc_ceq.values,
            "beta": beta_l, "ERP": [self.fw.ERP] * HORIZON, "projected_after_tax_cost_of_debt": [kd_at] * HORIZON,
            "revenueGrowth": growth[:HORIZON], "revenues": sales[1:],
            "margins": [margin0 + (i / HORIZON) * (0.10 - margin0) for i in range(1, HORIZON + 1)],
            "ebit": ebit, "sales_to_capital_ratio": s2c, "taxRate": [STAT_TAX] * HORIZON,
            "afterTaxOperatingIncome": ebit_at, "reinvestment": reinvest_l, "invested_capital": ic[1:],
            "ROIC": roic, "reinvestmentRate": reinv_rate, "FCFF": fcff, "projected_FCFF_value": fcff,
            "PVFCFF": fcff_series / disc_wacc
        })

        terminal = pd.Series({
            "Year": "Terminal", "cumWACC": disc_wacc.iloc[-1], "cumCostOfEquity": disc_ceq.iloc[-1],
            "beta": beta_l[-1], "ERP": self.fw.ERP, "projected_after_tax_cost_of_debt": kd_at,
            "revenueGrowth": gT, "revenues": sales[-1] * (1 + gT), "margins": df["margins"].iloc[-1],
            "ebit": ebit_T, "sales_to_capital_ratio": s2c[-1], "taxRate": STAT_TAX,
            "afterTaxOperatingIncome": ebit_AT, "reinvestment": gT / roicT * ebit_AT,
            "invested_capital": np.nan, "ROIC": roicT, "reinvestmentRate": gT / roicT, "FCFF": fcffT,
            "projected_FCFF_value": term_val, "PVFCFF": pv_term,
        })
        df = pd.concat([df, terminal.to_frame().T], ignore_index=True)

        return intrinsic, mkt_price, df

# ---------------------------------------------------------------------------
class MonteCarlo:
    def __init__(self, dcf: DCF):
        self.dcf = dcf

    def run(self, ticker, base_intrinsic):
        if base_intrinsic is None or base_intrinsic <= 0:
            return None
        mu = math.log(base_intrinsic) - 0.5 * self.dcf.sigma ** 2
        sims = np.random.lognormal(mu, self.dcf.sigma, 1000)
        shares_data = self.dcf.fw.data(ticker)
        shares = shares_data["quote"].get("sharesOutstanding", 0)
        if shares <= 0:
            return None
        return pd.DataFrame({"per_share": sims, "equity_value": sims * shares})

# ---------------------------------------------------------------------------
def process_ticker(tk, api, fw, dcf):
    """
    Processes a single ticker: performs DCF valuation, saves results,
    and runs Monte Carlo simulation.
    """
    tk = tk.upper().strip()
    print(f"\n{'='*40}\nProcessing Ticker: {tk}\n{'='*40}")

    intrinsic, mkt, df = dcf.value(tk)

    if intrinsic is None or mkt is None:
        raise ValueError("Valuation failed - insufficient data from API.")
    if mkt <= 0:
        raise ValueError("Invalid market price received from API.")

    up_pct = (intrinsic / mkt - 1) * 100
    s = "↑" if up_pct >= 0 else "↓"
    print(f"\nMarket price : ${mkt:,.4f}")
    print(f"Intrinsic    : ${intrinsic:,.4f}  ({s}{abs(up_pct):.1f} %)\n")

    output_filename = f"yearly_{tk}.csv"
    df.to_csv(output_filename, index=False)
    print(f"Year-by-year table saved to {output_filename}")

    mc = MonteCarlo(dcf)
    sims = mc.run(tk, intrinsic)
    if sims is not None:
        print("\nMonte Carlo Simulation Results:")
        print(sims.describe())
        fig = make_subplots(rows=1, cols=2, subplot_titles=("Equity Value", "Per Share"))
        fig.add_trace(go.Histogram(x=sims["equity_value"], nbinsx=50, name="Equity Value"), 1, 1)
        fig.add_trace(go.Histogram(x=sims["per_share"], nbinsx=50, name="Per Share Value"), 1, 2)
        fig.update_layout(title_text=f"{tk} Monte Carlo Simulation", showlegend=False)
        fig.show()
    
    print(f"\nSuccessfully processed {tk}.")

# ---------------------------------------------------------------------------
def main():
    # Initialize API and frameworks once
    api = FMP(FMP_API_KEY)
    fw = Framework(api)
    dcf = DCF(fw)

    try:
        ticker_df = pd.read_csv('beta.csv')
        tickers = ticker_df['Symbol'].drop_duplicates().tolist()
        print(f"Loaded {len(tickers)} unique tickers from beta.csv")
    except FileNotFoundError:
        print("Warning: 'beta.csv' not found. You can only process a single ticker.")
        tickers = []

    user_choice = input("Enter 'all' to process all stocks, or enter a Ticker symbol: ").strip()

    if user_choice.lower() == 'all':
        if not tickers:
            print("Cannot process 'all' because 'beta.csv' was not found or is empty.")
            return
        
        # Iterate over each ticker and call the function for all stocks
        for ticker in tickers:
            try:
                process_ticker(ticker, api, fw, dcf)
                print("Waiting 30 seconds before next request...")
                time.sleep(30)  # Normal delay between successful requests
            except Exception as e:
                print(f"\n--- ERROR ---")
                print(f"An error occurred while processing {ticker}: {e}")
                traceback.print_exc()
                print(f"--- END ERROR ---")
                print(f"Waiting 60 seconds before continuing...")
                time.sleep(60) # Longer delay after an error
    else:
        # Process only the specified ticker
        try:
            process_ticker(user_choice, api, fw, dcf)
        except Exception as e:
            print(f"\n--- ERROR ---")
            print(f"An error occurred while processing {user_choice.upper()}: {e}")
            traceback.print_exc()
            print(f"--- END ERROR ---")

# ---------------------------------------------------------------------------
if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nFatal script error: {exc}")
        traceback.print_exc()
