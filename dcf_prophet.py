### import packages
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import pandas as pd
import numpy as np
import datetime as dt
from datetime import datetime, timedelta
import scipy.stats
import sys
import openturns as ot
from IPython.display import display
import requests
from datetime import datetime
import yfinance as yf
import io
from yahooquery import Ticker, Screener
from sklearn.linear_model import LinearRegression
from bs4 import BeautifulSoup
import re
import json
from yahoo_fin import stock_info as si
import math
from reportlab.lib.pagesizes import letter, landscape
from reportlab.platypus import SimpleDocTemplate, Table
import pdfkit
import cmath
import os
import warnings
import traceback
import time
warnings.filterwarnings("ignore")


get_requests_count = 0
start_time = time.time()
# Ignore future warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

pd.options.display.max_columns=1000
pd.options.display.max_rows= 200
pd.options.display.float_format = '{:,.3f}'.format

# Adjust scroll-in-the-scroll in the entire Notebook
from IPython.display import Javascript

def data_frame_flattener(df_data):
    df=df_data.copy()
    try:
        df.columns=[' '.join(map(str,col)).strip() for col in df.columns.values]
    except:
        pass
    return(df)

def column_suffix_adder(df_data,
                        list_of_columns_to_add_suffix_on,
                        suffix):
    """Add specific siffix to specific columns"""
    df=df_data.copy()
    ### Add suffix or prefix to certain columns rename all columns
    new_names = [(i,i+suffix) for i in df[list_of_columns_to_add_suffix_on].columns.values]
    df.rename(columns = dict(new_names), inplace=True)
    return(df)

"""### Valuation functions

"""

def dynamic_converger(current,
                      expected,
                      number_of_steps,
                      period_to_begin_to_converge):
    """This Function is to project growth in 2 phase.
    Phase 1: You grow  Period after period for number of period specified.
    Phase 2: growth begin to converge to number_of_steps value.
    current: begining growth_rate
    expected: final growth rate
    period_to_begin_to_converge: Period to begin to transition to terminal growth value
    number_of_steps: number of period (years) to project growth."""
    number_of_steps =  int(number_of_steps)
    period_to_begin_to_converge = int(period_to_begin_to_converge)
    def converger(current,
                expected,
                number_of_steps):
        values = np.linspace(current,expected,number_of_steps+1)
        return(values)

    array_phase1 = np.array([current]*(period_to_begin_to_converge-1))

    array_phase2 = converger(current,
                       expected,
                       number_of_steps-period_to_begin_to_converge)
    result= pd.Series(np.concatenate((array_phase1,array_phase2)))
    return(result)

def dynamic_converger_multiple_phase(growth_rates_for_each_cylce,
                                     length_of_each_cylce,
                                     convergance_periods):
    list_of_results = []
    for cycle in range(len(length_of_each_cylce)):
        result = dynamic_converger(current = growth_rates_for_each_cylce[cycle][0],
                        expected = growth_rates_for_each_cylce[cycle][1],
                        number_of_steps = length_of_each_cylce[cycle],
                        period_to_begin_to_converge = convergance_periods[cycle])
        list_of_results.append(result)
    return(pd.concat(list_of_results,ignore_index=True))


def revenue_projector_multi_phase(revenue_base,
                                  revenue_growth_rate_cycle1_begin,
                                  revenue_growth_rate_cycle1_end,
                                  revenue_growth_rate_cycle2_begin,
                                  revenue_growth_rate_cycle2_end,
                                  revenue_growth_rate_cycle3_begin,
                                  revenue_growth_rate_cycle3_end = 0.028,
                                  length_of_cylcle1=3,
                                  length_of_cylcle2=4,
                                  length_of_cylcle3=3,
                                  revenue_convergance_periods_cycle1 =1,
                                  revenue_convergance_periods_cycle2=1,
                                  revenue_convergance_periods_cycle3=1):
    projected_revenue_growth = dynamic_converger_multiple_phase(growth_rates_for_each_cylce= [[revenue_growth_rate_cycle1_begin,revenue_growth_rate_cycle1_end],
                                                               [revenue_growth_rate_cycle2_begin,revenue_growth_rate_cycle2_end],
                                                               [revenue_growth_rate_cycle3_begin,revenue_growth_rate_cycle3_end]],
                                     length_of_each_cylce=[length_of_cylcle1,length_of_cylcle2,length_of_cylcle3],
                                     convergance_periods=[revenue_convergance_periods_cycle1,
                                                          revenue_convergance_periods_cycle2,
                                                          revenue_convergance_periods_cycle3])
    ### Compute Cummulative revenue_growth
    projected_revenue_growth_cumulative = (1+projected_revenue_growth).cumprod()
    projected_revneues = revenue_base*projected_revenue_growth_cumulative
    return(projected_revneues,projected_revenue_growth)

def operating_margin_projector(current_operating_margin,
                               terminal_operating_margin,
                               valuation_interval_in_years=10,
                               year_operating_margin_begins_to_converge_to_terminal_operating_margin=5):
    projectd_operating_margin = dynamic_converger(current_operating_margin,
                                                  terminal_operating_margin,
                                                  valuation_interval_in_years,
                                                  year_operating_margin_begins_to_converge_to_terminal_operating_margin)
    return(projectd_operating_margin)

def tax_rate_projector(current_effective_tax_rate,
                      marginal_tax_rate,
                      valuation_interval_in_years=10,
                      year_effective_tax_rate_begin_to_converge_marginal_tax_rate=5):
    """Project tax rate during valuation Cylce"""
    projected_tax_rate = dynamic_converger(current_effective_tax_rate,
                                           marginal_tax_rate,
                                           valuation_interval_in_years,
                                           year_effective_tax_rate_begin_to_converge_marginal_tax_rate)
    return(projected_tax_rate)

def cost_of_capital_projector(unlevered_beta,
                              terminal_unlevered_beta,
                              current_pretax_cost_of_debt,
                              terminal_pretax_cost_of_debt,
                              equity_value,
                              debt_value,
                              marginal_tax_rate=.21,
                              risk_free_rate=0.015,
                              ERP=0.055,
                              valuation_interval_in_years=10,
                              year_beta_begins_to_converge_to_terminal_beta=5,
                              year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=5):
    """Project Cost of Capiatal during valuation Cylce"""
    ### Compute Beta During Valuatio Cycle
    ### Company Levered Beta  = Unlevered beta * (1 + (1- tax rate) (Debt/Equity))
    company_beta = unlevered_beta * (1+(1-marginal_tax_rate)*(debt_value/equity_value))
    terminal_beta = terminal_unlevered_beta * (1+(1-marginal_tax_rate)*(debt_value/equity_value))
    beta_druing_valution_cycle = dynamic_converger(company_beta,
                                                   terminal_beta,
                                                   valuation_interval_in_years,
                                                   year_beta_begins_to_converge_to_terminal_beta)
    ### Compute Pre Tax Cost Of debt During Valuation Cycle
    pre_tax_cost_of_debt_during_valution_cycle = dynamic_converger(current_pretax_cost_of_debt,
                                                                   terminal_pretax_cost_of_debt,
                                                                   valuation_interval_in_years,
                                                                   year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt)

    total_capital = equity_value+debt_value
    equity_to_capital = equity_value/total_capital
    debt_to_capital = debt_value/total_capital
    after_tax_cost_of_debt_during_valution_cycle = pre_tax_cost_of_debt_during_valution_cycle*(1-marginal_tax_rate)
    cost_of_equity = risk_free_rate + (beta_druing_valution_cycle*ERP)
    cost_of_capital_during_valuatio_cycle = ((equity_to_capital*cost_of_equity)+
                                             (debt_to_capital*after_tax_cost_of_debt_during_valution_cycle))
    return(cost_of_capital_during_valuatio_cycle,beta_druing_valution_cycle,terminal_beta,cost_of_equity,after_tax_cost_of_debt_during_valution_cycle)

def revenue_growth_projector(revenue_growth_rate,
                             terminal_growth_rate=.028,
                             valuation_interval_in_years=10,
                             year_revenue_growth_begin_to_converge_to_terminal_growth_rate = 5):
    """Project revenue growth during valuation Cylce"""
    projected_revenue_growth = dynamic_converger(revenue_growth_rate,
                                                 terminal_growth_rate,
                                                 valuation_interval_in_years,
                                                 year_revenue_growth_begin_to_converge_to_terminal_growth_rate)
    return(projected_revenue_growth)

def revenue_projector(revenue_base,
                      revenue_growth_rate,
                      terminal_growth_rate,
                      valuation_interval_in_years,
                      year_revenue_growth_begin_to_converge_to_terminal_growth_rate):
    ### Estimate Revenue Growth
    projected_revenue_growth = revenue_growth_projector(revenue_growth_rate=revenue_growth_rate,
                                                        terminal_growth_rate = terminal_growth_rate,
                                                        valuation_interval_in_years=valuation_interval_in_years,
                                                        year_revenue_growth_begin_to_converge_to_terminal_growth_rate=year_revenue_growth_begin_to_converge_to_terminal_growth_rate)
    ### Compute Cummulative revenue_growth
    projected_revenue_growth_cumulative = (1+projected_revenue_growth).cumprod()
    projected_revneues = revenue_base*projected_revenue_growth_cumulative
    return(projected_revneues,projected_revenue_growth)

def sales_to_capital_projector(current_sales_to_capital_ratio,
                               terminal_sales_to_capital_ratio,
                               valuation_interval_in_years=10,
                               year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital=3):
    projectd_sales_to_capiatl = dynamic_converger(current_sales_to_capital_ratio,
                                                  terminal_sales_to_capital_ratio,
                                                  valuation_interval_in_years,
                                                  year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital)
    return(projectd_sales_to_capiatl)

def reinvestment_projector(revenue_base,
                           projected_revneues,
                           sales_to_capital_ratios,
                           asset_liquidation_during_negative_growth=0):
    reinvestment = (pd.concat([pd.Series(revenue_base),
                               projected_revneues],
                             ignore_index=False).diff().dropna()/sales_to_capital_ratios)
    reinvestment = reinvestment.where(reinvestment>0, (reinvestment*asset_liquidation_during_negative_growth))
    return(reinvestment)

"""# Valuator"""

def calculate_annualized_return(intrinsic_equity_future_value, equity_value, valuation_interval_in_years):
    # Handle the case where intrinsic_equity_future_value is 0 to avoid division by zero
    if np.isclose(intrinsic_equity_future_value, 0):
        return 0
    # Implementing the generalized formula for expected annualized return on equity
    sign = np.sign(intrinsic_equity_future_value)
    absolute_intrinsic_equity_future_value = np.abs(intrinsic_equity_future_value)

    return sign * ((absolute_intrinsic_equity_future_value / equity_value) ** (1/valuation_interval_in_years)) - 1

def calculate_total_excess_return(intrinsic_equity_present_value, equity_value):
    if equity_value == 0:
        raise ValueError("equity_value cannot be zero to avoid division by zero.")

    return (intrinsic_equity_present_value / equity_value) - 1

def calculate_annualized_return_e(intrinsic_equity_present_value, equity_value, valuation_interval_in_years):
    if equity_value == 0:
        raise ValueError("equity_value cannot be zero to avoid division by zero.")
    if valuation_interval_in_years == 0:
        raise ValueError("valuation_interval_in_years cannot be zero to avoid division by zero.")

    sign = np.sign(intrinsic_equity_present_value)

    return sign * ((np.abs(intrinsic_equity_present_value) / equity_value) ** (1 / valuation_interval_in_years)) - 1

def valuator_multi_phase(
    risk_free_rate,
    ERP,
    equity_value,
    debt_value,
    unlevered_beta,
    terminal_unlevered_beta,
    year_beta_begins_to_converge_to_terminal_beta,
    current_pretax_cost_of_debt,
    terminal_pretax_cost_of_debt,
    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt,
    current_effective_tax_rate,
    marginal_tax_rate,
    year_effective_tax_rate_begin_to_converge_marginal_tax_rate,
    revenue_base,
    revenue_growth_rate_cycle1_begin,
    revenue_growth_rate_cycle1_end,
    revenue_growth_rate_cycle2_begin,
    revenue_growth_rate_cycle2_end,
    revenue_growth_rate_cycle3_begin,
    revenue_growth_rate_cycle3_end,
    revenue_convergance_periods_cycle1,
    revenue_convergance_periods_cycle2,
    revenue_convergance_periods_cycle3,
    length_of_cylcle1,
    length_of_cylcle2,
    length_of_cylcle3,
    current_sales_to_capital_ratio,
    terminal_sales_to_capital_ratio,
    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital,
    current_operating_margin,
    terminal_operating_margin,
    year_operating_margin_begins_to_converge_to_terminal_operating_margin,
    additional_return_on_cost_of_capital_in_perpetuity=0.0,
    cash_and_non_operating_asset=0.0,
    asset_liquidation_during_negative_growth=0,
    current_invested_capital='implicit'):

    valuation_interval_in_years = int(length_of_cylcle1) + int(length_of_cylcle2) + int(length_of_cylcle3)
    terminal_growth_rate = revenue_growth_rate_cycle3_end
    ### Estimate Cost of Capital during the valution cycle
    projected_cost_of_capital, projected_beta , terminal_beta , projected_cost_of_equity , projected_after_tax_cost_of_debt = cost_of_capital_projector(unlevered_beta=unlevered_beta,
                                                          terminal_unlevered_beta=terminal_unlevered_beta,
                                                          current_pretax_cost_of_debt=current_pretax_cost_of_debt,
                                                          terminal_pretax_cost_of_debt=terminal_pretax_cost_of_debt,
                                                          equity_value=equity_value,
                                                          debt_value=debt_value,
                                                          marginal_tax_rate=marginal_tax_rate,
                                                          risk_free_rate=risk_free_rate,
                                                          ERP=ERP,
                                                          valuation_interval_in_years=valuation_interval_in_years,
                                                          year_beta_begins_to_converge_to_terminal_beta=year_beta_begins_to_converge_to_terminal_beta,
                                                          year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt)
    projected_cost_of_capital_cumulative= (1+projected_cost_of_capital).cumprod()
    projected_cost_of_equity_cumulative= (1+projected_cost_of_equity).cumprod()
    ### Estimate Future revnues, and growth

    projected_revneues,projected_revenue_growth = revenue_projector_multi_phase(revenue_base = revenue_base,
                                  revenue_growth_rate_cycle1_begin = revenue_growth_rate_cycle1_begin,
                                  revenue_growth_rate_cycle1_end = revenue_growth_rate_cycle1_end,
                                  revenue_growth_rate_cycle2_begin = revenue_growth_rate_cycle2_begin,
                                  revenue_growth_rate_cycle2_end = revenue_growth_rate_cycle2_end,
                                  revenue_growth_rate_cycle3_begin = revenue_growth_rate_cycle3_begin,
                                  revenue_growth_rate_cycle3_end = revenue_growth_rate_cycle3_end,
                                  length_of_cylcle1=length_of_cylcle1,
                                  length_of_cylcle2=length_of_cylcle2,
                                  length_of_cylcle3=length_of_cylcle3,
                                  revenue_convergance_periods_cycle1 = revenue_convergance_periods_cycle1,
                                  revenue_convergance_periods_cycle2 = revenue_convergance_periods_cycle2,
                                  revenue_convergance_periods_cycle3 = revenue_convergance_periods_cycle3)
    ### Estmimate tax rates
    projected_tax_rates = tax_rate_projector(current_effective_tax_rate=current_effective_tax_rate,
                                            marginal_tax_rate=marginal_tax_rate,
                                            valuation_interval_in_years=valuation_interval_in_years,
                                            year_effective_tax_rate_begin_to_converge_marginal_tax_rate=year_effective_tax_rate_begin_to_converge_marginal_tax_rate)
    ### Estimate slaes to capital ratio during valuation for reinvestment
    sales_to_capital_ratios = sales_to_capital_projector(current_sales_to_capital_ratio,
                               terminal_sales_to_capital_ratio,
                               valuation_interval_in_years=valuation_interval_in_years,
                               year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital=year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital)

    ### Estimate Reinvestemnt
    projected_reinvestment = reinvestment_projector(revenue_base=revenue_base,
                                                    projected_revneues = projected_revneues,
                                                    sales_to_capital_ratios=sales_to_capital_ratios,
                                                    asset_liquidation_during_negative_growth=asset_liquidation_during_negative_growth)

    ### Estimate invested Capital
    invested_capital = projected_reinvestment.copy()
    if current_invested_capital == 'implicit':
        current_invested_capital = revenue_base / current_sales_to_capital_ratio
    invested_capital[0] = invested_capital[0] + current_invested_capital
    invested_capital = invested_capital.cumsum()

    ### Operating Margin
    projected_operating_margins = operating_margin_projector(current_operating_margin,
                                                            terminal_operating_margin,
                                                            valuation_interval_in_years=valuation_interval_in_years,
                                                            year_operating_margin_begins_to_converge_to_terminal_operating_margin=year_operating_margin_begins_to_converge_to_terminal_operating_margin)
    ###EBIT
    projected_operating_income = projected_revneues * projected_operating_margins
    ### After Tax EBIT (EBI)
    projected_operating_income_after_tax = (projected_operating_income*(1-projected_tax_rates))
    ### FCFF: EBI-Reinvestment
    projected_FCFF = projected_operating_income_after_tax - projected_reinvestment
    ### compute ROIC
    ROIC = (projected_operating_income_after_tax/invested_capital)
    ### Compute Terminal Value
    terminal_cost_of_capital = projected_cost_of_capital[-1:].values
    terminal_cost_of_equity = projected_cost_of_equity[-1:].values
    if terminal_growth_rate < 0:
        terminal_reinvestment_rate=0
    else:
        terminal_reinvestment_rate = terminal_growth_rate/(terminal_cost_of_capital+additional_return_on_cost_of_capital_in_perpetuity)
    terminal_revenue = projected_revneues[-1:].values * (1+terminal_growth_rate)
    terminal_operating_income = terminal_revenue * terminal_operating_margin
    terminal_operating_income_after_tax = terminal_operating_income*(1-marginal_tax_rate)
    terminal_reinvestment = terminal_operating_income_after_tax* terminal_reinvestment_rate
    terminal_FCFF = terminal_operating_income_after_tax - terminal_reinvestment
    terminal_value = terminal_FCFF/(terminal_cost_of_capital-terminal_growth_rate)
    termimal_discount_rate = (terminal_cost_of_capital-terminal_growth_rate)*(1+projected_cost_of_capital).prod()
    termimal_equity_discount_rate = (terminal_cost_of_equity-terminal_growth_rate)*(1+projected_cost_of_equity).prod()


    ### Concatinate Projected Values with termianl values
    projected_cost_of_capital_cumulative_with_terminal_rate =pd.concat([projected_cost_of_capital_cumulative,
                                                    pd.Series(termimal_discount_rate)])
    projected_cost_of_equity_cumulative_with_terminal_rate = pd.concat([projected_cost_of_equity_cumulative,
                                                    pd.Series(termimal_equity_discount_rate)])

    projected_revenue_growth = pd.concat([projected_revenue_growth,
                                        pd.Series(terminal_growth_rate)])
    projected_revneues =pd.concat([projected_revneues,
                                  pd.Series(terminal_revenue)])
    projected_tax_rates = pd.concat([projected_tax_rates,
                                   pd.Series(marginal_tax_rate)])
    projected_reinvestment = pd.concat([projected_reinvestment,
                                        pd.Series(terminal_reinvestment)])


    ### Estimate invested Capital
    invested_capital = pd.concat([invested_capital,
                                        pd.Series(np.nan)])
    ### Estimate ROIC
    terminal_ROIC = terminal_cost_of_capital+additional_return_on_cost_of_capital_in_perpetuity
    ROIC = pd.concat([ROIC,
                      pd.Series(terminal_ROIC)])

    projected_operating_margins = pd.concat([projected_operating_margins,
                                        pd.Series(terminal_operating_margin)])
    projected_operating_income = pd.concat([projected_operating_income,
                                            pd.Series(terminal_operating_income)])
    projected_operating_income_after_tax = pd.concat([projected_operating_income_after_tax,
                                                      pd.Series(terminal_operating_income_after_tax)])
    projected_FCFF_value = pd.concat([projected_FCFF,
                                      pd.Series(terminal_value)])

    projected_FCFF = pd.concat([projected_FCFF,
                                pd.Series(terminal_FCFF)])

    projected_beta = pd.concat([projected_beta,
                                pd.Series(terminal_beta)])

    sales_to_capital_ratios = pd.concat([sales_to_capital_ratios,
                                pd.Series([terminal_sales_to_capital_ratio])])

    ### Add terminal cost of debt to the terminal year
    projected_after_tax_cost_of_debt_with_terminal = pd.concat([projected_after_tax_cost_of_debt,
                                                                pd.Series(projected_after_tax_cost_of_debt[-1:].values)])

    reinvestmentRate = projected_reinvestment/projected_operating_income_after_tax

    df_valuation = df_valuation = pd.DataFrame({"cumWACC":projected_cost_of_capital_cumulative_with_terminal_rate,
                                 "cumCostOfEquity":projected_cost_of_equity_cumulative_with_terminal_rate,
                                'beta':projected_beta,
                                 'ERP':ERP,
                                 'projected_after_tax_cost_of_debt':projected_after_tax_cost_of_debt_with_terminal,
                                'revenueGrowth':projected_revenue_growth,
                                "revneues":projected_revneues,
                                 "margins":projected_operating_margins,
                                 'ebit':projected_operating_income,
                                 "sales_to_capital_ratio":sales_to_capital_ratios,
                                "taxRate":projected_tax_rates,
                               'afterTaxOperatingIncome':projected_operating_income_after_tax,
                               "reinvestment":projected_reinvestment,
                               "invested_capital":invested_capital,
                               "ROIC":ROIC,
                               'reinvestmentRate':reinvestmentRate,
                               'FCFF':projected_FCFF,
                               'projected_FCFF_value':projected_FCFF_value})
    #### Add reinvestment rate and expected growth rate
    df_valuation['PVFCFF'] = df_valuation['FCFF']/df_valuation['cumWACC']
    value_of_operating_assets = df_valuation['PVFCFF'].sum()
    firm_value =  pd.Series(value_of_operating_assets + cash_and_non_operating_asset)[0]
    intrinsic_equity_present_value = firm_value - debt_value

    #### Future Frim, Debt and Equity Value
    cum_cost_of_debt_at_the_end_of_end_of_valuation =  (projected_after_tax_cost_of_debt+1).prod()
    cum_cost_of_capital_at_the_end_of_valuation =projected_cost_of_capital_cumulative[-1:].values[0]
    cum_cost_of_equity_at_the_end_of_valuation = projected_cost_of_equity_cumulative[-1:].values[0]

    ### FV Debt
    debt_future_value = cum_cost_of_debt_at_the_end_of_end_of_valuation * debt_value

    ### FV Frim
    firm_future_value = cum_cost_of_capital_at_the_end_of_valuation * cum_cost_of_capital_at_the_end_of_valuation

    ### FV Equity
    intrinsic_equity_future_value = intrinsic_equity_present_value * cum_cost_of_equity_at_the_end_of_valuation
    ## Returns
    def cum_return_calculator(value,
                            period,
                            append_nan=True):
        period = int(period)
        cum_return_series = pd.Series([1+value]*period).cumprod()
        if append_nan:
            cum_return_series = pd.concat([cum_return_series,pd.Series(np.nan)])
        return(cum_return_series)

    acceptable_annualized_return_on_equity = ((cum_cost_of_equity_at_the_end_of_valuation)**(1/valuation_interval_in_years))-1
    expected_annualized_return_on_equity = calculate_annualized_return(intrinsic_equity_future_value, equity_value, valuation_interval_in_years)
    excess_annualized_return_on_equity = calculate_annualized_return_e(intrinsic_equity_present_value, equity_value, valuation_interval_in_years)
    total_excess_return_on_equity_during_valuation_cycle = calculate_total_excess_return(intrinsic_equity_present_value, equity_value)
    # print("acceptable_annualized_return_on_equity",acceptable_annualized_return_on_equity)
    # print("expected_annualized_return_on_equity",expected_annualized_return_on_equity)
    # print("excess_annualized_return_on_equity",excess_annualized_return_on_equity)
    # print("total_excess_return_on_equity_during_valuation_cycle",total_excess_return_on_equity_during_valuation_cycle)
    # print(pd.Series([1+acceptable_annualized_return_on_equity]*valuation_interval_in_years))
    cum_acceptable_annualized_return_on_equity = cum_return_calculator(value = acceptable_annualized_return_on_equity,
                                                                       period = valuation_interval_in_years,
                                                                       append_nan=True)
    cum_expected_annualized_return_on_equity = cum_return_calculator(value = expected_annualized_return_on_equity,
                                                                       period = valuation_interval_in_years,
                                                                       append_nan=True)
    cum_excess_annualized_return_on_equity = cum_return_calculator(value = excess_annualized_return_on_equity,
                                                                       period = valuation_interval_in_years,
                                                                       append_nan=True)
    df_valuation['cum_acceptable_annualized_return_on_equity'] = cum_acceptable_annualized_return_on_equity
    df_valuation['cum_expected_annualized_return_on_equity'] = cum_expected_annualized_return_on_equity
    df_valuation['cum_excess_annualized_return_on_equity'] = cum_excess_annualized_return_on_equity
    df_valuation['cum_excess_annualized_return_on_equity_realized'] = df_valuation['cum_expected_annualized_return_on_equity']/df_valuation['cumCostOfEquity']
    df_valuation['excess_annualized_return_on_equity'] = pd.concat([pd.Series([excess_annualized_return_on_equity]*int(valuation_interval_in_years)),pd.Series(np.nan)])
    # print("valuation complete")
    return({'valuation':df_valuation,
            'firm_value':firm_value,
            'equity_value':intrinsic_equity_present_value,
            'cash_and_non_operating_asset':cash_and_non_operating_asset,
            'debt_value':debt_value,
            'value_of_operating_assets':value_of_operating_assets})

def point_estimate_describer(base_case_valuation):
    print('value of operating assets',np.round(base_case_valuation['value_of_operating_assets'],2),'\n',
        'cash and non operating asset',np.round(base_case_valuation['cash_and_non_operating_asset'],2),'\n',
        'debt value',np.round(base_case_valuation['debt_value'],2),'\n',
        'firm value',np.round(base_case_valuation['firm_value'],2),'\n',
        'Total terminal value',"${:.2f} billion".format(np.round(base_case_valuation['equity_value'],2)))
    df_valuation = base_case_valuation['valuation']
    df_valuation=  df_valuation.reset_index(drop=True)
    df_valuation['Year']= df_valuation.reset_index()['index']+1
    df_valuation.loc[df_valuation['Year'] == df_valuation['Year'].max(),"Year"] = 'Terminal'
    df_valuation= df_valuation.set_index("Year")
    return(df_valuation)

"""## Monte Carlo Simulation"""

def monte_carlo_valuator_multi_phase(
    risk_free_rate,
    ERP,
    equity_value,
    debt_value,
    unlevered_beta,
    terminal_unlevered_beta,
    year_beta_begins_to_converge_to_terminal_beta,
    current_pretax_cost_of_debt,
    terminal_pretax_cost_of_debt,
    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt,
    current_effective_tax_rate,
    marginal_tax_rate,
    year_effective_tax_rate_begin_to_converge_marginal_tax_rate,
     revenue_base,
     revenue_growth_rate_cycle1_begin,
     revenue_growth_rate_cycle1_end,
     revenue_growth_rate_cycle2_begin,
     revenue_growth_rate_cycle2_end,
     revenue_growth_rate_cycle3_begin,
     revenue_growth_rate_cycle3_end,
    revenue_convergance_periods_cycle1,
    revenue_convergance_periods_cycle2,
    revenue_convergance_periods_cycle3,
    length_of_cylcle1,
    length_of_cylcle2,
    length_of_cylcle3,
    current_sales_to_capital_ratio,
    terminal_sales_to_capital_ratio,
    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital,
    current_operating_margin,
    terminal_operating_margin,
    year_operating_margin_begins_to_converge_to_terminal_operating_margin,
    additional_return_on_cost_of_capital_in_perpetuity,
    cash_and_non_operating_asset,
    asset_liquidation_during_negative_growth,
    current_invested_capital,
    sample_size=1000,
    list_of_correlation_between_variables=[['additional_return_on_cost_of_capital_in_perpetuity','terminal_sales_to_capital_ratio',0.4],
                                           ['additional_return_on_cost_of_capital_in_perpetuity','terminal_operating_margin',.6]]):
    variables_distributsion = [risk_free_rate,
                                   ERP,
                                   equity_value,
                                   debt_value,
                                   unlevered_beta,
                                    terminal_unlevered_beta,
                                    year_beta_begins_to_converge_to_terminal_beta,
                                    current_pretax_cost_of_debt,
                                    terminal_pretax_cost_of_debt,
                                    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt,
                                    current_effective_tax_rate,
                                    marginal_tax_rate,
                                    year_effective_tax_rate_begin_to_converge_marginal_tax_rate,
                                    revenue_base,
                                    revenue_growth_rate_cycle1_begin,
                                    revenue_growth_rate_cycle1_end,
                                    revenue_growth_rate_cycle2_begin,
                                    revenue_growth_rate_cycle2_end,
                                    revenue_growth_rate_cycle3_begin,
                                    revenue_growth_rate_cycle3_end,
                                    revenue_convergance_periods_cycle1,
                                    revenue_convergance_periods_cycle2,
                                    revenue_convergance_periods_cycle3,
                                    length_of_cylcle1,
                                    length_of_cylcle2,
                                    length_of_cylcle3,
                                    current_sales_to_capital_ratio,
                                    terminal_sales_to_capital_ratio,
                                    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital,
                                    current_operating_margin,
                                    terminal_operating_margin,
                                    year_operating_margin_begins_to_converge_to_terminal_operating_margin,
                                    additional_return_on_cost_of_capital_in_perpetuity,
                                    cash_and_non_operating_asset,
                                    asset_liquidation_during_negative_growth,
                                    current_invested_capital]
    variable_names = ['risk_free_rate',
                                   'ERP',
                                   'equity_value',
                                   'debt_value',
                                   'unlevered_beta',
                                    'terminal_unlevered_beta',
                                    'year_beta_begins_to_converge_to_terminal_beta',
                                    'current_pretax_cost_of_debt',
                                    'terminal_pretax_cost_of_debt',
                                    'year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt',
                                    'current_effective_tax_rate',
                                    'marginal_tax_rate',
                                    'year_effective_tax_rate_begin_to_converge_marginal_tax_rate',
                                    'revenue_base',
                                    'revenue_growth_rate_cycle1_begin',
                                    'revenue_growth_rate_cycle1_end',
                                    'revenue_growth_rate_cycle2_begin',
                                    'revenue_growth_rate_cycle2_end',
                                    'revenue_growth_rate_cycle3_begin',
                                    'revenue_growth_rate_cycle3_end',
                                    'revenue_convergance_periods_cycle1',
                                    'revenue_convergance_periods_cycle2',
                                    'revenue_convergance_periods_cycle3',
                                    'length_of_cylcle1',
                                    'length_of_cylcle2',
                                    'length_of_cylcle3',
                                    'current_sales_to_capital_ratio',
                                    'terminal_sales_to_capital_ratio',
                                    'year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital',
                                    'current_operating_margin',
                                    'terminal_operating_margin',
                                    'year_operating_margin_begins_to_converge_to_terminal_operating_margin',
                                    'additional_return_on_cost_of_capital_in_perpetuity',
                                    'cash_and_non_operating_asset',
                                    'asset_liquidation_during_negative_growth',
                                    'current_invested_capital']
    ### The following variable  should have "year" in their definition but I did not think of ut. So I am adding them to list_of_columns_with_year_to_be_int
    list_of_columns_with_year_to_be_int = [s for s in variable_names if "year" in s] +['length_of_cylcle1','length_of_cylcle2','length_of_cylcle3',
                                                                                       'revenue_convergance_periods_cycle1',
                                                                                       'revenue_convergance_periods_cycle2','revenue_convergance_periods_cycle3']
    ### Build a DataFarame to have index - location of each variable in the correlation matrix
    dict_of_varible = dict(zip(variable_names,
                            range(0,len(variable_names))))
    df_variables = pd.DataFrame([dict_of_varible])

    ### Initaile Correlation Matrix
    R = ot.CorrelationMatrix(len(variables_distributsion))
    ### pair correlation between each variable
    for pair_of_variable in list_of_correlation_between_variables:
        location = df_variables[pair_of_variable[:2]].values[0]
        #print(location)
        R[int(location[0]),int(location[1])] = pair_of_variable[2]

    ### Build the correlation into composed distribution function
    ### For ot.NormalCopula The correlation matrix must be definite positive
    ### Here is an implementaion on how to get the nearest psd matirx https://stackoverflow.com/questions/43238173/python-convert-matrix-to-positive-semi-definite
    copula = ot.NormalCopula(R)
    BuiltComposedDistribution = ot.ComposedDistribution(variables_distributsion,
                                                        copula)
    ### Generate samples
    generated_sample = BuiltComposedDistribution.getSample(sample_size)
    df_generated_sample = pd.DataFrame.from_records(generated_sample, columns= variable_names)
    df_generated_sample[list_of_columns_with_year_to_be_int] = df_generated_sample[list_of_columns_with_year_to_be_int].apply(lambda x: round(x))
    print("Scenario Generation Complete", df_generated_sample.shape)
    df_generated_sample['full_valuation']= df_generated_sample.apply(lambda row:
                                                                    valuator_multi_phase(
                                                                        risk_free_rate = row['risk_free_rate'],
                                                                        ERP = row['ERP'],
                                                                        equity_value = row['equity_value'],
                                                                        debt_value = row['debt_value'],
                                                                        unlevered_beta = row['unlevered_beta'],
                                                                        terminal_unlevered_beta = row['terminal_unlevered_beta'],
                                                                        year_beta_begins_to_converge_to_terminal_beta = row['year_beta_begins_to_converge_to_terminal_beta'],
                                                                        current_pretax_cost_of_debt = row['current_pretax_cost_of_debt'],
                                                                        terminal_pretax_cost_of_debt = row['terminal_pretax_cost_of_debt'],
                                                                        year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt = row['year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt'],
                                                                        current_effective_tax_rate = row['current_effective_tax_rate'],
                                                                        marginal_tax_rate = row['marginal_tax_rate'],
                                                                        year_effective_tax_rate_begin_to_converge_marginal_tax_rate = row['year_effective_tax_rate_begin_to_converge_marginal_tax_rate'],
                                                                        revenue_base = row['revenue_base'],
                                                                        revenue_growth_rate_cycle1_begin = row['revenue_growth_rate_cycle1_begin'],
                                                                        revenue_growth_rate_cycle1_end = row['revenue_growth_rate_cycle1_end'],
                                                                        revenue_growth_rate_cycle2_begin = row['revenue_growth_rate_cycle2_begin'],
                                                                        revenue_growth_rate_cycle2_end = row['revenue_growth_rate_cycle2_end'],
                                                                        revenue_growth_rate_cycle3_begin = row['revenue_growth_rate_cycle3_begin'],
                                                                        revenue_growth_rate_cycle3_end = row['revenue_growth_rate_cycle3_end'],
                                                                        revenue_convergance_periods_cycle1 = row['revenue_convergance_periods_cycle1'],
                                                                        revenue_convergance_periods_cycle2 = row['revenue_convergance_periods_cycle2'],
                                                                        revenue_convergance_periods_cycle3 = row['revenue_convergance_periods_cycle3'],
                                                                        length_of_cylcle1 = row['length_of_cylcle1'],
                                                                        length_of_cylcle2 = row['length_of_cylcle2'],
                                                                        length_of_cylcle3 = row['length_of_cylcle3'],
                                                                        current_sales_to_capital_ratio = row['current_sales_to_capital_ratio'],
                                                                        terminal_sales_to_capital_ratio = row['terminal_sales_to_capital_ratio'],
                                                                        year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital = row['year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital'],
                                                                        current_operating_margin = row['current_operating_margin'],
                                                                        terminal_operating_margin = row['terminal_operating_margin'],
                                                                        year_operating_margin_begins_to_converge_to_terminal_operating_margin = row['year_operating_margin_begins_to_converge_to_terminal_operating_margin'],
                                                                        additional_return_on_cost_of_capital_in_perpetuity = row['additional_return_on_cost_of_capital_in_perpetuity'],
                                                                        cash_and_non_operating_asset = row['cash_and_non_operating_asset'],
                                                                        asset_liquidation_during_negative_growth=row['asset_liquidation_during_negative_growth'],
                                                                        current_invested_capital=row['current_invested_capital']),
                                                                        axis=1)
    ### extract the valuation result
    df_generated_sample['valuation'] = df_generated_sample['full_valuation'].apply(lambda x: x['valuation'])
    df_generated_sample['equity_valuation'] = df_generated_sample['full_valuation'].apply(lambda x: x['equity_value'])
    df_generated_sample['firm_valuation'] = df_generated_sample['full_valuation'].apply(lambda x: x['firm_value'])
    df_generated_sample['terminal_revenue'] = df_generated_sample['valuation'].apply(lambda x: x['revneues'].values[-1])
    df_generated_sample['terminal_operating_margin'] = df_generated_sample['valuation'].apply(lambda x: x['margins'].values[-1])
    df_generated_sample['terminal_reinvestmentRate'] = df_generated_sample['valuation'].apply(lambda x: x['reinvestmentRate'].values[-1])
    df_generated_sample['terminal_afterTaxOperatingIncome'] = df_generated_sample['valuation'].apply(lambda x: x['afterTaxOperatingIncome'].values[-1])
    df_generated_sample['terminal_FCFF'] = df_generated_sample['valuation'].apply(lambda x: x['FCFF'].values[-1])
    df_generated_sample['cumWACC'] = df_generated_sample['valuation'].apply(lambda x: x['cumWACC'][:-1].values)
    df_generated_sample['cumCostOfEquity'] = df_generated_sample['valuation'].apply(lambda x: x['cumCostOfEquity'][:-1].values)
    df_generated_sample['cum_acceptable_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['cum_acceptable_annualized_return_on_equity'][:-1].values)
    df_generated_sample['cum_expected_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['cum_expected_annualized_return_on_equity'][:-1].values)
    df_generated_sample['cum_excess_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['cum_excess_annualized_return_on_equity'][:-1].values)
    df_generated_sample['cum_excess_annualized_return_on_equity_realized'] = df_generated_sample['valuation'].apply(lambda x: x['cum_excess_annualized_return_on_equity_realized'][:-1].values)
    df_generated_sample['excess_annualized_return_on_equity'] = df_generated_sample['valuation'].apply(lambda x: x['excess_annualized_return_on_equity'][:-1].values)
    df_generated_sample['ROIC'] = df_generated_sample['valuation'].apply(lambda x: x['ROIC'][:-1].values)
    df_generated_sample['invested_capital'] = df_generated_sample['valuation'].apply(lambda x: x['invested_capital'][:-1].values)
    return(df_generated_sample)

"""### Plotly Charts"""

def histogram_plotter_plotly(data,
                              colmn_name,
                              xlabel,
                              title='Data',
                              bins=30,
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              histnorm='percent',
                              marginal=None,
                              height=470,
                              width=670):
    """Plot Historgam via Plotly"""
    fig = px.histogram(data,
                       x=colmn_name,
                       histnorm=histnorm,
                       nbins=bins,
                       labels={colmn_name:xlabel},
                       marginal=marginal)
    ### Make an educated guess on the y_max for line on the historgram
    n, bin_edges = np.histogram(data[colmn_name],bins=bins,density=False)
    bin_probability = n/float(n.sum())
    y_max = np.max(n/(n.sum())*100) *1.65
    ### Ad trace of percentiles
    for i in range(len(percentile)):
        fig = fig.add_trace(go.Scatter(x=[np.percentile(data[colmn_name],percentile[i]), np.percentile(data[colmn_name],percentile[i])],
                                       y=(0,y_max),
                                       mode="lines",
                                       name= str(percentile[i])+' Percentile',
                                       marker=dict(color=color[i])))
        #fig = fig.add_vline(x = np.percentile(data[colmn_name],percentile[i]), line_dash = 'dash',line_color=color[i])
        #print(str(percentile[i])+" Percentile",np.percentile(data[colmn_name],percentile[i]))
        fig.update_layout(height=height, width=width,title=title,
                          legend=dict(orientation="v"))
    return(fig)

def ecdf_plotter_plotly(data,
                              colmn_name,
                              xlabel,
                              title='Data',
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              marginal=None,
                              height=500,
                              width=700):
    """Plot ECDF via Plotly"""
    fig = px.ecdf(data,
                     x=colmn_name,
                     labels={colmn_name:xlabel},
                     marginal=marginal)
    for i in range(len(percentile)):
        fig = fig.add_trace(go.Scatter(x=[np.percentile(data[colmn_name],percentile[i]), np.percentile(data[colmn_name],percentile[i])],
                                       y=(0,1),
                                       mode="lines",
                                       name= str(percentile[i])+' Percentile',
                                       marker=dict(color=color[i])))
        #fig = fig.add_vline(x = np.percentile(data[colmn_name],percentile[i]), line_dash = 'dash',line_color=color[i])
        #print(str(percentile[i])+" Percentile",np.percentile(data[colmn_name],percentile[i]))
        fig.update_layout(height=height, width=width,title=title,
                          legend=dict(orientation="v"))
    return(fig)

def time_series_plotly(df_data,
                       x,
                       yleft,
                       yright,
                       height=500,
                       width=1600,
                       title=None):
    """ Graph 2 time series on 2 different y-axis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])

    # Add traces
    fig.add_trace(
        go.Scatter(x=df_data[x], y=df_data[yleft], name=yleft),
        secondary_y=False,
    )

    fig.add_trace(
        go.Scatter(x=df_data[x], y=df_data[yright], name=yright),
        secondary_y=True,
    )
    fig = fig.update_layout(height=height, width=width,title=title)
    return(fig)

def plotly_line_bar_chart(df_data,
                       x,
                       ybar,
                       yline,
                       height=500,
                       width=1600,
                       rangemode=None,
                       title=None):
    """ Graph 2 time series on 2 different y-axis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig.update_yaxes(rangemode='tozero')

    for bar_var in ybar:
        # Add traces
        fig.add_trace(
            go.Bar(x=df_data[x], y=df_data[bar_var],name=bar_var),
            secondary_y=False
            )
    #fig.update_yaxes(rangemode='tozero')
    for line_var in yline:
        fig.add_trace(
            go.Scatter(x=df_data[x], y=df_data[line_var],name=line_var),
            secondary_y=True,
            )
    if rangemode != None:
        fig.update_yaxes(rangemode=rangemode)
    fig = fig.update_layout(height=height, width=width,title=title)
    return(fig)


def plotly_line_dash_bar_chart(df_data,
                       x,
                       ybar,
                       yline,
                       ydash,
                       height=500,
                       width=1600,
                       rangemode=None,
                       title=None,
                       barmode='group',
                       texttemplate= "%{value}"
                       ):
    """ Graph 2 time series on 2 different y-axis"""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig.update_yaxes(rangemode='tozero')

    for bar_var in ybar:
        # Add traces
        fig.add_trace(
            go.Bar(x=df_data[x],
                   y=df_data[bar_var],
                   name=bar_var,
                   text = df_data[bar_var],
                   textposition="inside",
                   texttemplate= texttemplate,
                   textfont_color="white"),
            secondary_y=False,
            )
    for line_var in yline:
        fig.add_trace(
            go.Scatter(x=df_data[x],
                       y=df_data[line_var],
                       name=line_var
                       ),
            secondary_y=True,
            )

    for dash_var in ydash:
        fig.add_trace(
            go.Scatter(x=df_data[x],
                       y=df_data[dash_var],
                       name=dash_var,
                       line = dict(dash='dot')),
            secondary_y=True,
            )
    if rangemode != None:
        fig.update_yaxes(rangemode=rangemode)
    fig = fig.update_layout(height=height,
                            width=width,
                            title=title,
                            barmode=barmode)
    return(fig)

def line_plotter_with_error_bound(df_data,
                                  x,
                                  list_of_mid_point,
                                  list_of_lower_bound,
                                  list_of_upper_bound,
                                  list_of_bar=[],
                                  list_of_name=[],
                                  list_of_fillcolor= ['rgba(68, 68, 68, 0.3)'],
                                  list_of_line_color= ['rgb(31, 119, 180)'],
                                  title=None,
                                  yaxis_title=None,
                                  height=600,
                                  width=900):
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    # Create figure with secondary y-axis
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    #fig.update_yaxes(rangemode='tozero')
    for mid_point,lower_bound,upper_bound,name,fillcolor,line_color in zip(list_of_mid_point,
                                                                            list_of_lower_bound,
                                                                            list_of_upper_bound,
                                                                            list_of_name,
                                                                            list_of_fillcolor,
                                                                            list_of_line_color):
        # Add traces
        fig.add_trace (
            go.Scatter(
                name = name,
                x=df_data[x],
                y=df_data[mid_point],
                mode='lines',
                line=dict(color=line_color)
                ))
        fig.add_trace(
            go.Scatter(
                name= name + ' UB',
                x = df_data[x],
                y = df_data[upper_bound],
                mode='lines',
                marker=dict(color=line_color),
                line=dict(width=0),
                showlegend=False
                ))
        fig.add_trace(
            go.Scatter(
        name= name + ' LB',
        x= df_data[x],
        y=df_data[lower_bound],
        marker=dict(color=line_color),
        line=dict(width=0),
        mode='lines',
        fillcolor= fillcolor,
        fill='tonexty',
        showlegend=False)
        )

    for bar_var in list_of_bar:
        fig.add_trace(
            go.Bar(x=df_data[x],
                   y=df_data[bar_var],
                   name=bar_var,
                   text = df_data[bar_var],
                   marker=dict(color='rgba(46, 120, 237,.6)'),
                   textposition="inside",
                   #texttemplate= texttemplate,
                   textfont_color="white"),
            secondary_y=True,
            )

    fig.update_layout(
        yaxis2=dict(
        side="right",
        #rangemode='tozero',
        range=[-.05, .3],
        #overlaying="y",
        #tickmode="sync"
        )
        )
    # fig.update_yaxes(rangemode='tozero')
    fig.update_layout(yaxis_title= yaxis_title,
                      title= title,
                      hovermode="x",
                      height=height,
                      width=width,)
    return(fig)

"""### Valuation Describer"""

def return_values_from_list_extractor(df_data,
                                        col,
                                        add_1=True):
    "This function is to get the cost stats of specied col by year"
    df_cost_of_cap = pd.DataFrame(list(df_data[col].values))
    if add_1:
        df_cost_of_cap[-1] = 1.00
    else:
        df_cost_of_cap[-1] = 0
    df_cost_of_cap = pd.melt(df_cost_of_cap,
                                var_name=['year'],
                                value_name=col).dropna()
    df_cost_of_cap = df_cost_of_cap.groupby(['year'])[[col]].describe().reset_index()
    df_cost_of_cap = data_frame_flattener(df_cost_of_cap)
    df_cost_of_cap = df_cost_of_cap.set_index("year")
    return(df_cost_of_cap)

def return_values_from_list_extractor_step2(df_data,
                                            cum_ret_col,
                                            ret_col):
    list_of_df_return = []
    for col in cum_ret_col:
        df_res = return_values_from_list_extractor(df_data,
                                                   col=col)
        list_of_df_return.append(df_res)
    for col in ret_col:
        df_res = return_values_from_list_extractor(df_data,
                                                   col=col,
                                                   add_1=False)
        list_of_df_return.append(df_res)
    df_returns = pd.concat(list_of_df_return,axis=1)
    df_returns = df_returns.reset_index(drop=True).reset_index().rename(columns={"index":"year"})
    return(df_returns)

def valuation_describer(df_intc_valuation,sharesOutstanding):
    """Describe stats of monte dcf carlo simulation"""
    ### Get the Equity value at eahc percentile
    current_market_cap = df_intc_valuation['equity_value'].median()
    percentiles=np.arange(0, 110, 10)
    equity_value_at_each_percentile = np.percentile(df_intc_valuation['equity_valuation'],
                                                    percentiles)
    equity_value_at_20_percentile=equity_value_at_each_percentile[2]
    equity_value_at_80_percentile=equity_value_at_each_percentile[8]
    df_valuation_res = pd.DataFrame({"percentiles":percentiles,
                                     "equity_value":equity_value_at_each_percentile})
    df_valuation_res['current_market_cap'] = current_market_cap
    df_valuation_res['current_price_per_share'] = current_market_cap/sharesOutstanding
    df_valuation_res['equity_value_per_share'] = df_valuation_res['equity_value']/sharesOutstanding
    df_valuation_res['Price/Value']= df_valuation_res['current_market_cap']/df_valuation_res['equity_value']
    df_valuation_res['PNL']= (df_valuation_res['equity_value']/df_valuation_res['current_market_cap'])-1
    ### Histogram
    fig = histogram_plotter_plotly(data=df_intc_valuation,
                              colmn_name ='equity_valuation',
                              xlabel ='Market Cap',
                              title='Intrinsic Equity Value Distribution',
                              bins=200,
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              histnorm='percent',
                              height=510,
                              width=720)
    fig = fig.add_vline(x = current_market_cap, line_dash = 'dash',line_color='black',
                        annotation_text="-Current Market Cap",
                        annotation_font_size=10)
    ### Plot cummultaive distribution of intrincsict equity value
    fig_cdf = ecdf_plotter_plotly(data=df_intc_valuation,
                              colmn_name ='equity_valuation',
                              xlabel ='Market Cap',
                              title='Intrinsic Equity Value Cumulative Distribution',
                              percentile=[15,50,85],
                              color=['green','yellow','red'],
                              marginal='histogram',
                              height=510,
                              width=720)
    fig_cdf = fig_cdf.add_vline(x = current_market_cap, line_dash = 'dash',line_color='black',annotation_text="-Current Market Cap", annotation_font_size=10)
    ### Model Correlation Chart
    df_intc_valuation = df_intc_valuation.apply(pd.to_numeric, errors='coerce')
    fig_model_correlation_chart = px.bar(df_intc_valuation.rename(columns=dict(zip(df_intc_valuation.columns,
                                     [c.replace("_"," ") for c in df_intc_valuation.columns]))).corr(method='pearson')[['equity valuation']].sort_values(
                                         "equity valuation",ascending=False).reset_index(),
       x='index',
       y='equity valuation',
       title='Model Variable Pearson Correlation with Equity Intrinsic Value',
        height=730,
       width=1600,
       text_auto='.2f',
       labels={'index':'Model Variable',
               'equity valuation':'Correlation'})
    #### Return on Investment
    df_returns = return_values_from_list_extractor_step2(df_intc_valuation,
                                            cum_ret_col = ['cumWACC', 'cumCostOfEquity','cum_acceptable_annualized_return_on_equity','cum_expected_annualized_return_on_equity',
                                                           'cum_excess_annualized_return_on_equity','cum_excess_annualized_return_on_equity_realized'],
                                            ret_col=['excess_annualized_return_on_equity'])

    ### Plot returns
    fig_return = line_plotter_with_error_bound(df_data = df_returns,
                                  x='year',
                                  list_of_mid_point = ['cum_expected_annualized_return_on_equity 50%','cumCostOfEquity 50%'],
                                  list_of_lower_bound = ['cum_expected_annualized_return_on_equity min','cumCostOfEquity min'],
                                  list_of_upper_bound = ['cum_expected_annualized_return_on_equity max','cumCostOfEquity max'],
                                list_of_bar=['excess_annualized_return_on_equity 50%'],
                                  list_of_name=['Cum Expected Return','Cost of Equity'],
                                  list_of_fillcolor= ['rgba(59, 237, 157, 0.5)','rgba(255,84,167, 0.3)'],
                                  list_of_line_color= ['rgb(37, 162, 111)','rgb(238, 72, 103)'],
                                  title='Return on Equity Investment',
                                  yaxis_title='Cum Return',
                                  height=500,
                                  width=1400)
    #### ROIC and Invested Capital
    df_roic = return_values_from_list_extractor_step2(df_data = df_intc_valuation,
                                            cum_ret_col = [],
                                            ret_col = ['ROIC','invested_capital'])[1:]

    fig_roic_inv = plotly_line_dash_bar_chart(df_roic,
                       x='year',
                       ybar=['invested_capital 50%'],
                       yline=['ROIC 50%'],
                       ydash=[],
                       height=500,
                       width=1600,
                       rangemode=None,
                       title='Simulated Median ROIC and Invested Capital',
                       barmode='group',
                       #texttemplate= "%{value}"
                       ).update_layout(hovermode="x")


    fig_model_correlation_chart.show()
    fig_roic_inv.show()
    for col in ['revenue_growth_rate_cycle1_begin',
                'revenue_growth_rate_cycle1_end',
                'revenue_growth_rate_cycle2_begin',
                'revenue_growth_rate_cycle2_end',
                'revenue_growth_rate_cycle3_begin',
                'revenue_growth_rate_cycle3_end'
                'revenue_growth_rate',
                'risk_free_rate','ERP',
                'additional_return_on_cost_of_capital_in_perpetuity',
                'terminal_reinvestmentRate',
                'terminal_afterTaxOperatingIncome',
                'firm_valuation',
                'terminal_FCFF',
                'unlevered_beta',
                'terminal_unlevered_beta',
                'current_sales_to_capital_ratio',
                'terminal_sales_to_capital_ratio',
                'current_operating_margin',
                'terminal_operating_margin',
                'terminal_revenue']:
                try:
                    histogram_plotter_plotly(df_intc_valuation,
                                             colmn_name=col,
                                             xlabel=col.replace("_"," "),
                                             bins=200).show()
                except:
                    pass
    fig_cdf.show()
    fig.show()
    fig_return.show()
    return(df_valuation_res)


def convert_currency_yahoofin(src, dst, amount):
    symbol = f"{src}{dst}=X"
    if src == dst:
        return amount
    try:
        latest_data = si.get_data(symbol, interval="1m", start_date=datetime.now() - timedelta(days=2))
        latest_data.fillna(method='ffill', inplace=True)
        if latest_data.empty or pd.isna(latest_data.iloc[-1].close):
            raise ValueError("Data retrieval was unsuccessful or incomplete.")
        last_updated_datetime = latest_data.index[-1].to_pydatetime()
        latest_price = latest_data.iloc[-1].close
        return float(latest_price) * float(amount)
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def get_sector_yfinance(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    return info.get('sector', None)

def get_industry_yfinance(ticker_symbol):
    ticker = yf.Ticker(ticker_symbol)
    info = ticker.info
    return info.get('industry', None)

def get_industry(ticker_symbol):
    ticker = Ticker(ticker_symbol)
    summary_profile = ticker.summary_profile[ticker_symbol]
    if 'industry' in summary_profile:
        return summary_profile['industry']
    return None

def get_stocks_from_same_industry(ticker_symbol):
    sector = get_industry_yfinance(ticker_symbol)
    print(sector)
    if not sector:
        print(f"Could not find industry for {ticker_symbol}")
        return None
    normalized_sector = re.sub(r'[^a-zA-Z0-9]+', '_', sector).lower()
    s = Screener()
    print(f"Normalized Industry: {normalized_sector}")
    if normalized_sector not in s.available_screeners:
        print(f"No predefined screener available for sector: {normalized_sector}")
        return None
    data = s.get_screeners(normalized_sector)
    print(data)
    df = pd.DataFrame(data[normalized_sector]['quotes'])
    return df

def calculate_rolling_beta(stock_data, market_data, window_size):
    stock_returns = stock_data['Close'].pct_change().dropna()
    market_returns = market_data['Close'].pct_change().dropna()
    rolling_cov = stock_returns.rolling(window=window_size).cov(market_returns)
    rolling_var = market_returns.rolling(window=window_size).var()
    rolling_beta = rolling_cov / rolling_var
    return rolling_beta.dropna()

def get_unlevered_beta(TICKER):
    api_key = "435820a51119704ed53f7e7fb8a0cfec"
    test = f"https://financialmodelingprep.com/api/v3/quote/{TICKER}?apikey={api_key}"
    test2 = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{TICKER}?period=quarter&apikey={api_key}"
    test3 = f"https://financialmodelingprep.com/api/v3/profile/{TICKER}?apikey={api_key}"
    test4 = f"https://financialmodelingprep.com/api/v3/income-statement/{TICKER}?period=quarter&apikey={api_key}"
    response = requests.get(test)
    response2 = requests.get(test2)
    response3 = requests.get(test3)
    response4 = requests.get(test4)
    data = response.json()
    data2 = response2.json()
    data3 = response3.json()
    data4 = response4.json()
    if not data4:
        return None
    marketcap = data[0].get('marketCap')
    sharesOutstanding = data[0].get('sharesOutstanding')
    currency = data2[0].get('reportedCurrency')
    levered_beta = data3[0].get('beta')
    market_cap = convert_currency_yahoofin('USD', currency, marketcap)
    debt_value = data2[0].get('totalDebt')
    equity_value = market_cap
    pretax_income = sum(item['incomeBeforeTax'] for item in data4[:4])
    income_tax_expense = sum(item['incomeTaxExpense'] for item in data4[:4])
    if pretax_income == 0:
        effective_tax_rate = 0
    else:
        effective_tax_rate = income_tax_expense / pretax_income
    T = min(max(effective_tax_rate, 0), 1)
    if pd.isna(debt_value) or debt_value == 0:
        return levered_beta
    else:
        X = debt_value / equity_value
    return levered_beta / (1 + ((1 - T) * X))

def get_pretax_cost_of_debt(ticker):
    api_key = "435820a51119704ed53f7e7fb8a0cfec"
    base_url = "https://financialmodelingprep.com/api/v3"
    balance_sheet_url = f"{base_url}/balance-sheet-statement/{ticker}?period=quarter&apikey={api_key}"
    income_statement_url = f"{base_url}/income-statement/{ticker}?period=quarter&apikey={api_key}"
    try:
        balance_sheet_response = requests.get(balance_sheet_url)
        income_statement_response = requests.get(income_statement_url)
        balance_sheet_response.raise_for_status()
        income_statement_response.raise_for_status()
    except requests.RequestException as e:
        print(f"Error fetching data: {e}")
        return None
    balance_sheet_data = balance_sheet_response.json()
    income_statement_data = income_statement_response.json()
    if not income_statement_data:
        return None
    interest_expense = sum(item.get('interestExpense', 0) for item in income_statement_data[:4])
    total_debt = balance_sheet_data[0].get('totalDebt', 0)
    if total_debt == 0 or interest_expense == 0:
        return 0
    cost_of_debt = interest_expense / total_debt
    if cost_of_debt > 1.0:
        return 0
    return cost_of_debt

def get_year_cost_of_debt_converges(ticker, comparable_tickers):
    current_pretax_cost_of_debt = get_pretax_cost_of_debt(ticker)
    pretax_costs_of_debt = [get_pretax_cost_of_debt(ticker) for ticker in comparable_tickers]
    valid_cost = [cost for cost in pretax_costs_of_debt if cost is not None and not math.isnan(cost)]
    industry_average_pretax_cost_of_debt = sum(valid_cost) / len(valid_cost)
    omega = 0.5
    terminal_pretax_cost_of_debt = omega * current_pretax_cost_of_debt + (1 - omega) * industry_average_pretax_cost_of_debt
    if terminal_pretax_cost_of_debt == current_pretax_cost_of_debt:
        return 0
    annual_change = terminal_pretax_cost_of_debt - current_pretax_cost_of_debt
    years_until_convergence = (terminal_pretax_cost_of_debt - current_pretax_cost_of_debt) / annual_change
    return years_until_convergence

def fetch_growth_estimate(ticker, current_revenue, replacement_value=0.05):
    api_key = "435820a51119704ed53f7e7fb8a0cfec"
    test1 = f"https://financialmodelingprep.com/api/v3/analyst-estimates/{ticker}?apikey={api_key}"
    response1 = requests.get(test1)
    data1 = response1.json()
    current_date = datetime.now().strftime("%Y-%m-%d")
    filtered_data = [entry for entry in data1 if entry['date'] > current_date]
    filtered_data.sort(key=lambda x: x['date'])
    closest_entry = filtered_data[0]
    estimated_revenue_avg = closest_entry['estimatedRevenueAvg'] / 10**9
    growth1 = (estimated_revenue_avg - current_revenue) / current_revenue
    growth1 = growth1 * 100
    growth5 = (replacement_value + (growth1 * 2)) / 3
    return growth1, growth5

def estimate_cycle_length(growth1, growth5):
    growth_rate_change_per_year = (growth5 - growth1) / 4
    if growth_rate_change_per_year == 0:
        return 5
    years = (growth5 - growth1) / growth_rate_change_per_year
    return int(years)

def get_sales_to_capital_ratio(TICKER):
    api_key = "435820a51119704ed53f7e7fb8a0cfec"
    test2 = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{TICKER}?period=quarter&apikey={api_key}"
    test4 = f"https://financialmodelingprep.com/api/v3/income-statement/{TICKER}?period=quarter&apikey={api_key}"
    response2 = requests.get(test2)
    response4 = requests.get(test4)
    data2 = response2.json()
    data4 = response4.json()
    sales = sum(item['revenue'] for item in data4[:4])
    invested_capital = data2[0].get('totalStockholdersEquity') + data2[0].get('totalDebt')
    sales_to_capital_ratio = sales / invested_capital
    return sales_to_capital_ratio

def estimate_terminal_ratio_from_comparables(target_ticker, comparable_tickers):
    ratios = []
    for ticker in comparable_tickers:
        try:
            ratio = get_sales_to_capital_ratio(ticker)
            if not np.isnan(ratio):
                ratios.append(ratio)
        except Exception as e:
            print(f"Could not fetch data for {ticker} due to {e}")
            continue
    if not ratios:
        print("No valid data available for any of the tickers.")
        return np.nan
    terminal_ratio = np.median(ratios)
    return terminal_ratio

def years_to_converge(current_ratio, terminal_ratio, threshold_percentage=0.05):
    years = 0
    convergence_limit = 1e-9
    while abs(current_ratio - terminal_ratio) > convergence_limit:
        years += 1
        if current_ratio < terminal_ratio:
            current_ratio += threshold_percentage * (terminal_ratio - current_ratio)
        else:
            current_ratio -= threshold_percentage * (current_ratio - terminal_ratio)
    if years > 10:
        years = 5
    return years

def get_current_operating_margin(TICKER):
    try:
        api_key = "435820a51119704ed53f7e7fb8a0cfec"
        test4 = f"https://financialmodelingprep.com/api/v3/income-statement/{TICKER}?period=quarter&apikey={api_key}"
        print(test4)
        response4 = requests.get(test4)
        data4 = response4.json()
        operating_income = data4[0].get('operatingIncome')
        if operating_income == 0:
            operating_income = data4[0].get('netIncome')
        total_revenue = data4[0].get('revenue')
        return operating_income / total_revenue
    except Exception as e:
        print(f"Error fetching data for {TICKER}: {e}")
        return 0

def estimate_terminal_operating_margin(comparable_tickers):
    margins = []
    for ticker in comparable_tickers:
        try:
            margin = get_current_operating_margin(ticker)
            if margin is not None and not np.isnan(margin) and margin >= 0:
                margins.append(margin)
        except Exception as e:
            print(f"Couldn't fetch data for {ticker} due to {e}. Skipping...")
    if not margins:
        raise ValueError("Could not fetch data for any comparables or all fetched margins were negative")
    return sum(margins) / len(margins)

def year_margin_begins_to_converge(current_operating_margin, terminal_operating_margin, threshold=0.05):
    year = 0
    while abs(current_operating_margin - terminal_operating_margin) > threshold:
        current_operating_margin = (current_operating_margin + terminal_operating_margin) / 2
        year += 1
        if year > 100:
            raise ValueError("Convergence taking too long. Check the values and threshold.")
    if year == 0:
        return 1
    if year > 10:
        return 5
    return year

def calculate_upside_downside(intrinsic_value, current_cap):
    if current_cap <= 0:
        raise ValueError("Current market cap must be greater than 0.")
    difference = intrinsic_value - current_cap
    percentage = (difference / current_cap)
    return percentage

def calculate_cagr(PV, FV, n):
    if PV == 0:
        raise ValueError("Initial value (PV) cannot be zero.")
    ratio = float(FV) / float(PV)
    cagr_complex = (ratio ** (1.0 / float(n))) - 1
    cagr = cagr_complex.real
    if FV < PV and cagr > 0:
        return -cagr
    return cagr

ENDPOINT = "https://query2.finance.yahoo.com/v8/finance/chart/{}"
TICKER_SP500 = "^GSPC"
DURATION = 10
TODAY = int(datetime.now().timestamp())
TEN_YEARS_AGO = int((datetime.now() - pd.DateOffset(years=DURATION)).timestamp())
urlRFR = "https://query1.finance.yahoo.com/v7/finance/download/%5ETNX?period1=0&period2=9999999999&interval=1d&events=history&includeAdjustedClose=true"
api_key = "435820a51119704ed53f7e7fb8a0cfec"
headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/91.0.4472.124 Safari/537.36'
}
today = datetime.today().strftime('%Y-%m-%d')
one_year_ago = (datetime.today() - timedelta(days=365)).strftime('%Y-%m-%d')
urlRFR = f"https://financialmodelingprep.com/api/v4/treasury?from={one_year_ago}&to={today}&apikey={api_key}"
responseRFR = requests.get(urlRFR, headers=headers)
if responseRFR.status_code == 200:
    data = responseRFR.json()
    if data:
        last_valid_entry = data[-1]
        RFR = last_valid_entry.get('year10')
        if RFR:
            print(f"The 10-year treasury yield in the USA is {RFR}%")
        else:
            print("Error: No valid 10-year Treasury yield found in the latest entry.")
    else:
        print("Error: No data found in the response.")
else:
    print(f"Error: The request failed with status code {responseRFR.status_code}. Response: {responseRFR.text}")

urlSP500 = ENDPOINT.format(TICKER_SP500) + f"?period1={TEN_YEARS_AGO}&period2={TODAY}&interval=1d&events=history"
responseSP500 = requests.get(urlSP500, headers=headers)
if responseSP500.status_code != 200:
    raise Exception("Error fetching S&P 500 data.")
dataSP500_response = responseSP500.json()
if 'chart' in dataSP500_response and 'result' in dataSP500_response['chart'] and len(dataSP500_response['chart']['result']) > 0:
    result = dataSP500_response['chart']['result'][0]
    timestamps = result['timestamp']
    indicators = result['indicators']['quote'][0]
    dataSP500 = pd.DataFrame({
        'Date': pd.to_datetime(timestamps, unit='s'),
        'Open': indicators['open'],
        'High': indicators['high'],
        'Low': indicators['low'],
        'Close': indicators['close'],
        'Volume': indicators['volume'],
    })
    dataSP500.set_index('Date', inplace=True)
    dataSP500.dropna(inplace=True)
    print(dataSP500)
else:
    print("Error: No data found in the response.")

# The following functions that are referenced below must already be defined:
# valuator_multi_phase and point_estimate_describer are not defined here. 
# Keep them as references and do not remove.

def process_ticker(TICKER, excel_path):
    global dataSP500
    api_key = "435820a51119704ed53f7e7fb8a0cfec"
    test = f"https://financialmodelingprep.com/api/v3/quote/{TICKER}?apikey={api_key}"
    test2 = f"https://financialmodelingprep.com/api/v3/balance-sheet-statement/{TICKER}?period=annual&apikey={api_key}"
    test3 = f"https://financialmodelingprep.com/api/v3/profile/{TICKER}?apikey={api_key}"
    test4 = f"https://financialmodelingprep.com/api/v3/income-statement/{TICKER}?period=quarter&apikey={api_key}"
    print(test4)
    test5 = f"https://financialmodelingprep.com/api/v3/income-statement/{TICKER}?period=annual&apikey={api_key}"
    response = requests.get(test)
    response2 = requests.get(test2)
    response3 = requests.get(test3)
    response4 = requests.get(test4)
    response5 = requests.get(test5)
    data = response.json()
    print(data)
    data2 = response2.json()
    print(data2)
    data3 = response3.json()
    data4 = response4.json()
    print(data4)
    data5 = response5.json()
    marketcap = data[0].get('marketCap')
    sharesOutstanding = data[0].get('sharesOutstanding')
    current_price = data[0].get('price')
    currency = data2[0].get('reportedCurrency')
    total_revenue = sum(item['revenue'] for item in data4[:4])
    urlCOMPANY = ENDPOINT.format(TICKER) + f"?period1={TEN_YEARS_AGO}&period2={TODAY}&interval=1d&events=history&includeAdjustedClose=true"
    print(urlCOMPANY)
    responseCOMPANY = requests.get(urlCOMPANY, headers=headers)
    if responseCOMPANY.status_code != 200:
        raise Exception("Error fetching company data.")
    data_json = responseCOMPANY.json()
    if "chart" in data_json and "result" in data_json["chart"]:
        result = data_json["chart"]["result"][0]
        timestamps = result["timestamp"]
        indicators = result.get("indicators", {})
        close_prices = indicators.get("quote", [{}])[0].get("close", [])
        dataCOMPANY = pd.DataFrame({
            "Date": pd.to_datetime(timestamps, unit='s'),
            "Close": close_prices
        }).set_index("Date")
        dataCOMPANY.dropna(inplace=True)
        historical_beta = calculate_rolling_beta(dataCOMPANY, dataSP500, DURATION)
    else:
        raise Exception("Data structure does not match expected format.")
    initial_value = dataSP500['Close'].iloc[0]
    final_value = dataSP500['Close'].iloc[-1]
    Rm = ((final_value / initial_value) ** (1 / DURATION) - 1)
    risk_free_rate1 = RFR / 100
    ERP1 = Rm - risk_free_rate1
    print(f"Equity Risk Premium: {ERP1 * 100:.2f}%")
    market_cap = convert_currency_yahoofin('USD', currency, marketcap) / 10**9
    equity_value1 = market_cap
    print(f"The equity value (market cap) of {TICKER} is approximately ${market_cap:.2f} billion.")
    total_debt = data2[0].get('totalDebt')
    if pd.isna(total_debt) or total_debt == 0:
        debt_value1 = 0
    else:
        debt_value1 = total_debt / 10**9
    print(f"The total debt of {TICKER} is approximately ${debt_value1:.2f} billion.")
    cash_and_cash_equivalents = data2[0].get('cashAndCashEquivalents')
    Other_Non_Current_Assets = data2[0].get('otherNonCurrentAssets')
    intangibleAssets = data2[0].get('intangibleAssets')
    goodwill = data2[0].get('goodwill')
    Investments_and_Advances = data2[0].get('longTermInvestments')
    cash_and_non_operating_asset1 = np.nansum([
        cash_and_cash_equivalents,
        Other_Non_Current_Assets,
        Investments_and_Advances,
        goodwill,
        intangibleAssets
    ]) / 10**9
    print(f"Cash and non-operating assets of {TICKER} is approximately ${cash_and_non_operating_asset1:.2f} billion.")
    df_result = get_stocks_from_same_industry(TICKER)
    comparable_tickers = df_result['symbol'].tolist()
    unlevered_betas = [get_unlevered_beta(ticker) for ticker in comparable_tickers]
    unlevered_betas = [beta for beta in unlevered_betas if beta is not None]
    valid_betas = [beta for beta in unlevered_betas if not math.isnan(beta)]
    industry_average_unlevered_beta = sum(valid_betas) / len(valid_betas)
    omega = 0.5
    unlevered_beta1 = get_unlevered_beta(TICKER)
    if unlevered_beta1 is None:
        unlevered_beta1 = industry_average_unlevered_beta
    terminal_unlevered_beta1 = omega * unlevered_beta1 + (1 - omega) * industry_average_unlevered_beta
    print(f"The estimated unlevered beta is: {unlevered_beta1}")
    print(f"The estimated terminal unlevered beta is: {terminal_unlevered_beta1}")
    X = np.array(range(len(historical_beta))).reshape(-1, 1)
    y = historical_beta.values
    model = LinearRegression().fit(X, y)
    slope = model.coef_[0]
    intercept = model.intercept_
    intersection_point = (terminal_unlevered_beta1 - intercept) / slope
    intersection_in_years = intersection_point
    year_beta_begins_to_converge_to_terminal_beta1 = 1
    pretax_income = sum(item['incomeBeforeTax'] for item in data4[:4])
    pretax_income2 = data5[0].get('incomeBeforeTax')
    income_tax_expense = sum(item['incomeTaxExpense'] for item in data4[:4])
    income_tax_expense2 = data5[0].get('incomeTaxExpense')
    tax_rate = 0.25
    print(f"Current Effective Tax Rate: {tax_rate*100:.2f}%")
    current_effective_tax_rate1 = tax_rate
    current_pretax_cost_of_debt1 = get_pretax_cost_of_debt(TICKER)
    print(f"Current Pretax Cost of Debt: {current_pretax_cost_of_debt1*100:.2f}%")
    pretax_costs_of_debt = [get_pretax_cost_of_debt(ticker) for ticker in comparable_tickers]
    pretax_costs_of_debt = [cost for cost in pretax_costs_of_debt if cost is not None]
    valid_cost = [cost for cost in pretax_costs_of_debt if not math.isnan(cost) and cost >= 0]
    industry_average_pretax_cost_of_debt = sum(valid_cost) / len(valid_cost)
    omega = 0.5
    terminal_pretax_cost_of_debt1 = omega * current_pretax_cost_of_debt1 + (1 - omega) * industry_average_pretax_cost_of_debt
    print(f"The estimated terminal pre-tax cost of debt is: {terminal_pretax_cost_of_debt1*100:.2f}%")
    year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt1 = 1
    marginal_tax_rate1 = 0.25
    year_effective_tax_rate_begin_to_converge_marginal_tax_rate1 = 1
    revenue_base1 = total_revenue / 10**9
    print(f"The total revenue of {TICKER} is approximately ${revenue_base1:.2f} billion")
    growth1, growth5 = fetch_growth_estimate(TICKER, revenue_base1, ERP1*100)
    revenue_growth_rate_cycle1_begin1 = growth1/100
    revenue_growth_rate_cycle1_end1 = growth5/100
    length_of_cylcle1_1 = 1
    revenue_growth_rate_cycle2_begin1 = (revenue_growth_rate_cycle1_begin1 + ERP1 + revenue_growth_rate_cycle1_begin1)/3
    revenue_growth_rate_cycle2_end1 = (revenue_growth_rate_cycle1_begin1 + ERP1 + revenue_growth_rate_cycle1_begin1 + ERP1)/4
    length_of_cylcle2_1 = estimate_cycle_length(revenue_growth_rate_cycle2_begin1, revenue_growth_rate_cycle2_end1)
    revenue_growth_rate_cycle3_begin1 = (revenue_growth_rate_cycle2_begin1 + ERP1 + ERP1)/3
    revenue_growth_rate_cycle3_end1 = ERP1
    length_of_cylcle3_1 = 1
    revenue_convergance_periods_cycle1_1 = 1
    revenue_convergance_periods_cycle2_1 = 1
    revenue_convergance_periods_cycle3_1 = 1
    current_sales_to_capital_ratio1 = get_sales_to_capital_ratio(TICKER)
    print(f"the current sales to capital ratio is {current_sales_to_capital_ratio1}")
    terminal_sales_to_capital_ratio1 = (estimate_terminal_ratio_from_comparables(TICKER, comparable_tickers)
                                        + get_sales_to_capital_ratio(TICKER)
                                        + get_sales_to_capital_ratio(TICKER))/3
    print(f"the terminal sales to capital ratio is {terminal_sales_to_capital_ratio1}")
    year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital1 = 1
    current_operating_margin1 = get_current_operating_margin(TICKER)
    terminal_operating_margin1 = (estimate_terminal_operating_margin(comparable_tickers)*0.2) + (get_current_operating_margin(TICKER)*0.8)
    year_operating_margin_begins_to_converge_to_terminal_operating_margin1 = 1
    additional_return_on_cost_of_capital_in_perpetuity1 = 0.02
    asset_liquidation_during_negative_growth1 = 0
    current_invested_capital = revenue_base1 / current_sales_to_capital_ratio1

    # The following functions (valuator_multi_phase and point_estimate_describer) are not defined here
    # They must be defined elsewhere. We are keeping them as is.
    base_case_valuation = valuator_multi_phase(
        risk_free_rate=risk_free_rate1,
        ERP=ERP1,
        equity_value=equity_value1,
        debt_value=debt_value1,
        cash_and_non_operating_asset=cash_and_non_operating_asset1,
        unlevered_beta=unlevered_beta1,
        terminal_unlevered_beta=terminal_unlevered_beta1,
        year_beta_begins_to_converge_to_terminal_beta=year_beta_begins_to_converge_to_terminal_beta1,
        current_pretax_cost_of_debt=current_pretax_cost_of_debt1,
        terminal_pretax_cost_of_debt=terminal_pretax_cost_of_debt1,
        year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt=year_cost_of_debt_begins_to_converge_to_terminal_cost_of_debt1,
        current_effective_tax_rate=current_effective_tax_rate1,
        marginal_tax_rate=marginal_tax_rate1,
        year_effective_tax_rate_begin_to_converge_marginal_tax_rate=year_effective_tax_rate_begin_to_converge_marginal_tax_rate1,
        revenue_base=revenue_base1,
        revenue_growth_rate_cycle1_begin=revenue_growth_rate_cycle1_begin1,
        revenue_growth_rate_cycle1_end=revenue_growth_rate_cycle1_end1,
        length_of_cylcle1=length_of_cylcle1_1,
        revenue_growth_rate_cycle2_begin=revenue_growth_rate_cycle2_begin1,
        revenue_growth_rate_cycle2_end=revenue_growth_rate_cycle2_end1,
        length_of_cylcle2=length_of_cylcle2_1,
        revenue_growth_rate_cycle3_begin=revenue_growth_rate_cycle3_begin1,
        revenue_growth_rate_cycle3_end=revenue_growth_rate_cycle3_end1,
        length_of_cylcle3=length_of_cylcle3_1,
        revenue_convergance_periods_cycle1=revenue_convergance_periods_cycle1_1,
        revenue_convergance_periods_cycle2=revenue_convergance_periods_cycle2_1,
        revenue_convergance_periods_cycle3=revenue_convergance_periods_cycle3_1,
        current_sales_to_capital_ratio=current_sales_to_capital_ratio1,
        terminal_sales_to_capital_ratio=terminal_sales_to_capital_ratio1,
        year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital=year_sales_to_capital_begins_to_converge_to_terminal_sales_to_capital1,
        current_operating_margin=current_operating_margin1,
        terminal_operating_margin=terminal_operating_margin1,
        year_operating_margin_begins_to_converge_to_terminal_operating_margin=year_operating_margin_begins_to_converge_to_terminal_operating_margin1,
        additional_return_on_cost_of_capital_in_perpetuity=additional_return_on_cost_of_capital_in_perpetuity1,
        asset_liquidation_during_negative_growth=asset_liquidation_during_negative_growth1
    )

    df_valuation = point_estimate_describer(base_case_valuation)
    intrinsic_equity_value = convert_currency_yahoofin(currency, 'USD', base_case_valuation['equity_value'])
    cap2 = marketcap / 10**9
    print(cap2)
    cagr = calculate_upside_downside(intrinsic_equity_value, cap2)
    print(f"The UPSIDE/DOWNSIDE is {cagr:.2%}")
    if os.path.exists(excel_path):
        results_df = pd.read_excel(excel_path)
    else:
        results_df = pd.DataFrame(columns=['Ticker', 'Upside/Downside'])
    new_row = pd.DataFrame({'Ticker': [TICKER], 'Upside/Downside': [cagr]})
    results_df = pd.concat([results_df, new_row], ignore_index=True)
    results_df.to_excel(excel_path, index=False)
    excel_path = f"dcf_{TICKER}.xlsx"
    df_valuation.to_excel(excel_path)
    number_of_share = sharesOutstanding / 10**9
    print(sharesOutstanding)
    real_value = intrinsic_equity_value/number_of_share
    int_value = convert_currency_yahoofin(currency,'USD', real_value)
    print(f"$$Real value per share$$ : ${real_value}")

# Read the CSV file
ticker_df = pd.read_csv('beta.csv')
tickers = ticker_df['Symbol'].drop_duplicates().tolist()

excel_path = 'results.xlsx'

# Ask the user for their choice
user_choice = input("Enter 'all' to process all stocks, or enter a Ticker symbol to process only that stock: ")

if user_choice.lower() == 'all':
    # Iterate over each ticker and call the function for all stocks
    for ticker in tickers:
        try:
            process_ticker(ticker, excel_path)
            time.sleep(30)  # Normal delay between successful requests
        except Exception as e:
            print(f"An error occurred while processing {ticker}: {e}")
            traceback.print_exc()
            time.sleep(60)
else:
    # Process only the specified ticker
    if user_choice != 'all':
        try:
            process_ticker(user_choice, excel_path)
        except Exception as e:
            print(f"An error occurred while processing {user_choice}: {e}")
            traceback.print_exc()
    else:
        print(f"Ticker '{user_choice}' not found in the list.")
