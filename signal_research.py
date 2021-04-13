import datetime
import pandas as pd
import yfinance as yf
from pandas_datareader import data as pdr
import numpy as np
import matplotlib.pyplot as plt
src = "C:/Users/prasa/PycharmProjects/PortfolioResearch/"
start = '2000-12-31'
end = '2019-12-31'
# end = datetime.date.today().strftime('%Y-%m-%d')
yf.pdr_override()


class SignalResearch:
    symbol_df = pd.read_excel(src + 'universe.xlsx', sheet_name='high_low', index_col=[0])
    tickers = symbol_df.index.tolist()
    tickers = tickers[:2]
    # def __init__(self):
    #     self.yahoo_data = pdr.get_data_yahoo(self.tickers, start=start, end=end)

    def get_yahoo_data(self):
        self.ydata = pdr.get_data_yahoo(self.tickers, start=start, end=end)
        return self.ydata

    def drawdown(self, data):
        self.returns = data.fillna(0)
        windx = self.returns.add(1).cumprod()
        prev_peaks = windx.cummax()
        dd = (windx - prev_peaks) / prev_peaks
        return dd.min()

    def nperiod_signal(self, px_data, trade_signal, nper=252, horizon=5):

        returns_dict = {}
        win_rate=[]
        realized_ret = []
        realized_risk = []
        nof_trades = []
        max_ret = []
        min_ret = []
        total_trades = []
        dd_strategy = []
        dd_stocks = []

        for symbol in self.tickers:
            adj_close = px_data['Adj Close', symbol].copy()
            if trade_signal == 'high':
                nper_signal = adj_close.rolling(window=nper).apply(max)
            else:
                nper_signal = adj_close.rolling(window=nper).apply(min)

            adj_start = nper_signal.dropna().index[0]
            trade_df = pd.DataFrame(index=nper_signal.index)
            position = 0
            trade_count = 1
            closing_counter = 0
            for counter in range(len(nper_signal)):
                px = adj_close.iloc[counter]
                hi_lo = nper_signal.iloc[counter]

                if pd.isna(hi_lo):
                    trade_df.loc[trade_df.index[counter], 'Trades'] = 'No Trade'
                    trade_df.loc[trade_df.index[counter], 'Entry'] = 0
                    trade_df.loc[trade_df.index[counter], 'Exit'] = 0
                    trade_df.loc[trade_df.index[counter], 'ClosingPx'] = 0
                    trade_df.loc[trade_df.index[counter], 'Returns'] = 0.0
                    trade_df.loc[trade_df.index[counter], 'HPR'] = 0.0
                    trade_df.loc[trade_df.index[counter], 'Comment'] = 'No Open Trades'
                    position = 0

                elif px == hi_lo and position == 0:
                    trade_df.loc[trade_df.index[counter], 'Trades'] = 'Position Open'
                    buy_px = adj_close.iloc[counter]
                    trade_df.loc[trade_df.index[counter], 'Entry'] = 1
                    trade_df.loc[trade_df.index[counter], 'Exit'] = 0
                    trade_df.loc[trade_df.index[counter], 'ClosingPx'] = px
                    trade_df.loc[trade_df.index[counter], 'Returns'] = 0.0
                    trade_df.loc[trade_df.index[counter], 'HPR'] = 0.0
                    trade_df.loc[trade_df.index[counter], 'Comment'] = f"Opened trade #{trade_count}"
                    closing_counter = counter + horizon
                    position = 1

                elif counter < closing_counter and position == 1:
                    trade_df.loc[trade_df.index[counter], 'ClosingPx'] = px
                    trade_df.loc[trade_df.index[counter], 'Returns'] = adj_close.iloc[counter] \
                                                                       / adj_close.iloc[counter - 1] - 1

                elif counter == closing_counter and position == 1:
                    trade_df.loc[trade_df.index[counter], 'Trades'] = 'Position Closed'
                    trade_df.loc[trade_df.index[counter], 'Exit'] = 1
                    trade_df.loc[trade_df.index[counter], 'ClosingPx'] = px
                    trade_df.loc[trade_df.index[counter], 'Comment'] = f"Closed trade #{trade_count}"
                    trade_df.loc[trade_df.index[counter], 'Returns'] = adj_close.iloc[counter] \
                                                                       / adj_close.iloc[counter - 1] - 1
                    trade_df.loc[trade_df.index[counter], 'HPR'] = adj_close.iloc[counter] / buy_px - 1
                    position = 0
                    buy_px = 0.0
                    trade_count +=1

                else:
                    position == position
                    trade_df.loc[trade_df.index[counter], 'Returns'] = np.nan

            hpr_return = trade_df[trade_df.HPR!=0].HPR.dropna()
            gt_zero = hpr_return>0
            win_rate.append(gt_zero.sum()/gt_zero.count())

            realized_ret.append(trade_df[trade_df.Trades=='Position Closed'].HPR.mean())
            realized_risk.append(trade_df[trade_df.Trades=='Position Closed'].HPR.std())
            nof_trades.append(trade_df[trade_df.Trades=='Position Closed'].HPR.count())
            max_ret.append(trade_df[trade_df.Trades=='Position Closed'].HPR.max())
            min_ret.append(trade_df[trade_df.Trades=='Position Closed'].HPR.min())
            dd_strategy.append(self.drawdown(trade_df.HPR))
            dd_stocks.append(self.drawdown(adj_close.pct_change().fillna(0)))

            returns_dict.update({symbol:trade_df.Returns})
            year = horizon / 250
            ann_ret = [(1+r)**(1/year)-1 for r in realized_ret]
            ann_risk = [s * np.sqrt(1 / year) for s in realized_risk]
            total_trades.append(len(hpr_return))

        arr = np.array([nof_trades, realized_ret, realized_risk ,ann_ret, ann_risk, win_rate, max_ret, min_ret,
                        dd_strategy, dd_stocks])
        dff = pd.DataFrame(arr.T, index=self.tickers, columns=['#Trades', 'MRR','MSD', 'Annualized Return',
                                                               'Annualized Risk', 'Win Ratio', 'Best Return',
                                                               'Worst Return', 'DD_Strategy', 'DD_Asset'])
        dff[['MRR','MSD', 'Annualized Return', 'Annualized Risk', 'Win Ratio', 'Best Return', 'Worst Return',
             'DD_Strategy', 'DD_Asset']] = \
            dff[['MRR','MSD', 'Annualized Return', 'Annualized Risk', 'Win Ratio', 'Best Return', 'Worst Return',
                 'DD_Strategy', 'DD_Asset']].\
                applymap(lambda x: "{:.2%}".format(x))

        dff['Lookback Window/Holding Period'] = f"{nper} days / {horizon} days"

        eqw_portfolio = pd.DataFrame(returns_dict)
        eqw_ann_ret = eqw_portfolio.fillna(0).mean(axis=1).mean() * 250
        eqw_ann_std = eqw_portfolio.fillna(0).mean(axis=1).std() * np.sqrt(250)
        port_cols = ['Annualized Returns', 'Annualized Risk']
        port_df = pd.DataFrame([eqw_ann_ret, eqw_ann_std], columns=[f"{nper} days / {horizon} days"], index = port_cols)
        return dff, port_df

    def run_main(self, tradeSig):
        objSignal = SignalResearch()
        yahoo_data = self.get_yahoo_data()
        comb_df = pd.DataFrame()
        comb_port = pd.DataFrame()
        for win in [90, 180]:
            for horizon in [30, 45, 90]:
                data_frame, portfolio = objSignal.nperiod_signal(yahoo_data, tradeSig, win, horizon)
                comb_df = pd.concat([comb_df, data_frame])
                comb_port = pd.concat([comb_port, portfolio], axis=1)

        grouped_df = comb_df.groupby(['Lookback Window/Holding Period', comb_df.index]).last()
        grouped_df.to_html(src + 'holdings.html')

        comb_port.loc['Annualized Returns', 'Eqw Portfolio'] = yahoo_data['Adj Close'].pct_change().fillna(0).mean(
            axis=1).mean() * 250

        comb_port.loc['Annualized Risk', 'Eqw Portfolio'] = yahoo_data['Adj Close'].pct_change().fillna(0).mean(
            axis=1).std() * np.sqrt(250)

        comb_port = comb_port.T
        comb_port.loc[:, 'Sharpe'] = comb_port.iloc[:, 0] / comb_port.iloc[:, 1]
        comb_port = comb_port.applymap(lambda x: "{:.2%}".format(x))
        comb_port.to_html(src + 'portfolios.html')

        print('test')


if __name__ == "__main__":
    objSignal = SignalResearch()
    objSignal.run_main('high')
