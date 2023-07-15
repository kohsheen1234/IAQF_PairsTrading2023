import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn import linear_model

from data_folding import DataFolder


def reg(x: np.array, y: np.array):
    """
    Regression between x and y.
    y = beta @ x + intercept

    :param x: explanatory variable, shape: (N, 1)
    :param y: response variable, shape: (N, 1)
    :return: 1. spread/residual, y - beta @ x
             2. beta
    """
    regression_model = linear_model.LinearRegression()
    regression_model.fit(x, y)
    beta = regression_model.coef_[0]
    spread = y - x * beta
    return spread, beta[0]


def log_prices(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculates the log-prices given price.

    :param df: Dataframe with prices
    :return: Dataframe with log returns.
    """
    log_returns_df = (np.log(df))
    return log_returns_df


class FixedThresholdPairsTrader:
    """
    Creates a fixed spread on the full data. Constructs fixed confidence intervals based on this spread.

    Spread is defined as asset2 - asset1.
    Thus, if spread > upper_ci, it means we ought to short asset2 and long asset1.
    And, if spread < lower_ci, it means we ought to long asset2 and short asset1.
    """

    def __init__(self, assets: pd.DataFrame):
        self._asset1 = assets.iloc[:, 0]
        self._asset2 = assets.iloc[:, 1]
        self._asset1_name = assets.columns[0]
        self._asset2_name = assets.columns[1]

    def calculate_spread(self):
        log_prices_asset1, log_prices_asset2 = log_prices(self._asset1), log_prices(self._asset2)
        X, Y = log_prices_asset1.values.reshape(-1, 1), log_prices_asset2.values.reshape(-1, 1)
        spread, _ = reg(X, Y)
        return spread

    def backtest(
            self,
            use_hedge_ratio: bool = True,
            entry_threshold: float = 2.0,
            exit_threshold: float = 0.5
    ) -> pd.DataFrame:
        log_prices_asset1, log_prices_asset2 = log_prices(self._asset1), log_prices(self._asset2)

        X = log_prices_asset1.values.reshape(-1, 1)
        Y = log_prices_asset2.values.reshape(-1, 1)

        spread, beta = reg(X, Y)
        spread_df = pd.DataFrame(spread, columns=["spread"], index=log_prices_asset1.index)

        timestamps = spread_df.index
        positions = []
        entered_short = False
        entered_long = False

        entry_upper_ci = spread_df["spread"].mean() + entry_threshold * spread_df["spread"].std()
        entry_lower_ci = spread_df["spread"].mean() - entry_threshold * spread_df["spread"].std()

        exit_upper_ci = spread_df["spread"].mean() + exit_threshold * spread_df["spread"].std()
        exit_lower_ci = spread_df["spread"].mean() - exit_threshold * spread_df["spread"].std()

        for time in timestamps:
            # Get current price of the spread
            spread_price_t = spread_df["spread"].loc[time]

            # ALLOCATION:
            # If spread crosses the upper-bound, and the position is not
            if entered_long:
                if spread_price_t > exit_lower_ci:
                    position = np.array([0.0, 0.0])
                    entered_long = False
                else:
                    position = np.array([-1.0, 1.0]) if not use_hedge_ratio else np.array([-beta, 1.0])

            elif entered_short:
                if spread_price_t < exit_upper_ci:
                    position = np.array([0.0, 0.0])
                    entered_short = False
                else:
                    position = np.array([1.0, -1.0]) if not use_hedge_ratio else np.array([beta, -1.0])

            else:
                if spread_price_t >= entry_upper_ci:
                    position = np.array([1.0, -1.0]) if not use_hedge_ratio else np.array([beta, -1.0])
                elif spread_price_t <= entry_lower_ci:
                    position = np.array([-1.0, 1.0]) if not use_hedge_ratio else np.array([-beta, 1.0])
                else:
                    position = np.array([0.0, 0.0])

            positions.append(position)

        positions_np = np.stack(positions)
        positions_df = pd.DataFrame(positions_np, index=timestamps, columns=[self._asset1_name, self._asset2_name])
        return positions_df

    def backtest_actions(self, entry_threshold: float = 2.0, exit_threshold: float = 0.5):
        log_prices_asset1, log_prices_asset2 = log_prices(self._asset1), log_prices(self._asset2)

        X = log_prices_asset1.values.reshape(-1, 1)
        Y = log_prices_asset2.values.reshape(-1, 1)

        spread, beta = reg(X, Y)
        spread_df = pd.DataFrame(spread, columns=["spread"], index=log_prices_asset1.index)

        upper_ci = spread_df["spread"].mean() + entry_threshold * spread_df["spread"].std()
        lower_ci = spread_df["spread"].mean() - entry_threshold * spread_df["spread"].std()

        upper_ci_inner = spread_df["spread"].mean() + exit_threshold * spread_df["spread"].std()
        lower_ci_inner = spread_df["spread"].mean() - exit_threshold * spread_df["spread"].std()

        actions = []
        position = 0

        for i in range(spread_df.shape[0]):
            spread_price_t = spread_df.iloc[i, 0]

            if position == 0:
                if spread_price_t > upper_ci:
                    # Our spread is y - x, so if y-x crosses upper ci, it means y became expensive,
                    # so we should be shorting y, and buying x.
                    actions.append([1, -1])
                    position = -1
                elif spread_price_t < lower_ci:
                    actions.append([-1, 1])
                    position = 1
                else:
                    actions.append([0, 0])

            else:
                if position == -1:
                    if spread_price_t < upper_ci_inner:
                        actions.append([-1, 1])
                        position = 0
                    else:
                        actions.append([0, 0])
                else:
                    if spread_price_t > lower_ci_inner:
                        actions.append([1, -1])
                        position = 0
                    else:
                        actions.append([0, 0])

        actions = pd.DataFrame(actions, index=spread_df.index, columns=[self._asset1_name, self._asset2_name])
        return actions

    def pnl_actions(self, actions: pd.DataFrame, future_returns_df: pd.DataFrame):
        # assumptions
        capital = 100
        w = [1, 1]
        portfolio = pd.DataFrame(columns=[self._asset1_name, self._asset2_name], index=actions.index)
        # day 0 balance
        portfolio['Cash'] = 0
        portfolio.iloc[:, :2] = 0

        for i in range(actions.shape[0]):
            if i != 0:
                portfolio.iloc[i, 2] = portfolio.iloc[i - 1, 2]

            if portfolio.iloc[i - 1, 0] == 0:
                # if no stocks in portfolio
                portfolio.iloc[i, 0] = actions.iloc[i, 0] * w[0] * capital
                portfolio.iloc[i, 1] = actions.iloc[i, 1] * w[1] * capital
                portfolio.iloc[i, 2] = portfolio.iloc[i, 2] - portfolio.iloc[i, 0] - portfolio.iloc[i, 1]
            else:
                # if have positions
                # calculate day-end
                portfolio.iloc[i, 0] = portfolio.iloc[i - 1, 0] + (
                        future_returns_df.loc[portfolio.index[i], self._asset1_name]
                ) * portfolio.iloc[i - 1, 0]
                portfolio.iloc[i, 1] = portfolio.iloc[i - 1, 1] + (
                        future_returns_df.loc[portfolio.index[i], self._asset2_name]
                ) * portfolio.iloc[i - 1, 1]

                # if not hold
                if actions.iloc[i, 0] != 0:
                    act0 = actions.iloc[i, 0] * abs(portfolio.iloc[i, 0])
                    act1 = actions.iloc[i, 1] * abs(portfolio.iloc[i, 1])
                    portfolio.iloc[i, 2] = portfolio.iloc[i, 2] - act0 - act1
                    portfolio.iloc[i, 0] = portfolio.iloc[i, 0] + act0
                    portfolio.iloc[i, 1] = portfolio.iloc[i, 1] + act1

        return portfolio


class DynamicThresholdPairsTrader:
    def __init__(self, assets: pd.DataFrame):
        self._asset1 = assets.iloc[:, 0]
        self._asset2 = assets.iloc[:, 1]
        self._asset1_name = assets.columns[0]
        self._asset2_name = assets.columns[1]

    def backtest(
            self,
            formation_period: int,
            testing_period: int,
            use_hedge_ratio: bool = True,
            entry_threshold: float = 1.5,
            exit_threshold: float = 0.5,
    ) -> pd.DataFrame:
        log_returns_asset1, log_returns_asset2 = log_prices(self._asset1), log_prices(self._asset2)

        timestamps = log_returns_asset1.index
        folder = DataFolder(
            timestamps=timestamps,
            formation_period=formation_period,
            test_period=testing_period
        )

        test_times = []
        positions = []

        for fold in folder:
            x_train = log_returns_asset1.loc[fold.formation_set]
            y_train = log_returns_asset2.loc[fold.formation_set]

            # FIND THE SPREAD AND CONFIDENCE INTERVALS IN THE FORMATION PERIOD.
            spread_train, beta_train = reg(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

            entry_upper_ci = spread_train.mean() + entry_threshold * spread_train.std()
            entry_lower_ci = spread_train.mean() - entry_threshold * spread_train.std()

            exit_upper_ci = spread_train.mean() + exit_threshold * spread_train.std()
            exit_lower_ci = spread_train.mean() - exit_threshold * spread_train.std()

            entered_short = False
            entered_long = False

            # Backtesting
            for test_t in fold.test_set:
                # APPLY THE CIs AND COLLECT PORTFOLIOS ON THE TEST PERIOD.
                x_test_t = log_returns_asset1.loc[test_t]
                y_test_t = log_returns_asset2.loc[test_t]

                # Get current price of the spread
                spread_price_t = y_test_t - x_test_t * beta_train

                # Allocation:
                if entered_long:
                    if spread_price_t > exit_lower_ci:
                        position = np.array([0.0, 0.0])
                        entered_long = False
                    else:
                        position = np.array([-1.0, 1.0]) if not use_hedge_ratio else np.array([-beta_train, 1.0])

                elif entered_short:
                    if spread_price_t < exit_upper_ci:
                        position = np.array([0.0, 0.0])
                        entered_short = False
                    else:
                        position = np.array([1.0, -1.0]) if not use_hedge_ratio else np.array([beta_train, -1.0])

                else:
                    if spread_price_t >= entry_upper_ci:
                        position = np.array([1.0, -1.0]) if not use_hedge_ratio else np.array([beta_train, -1.0])

                    elif spread_price_t <= entry_lower_ci:
                        position = np.array([-1.0, 1.0]) if not use_hedge_ratio else np.array([-beta_train, 1.0])

                    else:
                        position = np.array([0.0, 0.0])

                test_times.append(test_t)
                positions.append(position)

            positions[-1] = np.array([0.0, 0.0])

        positions_np = np.stack(positions)
        positions_df = pd.DataFrame(positions_np, index=test_times, columns=[self._asset1_name, self._asset2_name])
        return positions_df

    def backtest_actions(
            self,
            formation_period: int,
            testing_period: int,
            entry_threshold: float = 1.5,
            exit_threshold: float = 0.5,
    ) -> pd.DataFrame:
        log_returns_asset1, log_returns_asset2 = log_prices(self._asset1), log_prices(self._asset2)

        timestamps = log_returns_asset1.index
        folder = DataFolder(
            timestamps=timestamps,
            formation_period=formation_period,
            test_period=testing_period
        )

        test_times = []
        actions = []

        for fold in folder:
            x_train = log_returns_asset1.loc[fold.formation_set]
            y_train = log_returns_asset2.loc[fold.formation_set]

            # FIND THE SPREAD AND CONFIDENCE INTERVALS IN THE FORMATION PERIOD.
            spread_train, beta_train = reg(x_train.values.reshape(-1, 1), y_train.values.reshape(-1, 1))

            upper_ci = spread_train.mean() + entry_threshold * spread_train.std()
            lower_ci = spread_train.mean() - entry_threshold * spread_train.std()

            upper_ci_inner = spread_train.mean() + exit_threshold * spread_train.std()
            lower_ci_inner = spread_train.mean() - exit_threshold * spread_train.std()

            position = 0

            # Backtesting
            for test_t in fold.test_set:
                # APPLY THE CIs AND COLLECT PORTFOLIOS ON THE TEST PERIOD.
                x_test_t = log_returns_asset1.loc[test_t]
                y_test_t = log_returns_asset2.loc[test_t]

                # Get current price of the spread
                spread_price_t = y_test_t - x_test_t * beta_train

                # Allocation:
                if position == 0:
                    if spread_price_t > upper_ci:
                        # Our spread is y - x, so if y-x crosses upper ci, it means y became expensive,
                        # so we should be shorting y, and buying x.
                        actions.append([1, -1])
                        position = -1
                    elif spread_price_t < lower_ci:
                        actions.append([-1, 1])
                        position = 1
                    else:
                        actions.append([0, 0])

                else:
                    if position == -1:
                        if spread_price_t < upper_ci_inner:
                            actions.append([-1, 1])
                            position = 0
                        else:
                            actions.append([0, 0])
                    else:
                        if spread_price_t > lower_ci_inner:
                            actions.append([1, -1])
                            position = 0
                        else:
                            actions.append([0, 0])

                test_times.append(test_t)
            # we liquidate the position after reaching end of current testing period
            if position == 0:
                actions[-1] = [0, 0]
            elif position == 1:
                actions[-1] = [1, -1]
            else:
                actions[-1] = [-1, 1]

        actions_np = np.stack(actions)
        actions_df = pd.DataFrame(actions_np, index=test_times, columns=[self._asset1_name, self._asset2_name])
        return actions_df

    def pnl_actions(self, actions: pd.DataFrame, future_returns_df: pd.DataFrame):
        # assumptions
        capital = 100
        w = [1, 1]
        portfolio = pd.DataFrame(columns=[self._asset1_name, self._asset2_name], index=actions.index)
        # day 0 balance
        portfolio['Cash'] = 0
        portfolio.iloc[:, :2] = 0

        for i in range(actions.shape[0]):
            if i != 0:
                portfolio.iloc[i, 2] = portfolio.iloc[i - 1, 2]

            if portfolio.iloc[i - 1, 0] == 0:
                # if no stocks in portfolio
                portfolio.iloc[i, 0] = actions.iloc[i, 0] * w[0] * capital
                portfolio.iloc[i, 1] = actions.iloc[i, 1] * w[1] * capital
                portfolio.iloc[i, 2] = portfolio.iloc[i, 2] - portfolio.iloc[i, 0] - portfolio.iloc[i, 1]
            else:
                # if have positions
                # calculate day-end
                portfolio.iloc[i, 0] = portfolio.iloc[i - 1, 0] + (
                        future_returns_df.loc[portfolio.index[i], self._asset1_name]
                ) * portfolio.iloc[i - 1, 0]
                portfolio.iloc[i, 1] = portfolio.iloc[i - 1, 1] + (
                        future_returns_df.loc[portfolio.index[i], self._asset2_name]
                ) * portfolio.iloc[i - 1, 1]

                # if not hold
                if actions.iloc[i, 0] != 0:
                    act0 = actions.iloc[i, 0] * abs(portfolio.iloc[i, 0])
                    act1 = actions.iloc[i, 1] * abs(portfolio.iloc[i, 1])
                    portfolio.iloc[i, 2] = portfolio.iloc[i, 2] - act0 - act1
                    portfolio.iloc[i, 0] = portfolio.iloc[i, 0] + act0
                    portfolio.iloc[i, 1] = portfolio.iloc[i, 1] + act1

        return portfolio


if __name__ == "__main__":
    import yfinance as yf

    # downloading SP500 and Russell 2000 index OHLC price datas for a 23-year period
    SP_data = yf.download("^GSPC", start="2000-01-01", end="2023-01-20")
    RS_data = yf.download("^RUT", start="2000-01-01", end="2023-01-20")
    prices_np = np.stack([SP_data["Adj Close"].values, RS_data["Adj Close"].values]).T
    prices_df = pd.DataFrame(index=SP_data.index, data=prices_np, columns=["SP500", "RS2000"])
    forward_returns_df = prices_df.pct_change()

    pairs_trader = FixedThresholdPairsTrader(asset1=prices_df["SP500"], asset2=prices_df["RS2000"])
    portfolio_returns = pairs_trader.get_pnl(forward_returns_df)
    daily_returns = portfolio_returns.sum(axis=1)

    plt.figure()
    plt.plot(np.cumsum([np.log(1 + r) for r in daily_returns]))
    plt.show()

    dynamic_thresh_pairs_trader = DynamicThresholdPairsTrader(asset1=prices_df["SP500"], asset2=prices_df["RS2000"])
    portfolio_returns_dyn_thresh = dynamic_thresh_pairs_trader.get_pnl(forward_returns_df)
    daily_returns = portfolio_returns_dyn_thresh.sum(axis=1)
    plt.figure()
    plt.plot(np.cumprod([(1 + r) for r in daily_returns]))
    plt.show()
