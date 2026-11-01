# pragma: no cover
from datetime import datetime
from pandas import DataFrame
from freqtrade.strategy import IStrategy, Trade, stoploss_from_open
import talib.abstract as ta
import pandas as pd


class RobustZScoreHybrid(IStrategy):
    """
    Estrategia híbrida: reversión a la media + rupturas.
    - Reversión: Z-price extremo + Z-vol bajo + ADX bajo + KER bajo.
    - Ruptura:   Donchian break + Z-vol alto + ADX alto + KER alto + DI alineado.
    DCA solo en reversión. Stoploss dinámico por ATR, más ajustado en rupturas.
    """

    INTERFACE_VERSION = 3
    timeframe = "15m"
    can_short = True

    stoploss = -0.10
    use_custom_stoploss = True
    trailing_stop = False

    minimal_roi = {"0": 0.04, "30": 0.02, "60": 0.01, "120": 0.0}

    position_adjustment_enable = True
    max_entry_position_adjustment = 2

    process_only_new_candles = True
    startup_candle_count = 120

    # --- Indicadores ---
    zscore_window = 50
    adx_period = 14
    di_period = 14
    ker_period = 20
    atr_period = 14
    donchian_period = 20

    # --- Reversión ---
    z_entry = 2.5
    z_dca_add = 0.8
    z_exit = 0.5
    adx_lateral = 22
    ker_lateral = 0.35
    zvol_max_reversion = 1.0

    # --- Ruptura ---
    zvol_min_breakout = 1.5
    adx_trend = 25
    ker_trend = 0.5

    # --- Stake ---
    stake_amount = "unlimited"
    custom_stake_amount_risk = 0.01

    # ------------------------------------------------------------------
    # INDICADORES
    # ------------------------------------------------------------------
    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Z-score robusto precio
        med_p = dataframe["close"].rolling(self.zscore_window).median()
        mad_p = (dataframe["close"] - med_p).abs().rolling(self.zscore_window).median()
        dataframe["z_price"] = (dataframe["close"] - med_p) / (1.4826 * mad_p + 1e-10)

        # Z-score robusto volumen
        med_v = dataframe["volume"].rolling(self.zscore_window).median()
        mad_v = (dataframe["volume"] - med_v).abs().rolling(self.zscore_window).median()
        dataframe["z_vol"] = (dataframe["volume"] - med_v) / (1.4826 * mad_v + 1e-10)

        # ADX + DI
        dataframe["adx"] = ta.ADX(dataframe, timeperiod=self.adx_period)
        dataframe["plus_di"] = ta.PLUS_DI(dataframe, timeperiod=self.di_period)
        dataframe["minus_di"] = ta.MINUS_DI(dataframe, timeperiod=self.di_period)

        # KER
        change = dataframe["close"].diff(self.ker_period).abs()
        vol_path = dataframe["close"].diff().abs().rolling(self.ker_period).sum()
        dataframe["ker"] = change / (vol_path + 1e-10)

        # ATR
        dataframe["atr"] = ta.ATR(dataframe, timeperiod=self.atr_period)
        dataframe["atr_pct"] = dataframe["atr"] / dataframe["close"]

        # Donchian (nivel de ruptura, excluyendo la vela actual)
        dataframe["dc_high"] = dataframe["high"].rolling(self.donchian_period).max().shift(1)
        dataframe["dc_low"] = dataframe["low"].rolling(self.donchian_period).min().shift(1)

        return dataframe

    # ------------------------------------------------------------------
    # ENTRADAS
    # ------------------------------------------------------------------
    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        valid = (
            (dataframe["volume"] > 0)
            & dataframe["z_price"].notna()
            & dataframe["ker"].notna()
            & dataframe["adx"].notna()
            & dataframe["dc_high"].notna()
        )

        # --- REVERSIÓN ---
        rev_long = (
            valid
            & (dataframe["z_price"] < -self.z_entry)
            & (dataframe["z_vol"] < self.zvol_max_reversion)
            & (dataframe["adx"] < self.adx_lateral)
            & (dataframe["ker"] < self.ker_lateral)
        )
        rev_short = (
            valid
            & (dataframe["z_price"] > self.z_entry)
            & (dataframe["z_vol"] < self.zvol_max_reversion)
            & (dataframe["adx"] < self.adx_lateral)
            & (dataframe["ker"] < self.ker_lateral)
        )

        # --- RUPTURA ---
        brk_long = (
            valid
            & (dataframe["close"] > dataframe["dc_high"])
            & (dataframe["z_vol"] > self.zvol_min_breakout)
            & (dataframe["adx"] > self.adx_trend)
            & (dataframe["ker"] > self.ker_trend)
            & (dataframe["plus_di"] > dataframe["minus_di"])
            & (dataframe["z_price"] > 0)
        )
        brk_short = (
            valid
            & (dataframe["close"] < dataframe["dc_low"])
            & (dataframe["z_vol"] > self.zvol_min_breakout)
            & (dataframe["adx"] > self.adx_trend)
            & (dataframe["ker"] > self.ker_trend)
            & (dataframe["minus_di"] > dataframe["plus_di"])
            & (dataframe["z_price"] < 0)
        )

        dataframe.loc[rev_long, ["enter_long", "enter_tag"]] = (1, "reversion_long")
        dataframe.loc[rev_short, ["enter_short", "enter_tag"]] = (1, "reversion_short")
        dataframe.loc[brk_long, ["enter_long", "enter_tag"]] = (1, "breakout_long")
        dataframe.loc[brk_short, ["enter_short", "enter_tag"]] = (1, "breakout_short")

        return dataframe

    # ------------------------------------------------------------------
    # SALIDAS: todas en custom_exit para distinguir por tag
    # ------------------------------------------------------------------
    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        return dataframe

    def custom_exit(self, pair, trade, current_time, current_rate, current_profit, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        z = last["z_price"]
        adx = last["adx"]
        ker = last["ker"]
        plus_di = last["plus_di"]
        minus_di = last["minus_di"]
        tag = trade.enter_tag or ""

        # Reversión: cerrar cuando el Z se normaliza
        if tag == "reversion_long" and z > -self.z_exit:
            return "reversion_target_long"
        if tag == "reversion_short" and z < self.z_exit:
            return "reversion_target_short"

        # Ruptura: cerrar cuando muere la tendencia o falla la estructura
        if tag == "breakout_long":
            if adx < self.adx_lateral:
                return "breakout_adx_dead"
            if ker < self.ker_lateral:
                return "breakout_ker_dead"
            if minus_di > plus_di:
                return "breakout_di_flip"
            if z < 0:
                return "breakout_failed"
        if tag == "breakout_short":
            if adx < self.adx_lateral:
                return "breakout_adx_dead"
            if ker < self.ker_lateral:
                return "breakout_ker_dead"
            if plus_di > minus_di:
                return "breakout_di_flip"
            if z > 0:
                return "breakout_failed"

        return None

    # ------------------------------------------------------------------
    # DCA: solo en reversión
    # ------------------------------------------------------------------
    def adjust_trade_position(self, trade, current_time, current_rate, current_profit,
                              min_stake, max_stake, current_entry_rate, current_exit_rate,
                              current_entry_profit, current_exit_profit, **kwargs):
        if trade.has_open_orders:
            return None

        tag = trade.enter_tag or ""
        if "breakout" in tag:
            return None  # nunca promediar rupturas

        dataframe, _ = self.dp.get_analyzed_dataframe(pair=trade.pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        last = dataframe.iloc[-1]
        z_now = last["z_price"]
        adx_now = last["adx"]

        if trade.is_long:
            if z_now < -(self.z_entry + self.z_dca_add) and adx_now < self.adx_lateral:
                return trade.stake_amount
        else:
            if z_now > (self.z_entry + self.z_dca_add) and adx_now < self.adx_lateral:
                return trade.stake_amount

        return None

    # ------------------------------------------------------------------
    # STOPLOSS DINÁMICO POR RÉGIMEN
    # ------------------------------------------------------------------
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, after_fill, **kwargs):
        dataframe, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if dataframe is None or dataframe.empty:
            return None

        atr_pct = dataframe.iloc[-1]["atr_pct"]
        if pd.isna(atr_pct) or atr_pct <= 0:
            return None

        tag = trade.enter_tag or ""

        if "breakout" in tag:
            # Rupturas: fallar rápido y pasar a trailing si ya está en beneficio
            if current_profit > 0.02:
                stop_distance = min(max(1.0 * atr_pct, 0.01), 0.10)
            else:
                stop_distance = min(max(1.5 * atr_pct, 0.02), 0.15)
        else:
            # Reversión: necesita aire, el precio puede seguir estirándose
            stop_distance = min(max(2.5 * atr_pct, 0.03), 0.25)

        return stoploss_from_open(
            stop_distance,
            current_profit,
            is_short=trade.is_short,
            leverage=trade.leverage,
        )

    # ------------------------------------------------------------------
    # STAKE
    # ------------------------------------------------------------------
    def custom_stake_amount(self, pair, current_time, current_rate, proposed_stake,
                            min_stake, max_stake, leverage, entry_tag, side, **kwargs):
        total = self.wallets.get_total_stake_amount()
        target = total * self.custom_stake_amount_risk / max(len(self.dp.current_whitelist()), 1)
        return max(min(target, max_stake), min_stake or 0.0)