from abc import ABC, abstractmethod
from typing import List, Optional
from collections import deque
from models import MarketDataPoint, Signal

class Strategy(ABC):
    @abstractmethod
    def generate_signals(self, tick: MarketDataPoint) -> List[Signal]:
        """
        Given a new MarketDataPoint, decide whether to emit trading signals.
        Returns a list (could be empty, or contain 1+ Signal objects).
        """
        pass


# --------------------------
# Strategy 1: SMA Crossover
# --------------------------
class SMACrossoverStrategy(Strategy):
    """
    Short SMA vs Long SMA crossover.
    BUY when short SMA crosses above long SMA.
    SELL when short SMA crosses below long SMA.
    """
    
    def __init__(self, symbol: str, short_window: int = 10, long_window: int = 30):
        self._symbol = symbol
        self._short_window = short_window
        self._long_window = long_window
        self._price_history = deque(maxlen=long_window)
        self._previous_short_sma = None
        self._previous_long_sma = None

    def _calculate_sma(self, window: int) -> Optional[float]:
        """Calculate Simple Moving Average for the given window."""
        if len(self._price_history) < window:
            return None
        return sum(list(self._price_history)[-window:]) / window

    def generate_signals(self, tick: MarketDataPoint) -> List[Signal]:
        if tick.symbol != self._symbol:
            return []

        out: List[Signal] = []
        
        # Add current price to history
        self._price_history.append(tick.price)
        
        # Calculate current SMAs
        current_short_sma = self._calculate_sma(self._short_window)
        current_long_sma = self._calculate_sma(self._long_window)
        
        # Need both SMAs to be available and previous values for crossover detection
        if (current_short_sma is None or current_long_sma is None or 
            self._previous_short_sma is None or self._previous_long_sma is None):
            # Store current values for next iteration
            self._previous_short_sma = current_short_sma
            self._previous_long_sma = current_long_sma
            return out
        
        # Check for crossover signals
        # BUY: short SMA crosses above long SMA
        if (self._previous_short_sma <= self._previous_long_sma and 
            current_short_sma > current_long_sma):
            out.append(Signal(
                tick.timestamp, 
                tick.symbol, 
                "BUY", 
                1, 
                reason=f"SMA crossover: {current_short_sma:.2f} > {current_long_sma:.2f}", 
                strategy="SMA_CROSSOVER"
            ))
        
        # SELL: short SMA crosses below long SMA
        elif (self._previous_short_sma >= self._previous_long_sma and 
              current_short_sma < current_long_sma):
            out.append(Signal(
                tick.timestamp, 
                tick.symbol, 
                "SELL", 
                1, 
                reason=f"SMA crossover: {current_short_sma:.2f} < {current_long_sma:.2f}", 
                strategy="SMA_CROSSOVER"
            ))
        
        # Store current values for next iteration
        self._previous_short_sma = current_short_sma
        self._previous_long_sma = current_long_sma
        
        return out



# --------------------------
# Strategy 3: Random buy and sell
# --------------------------
class RandomBuyAndSellStrategy(Strategy):
    """
    Randomly BUY or SELL 1 share.
    """

    def __init__(self, symbol: str, capital: float):
        self._symbol = symbol
        self._capital = capital

    def generate_signals(self, tick: MarketDataPoint) -> List[Signal]:
        if tick.symbol != self._symbol:
            return []

        out: List[Signal] = []

        if random.random() < 0.5:
            out.append(Signal(tick.timestamp, tick.symbol, "BUY", 1, reason="Random buy", strategy="RANDOM_BUY_AND_SELL"))

        return out