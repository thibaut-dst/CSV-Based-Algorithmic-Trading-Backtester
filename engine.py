# engine.py
import csv
import logging
from datetime import datetime
from typing import List, Dict

from models import MarketDataPoint, Signal, Order
from strategies import Strategy

logger = logging.getLogger(__name__)


class ExecutionEngine:
    """
    Core engine that ties everything together:
    - Buffers market data
    - Runs strategies to generate signals
    - Converts signals to orders
    - Updates positions when orders are "executed"
    """

    def __init__(self):
        # Containers
        self._market_data: List[MarketDataPoint] = []  # all ticks
        self._signals: List[Signal] = []               # all signals
        self._orders: List[Order] = []                 # all orders
        self._positions: Dict[str, Dict] = {}          # {symbol: {"quantity": int, "avg_price": float}}

    # -----------------
    # Data loading
    # -----------------
    def load_data(self, csv_path: str):
        """Read market data from a CSV file and store as MarketDataPoint list."""
        with open(csv_path, "r", newline="") as file:
            reader = csv.reader(file)
            next(reader)  # skip header
            for row in reader:
                tick = MarketDataPoint(
                    timestamp=datetime.fromisoformat(row[0]),
                    symbol=row[1],
                    price=float(row[2]),
                )
                self._market_data.append(tick)
        logger.info(f"Loaded {len(self._market_data)} ticks from {csv_path}")

    # -----------------
    # Backtest loop
    # -----------------
    def run(self, strategies: List[Strategy]):
        """Run the backtest by feeding ticks to each strategy."""
        for tick in self._market_data:
            for strat in strategies:
                signals = strat.generate_signals(tick)
                if signals:
                    self._signals.extend(signals)
                    for sig in signals:
                        order = self._signal_to_order(sig, tick.price)
                        self._orders.append(order)
                        self._update_positions(order)

    # -----------------
    # Helpers
    # -----------------
    def _signal_to_order(self, signal: Signal, price: float) -> Order:
        """Convert a trading signal into an order at the current market price."""
        qty = signal.qty if signal.side == "BUY" else -signal.qty
        order = Order(signal.symbol, qty, price, status="FILLED")
        logger.info(f"Created order: {order.symbol} {order.quantity}@{order.price} [{order.status}]")
        return order

    def _update_positions(self, order: Order):
        """Update portfolio positions based on a filled order."""
        pos = self._positions.setdefault(order.symbol, {"quantity": 0, "avg_price": 0.0})

        if order.quantity > 0:  # BUY
            total_cost = pos["quantity"] * pos["avg_price"] + order.quantity * order.price
            pos["quantity"] += order.quantity
            pos["avg_price"] = total_cost / pos["quantity"]
        else:  # SELL
            pos["quantity"] += order.quantity  # quantity is negative
            if pos["quantity"] == 0:
                pos["avg_price"] = 0.0  # reset if flat

        logger.info(f"Updated position: {order.symbol} → {pos}")

    # -----------------
    # Accessors
    # -----------------
    @property
    def positions(self) -> Dict[str, Dict]:
        return self._positions

    @property
    def orders(self) -> List[Order]:
        return self._orders

    @property
    def signals(self) -> List[Signal]:
        return self._signals

