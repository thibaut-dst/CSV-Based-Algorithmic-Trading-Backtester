import pytest
from datetime import datetime
from models import MarketDataPoint, Order

def test_order_status_update():
    order = Order(symbol="AAPL", quantity=10, price=150, status="pending")
    print(f"Before update: status={order.status}")
    order.status = "completed"
    print(f"After update: status={order.status}")
    assert order.status == "completed"

def test_market_data_point_immutable():
    mdp = MarketDataPoint(timestamp=datetime.now(), symbol="AAPL", price=150.0)
    print(f"Original price={mdp.price}")
    from dataclasses import FrozenInstanceError
    with pytest.raises(FrozenInstanceError):
        mdp.price = 155.0
    print("Attempting to update price raised FrozenInstanceError as expected")
