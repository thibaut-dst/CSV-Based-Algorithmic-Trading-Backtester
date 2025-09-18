import csv
import dataclasses
from dataclasses import dataclass, FrozenInstanceError
from datetime import datetime
from models import MarketDataPoint, Order




market_data_points = []

with open('market_data.csv', 'r', newline='') as file:
    reader = csv.reader(file)
    next(reader)  # Skip header row if present
    for row in reader:
        point = MarketDataPoint(
            timestamp=datetime.fromisoformat(row[0]),
            symbol=row[1],
            price=float(row[2])
        )
        market_data_points.append(point)

#print(market_data_points)