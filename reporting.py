#!/usr/bin/env python3
"""
Performance Reporting Module for CSV-Based Algorithmic Trading Backtester

This module provides comprehensive performance analysis including:
- Total return calculation
- Periodic returns analysis
- Sharpe ratio computation
- Maximum drawdown analysis
- Professional reporting in Markdown format
- ASCII equity curve visualization
"""

import math
import statistics
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Any
import logging

from models import MarketDataPoint, Order, Signal
from engine import ExecutionEngine

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Comprehensive performance analysis for trading strategies.
    Calculates key metrics and generates professional reports.
    """
    
    def __init__(self, engine: ExecutionEngine, initial_capital: float = 100000.0):
        """
        Initialize the performance analyzer.
        
        Args:
            engine: ExecutionEngine instance with completed backtest
            initial_capital: Starting capital for performance calculations
        """
        self.engine = engine
        self.initial_capital = initial_capital
        self.portfolio_history = []
        self.returns_series = []
        self.metrics = {}
        
        logger.info(f"Initialized PerformanceAnalyzer with ${initial_capital:,.2f} initial capital")
    
    def calculate_portfolio_history(self) -> List[Dict[str, Any]]:
        """
        Calculate portfolio value at each point in time during the backtest.
        
        Returns:
            List of portfolio snapshots with timestamp, value, positions, etc.
        """
        if not self.engine._market_data:
            logger.warning("No market data available for portfolio calculation")
            return []
        
        logger.info("Calculating portfolio value history...")
        
        # Start with initial portfolio
        current_portfolio = {
            'cash': self.initial_capital,
            'positions': {},
            'total_value': self.initial_capital
        }
        
        # Track orders by timestamp for portfolio updates
        orders_by_time = {}
        for order in self.engine.orders:
            if order.status == "FILLED":
                # Find the corresponding tick for this order
                for tick in self.engine._market_data:
                    # Match orders to ticks (simplified - in reality you'd want more precise matching)
                    if tick.symbol == order.symbol:
                        if tick.timestamp not in orders_by_time:
                            orders_by_time[tick.timestamp] = []
                        orders_by_time[tick.timestamp].append(order)
                        break
        
        portfolio_snapshots = []
        
        # Process each tick chronologically
        for tick in self.engine._market_data:
            # Update portfolio with any orders executed at this time
            if tick.timestamp in orders_by_time:
                for order in orders_by_time[tick.timestamp]:
                    signal_side = getattr(order, '_signal_side', 'BUY')
                    
                    if signal_side == 'BUY':
                        # Buy order: decrease cash, increase position
                        cost = order.quantity * order.price
                        current_portfolio['cash'] -= cost
                        
                        if order.symbol not in current_portfolio['positions']:
                            current_portfolio['positions'][order.symbol] = 0
                        current_portfolio['positions'][order.symbol] += order.quantity
                        
                    elif signal_side == 'SELL':
                        # Sell order: increase cash, decrease position
                        proceeds = order.quantity * order.price
                        current_portfolio['cash'] += proceeds
                        
                        if order.symbol in current_portfolio['positions']:
                            current_portfolio['positions'][order.symbol] -= order.quantity
                            if current_portfolio['positions'][order.symbol] == 0:
                                del current_portfolio['positions'][order.symbol]
            
            # Calculate current portfolio value using current market prices
            portfolio_value = current_portfolio['cash']
            for symbol, quantity in current_portfolio['positions'].items():
                if tick.symbol == symbol:  # Use current tick price for this symbol
                    portfolio_value += quantity * tick.price
                else:
                    # For other symbols, use last known price (simplified)
                    # In a real system, you'd maintain price history for all symbols
                    portfolio_value += quantity * 100  # Placeholder
            
            # Store portfolio snapshot
            snapshot = {
                'timestamp': tick.timestamp,
                'total_value': portfolio_value,
                'cash': current_portfolio['cash'],
                'positions': current_portfolio['positions'].copy(),
                'market_price': {tick.symbol: tick.price}
            }
            portfolio_snapshots.append(snapshot)
        
        self.portfolio_history = portfolio_snapshots
        logger.info(f"Generated {len(portfolio_snapshots)} portfolio snapshots")
        return portfolio_snapshots
    
    def calculate_returns(self) -> List[float]:
        """
        Calculate periodic returns from portfolio history.
        
        Returns:
            List of periodic returns
        """
        if not self.portfolio_history:
            self.calculate_portfolio_history()
        
        if len(self.portfolio_history) < 2:
            logger.warning("Insufficient data for return calculation")
            return []
        
        returns = []
        for i in range(1, len(self.portfolio_history)):
            prev_value = self.portfolio_history[i-1]['total_value']
            curr_value = self.portfolio_history[i]['total_value']
            
            if prev_value > 0:
                period_return = (curr_value - prev_value) / prev_value
                returns.append(period_return)
            else:
                returns.append(0.0)
        
        self.returns_series = returns
        logger.info(f"Calculated {len(returns)} periodic returns")
        return returns
    
    def calculate_total_return(self) -> float:
        """Calculate total return over the entire backtest period."""
        if not self.portfolio_history:
            self.calculate_portfolio_history()
        
        if not self.portfolio_history:
            return 0.0
        
        final_value = self.portfolio_history[-1]['total_value']
        total_return = (final_value - self.initial_capital) / self.initial_capital
        
        logger.info(f"Total return: {total_return:.4f} ({total_return*100:.2f}%)")
        return total_return
    
    def calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """
        Calculate Sharpe ratio.
        
        Args:
            risk_free_rate: Annual risk-free rate (default 2%)
            
        Returns:
            Sharpe ratio
        """
        if not self.returns_series:
            self.calculate_returns()
        
        if not self.returns_series or len(self.returns_series) < 2:
            logger.warning("Insufficient returns data for Sharpe ratio calculation")
            return 0.0
        
        # Convert annual risk-free rate to period rate
        # Assuming daily periods (adjust based on your data frequency)
        periods_per_year = 252  # Trading days
        period_risk_free_rate = risk_free_rate / periods_per_year
        
        # Calculate excess returns
        excess_returns = [r - period_risk_free_rate for r in self.returns_series]
        
        if not excess_returns:
            return 0.0
        
        mean_excess_return = statistics.mean(excess_returns)
        
        if len(excess_returns) < 2:
            return 0.0
        
        std_excess_return = statistics.stdev(excess_returns)
        
        if std_excess_return == 0:
            return 0.0
        
        # Annualize the Sharpe ratio
        sharpe_ratio = (mean_excess_return / std_excess_return) * math.sqrt(periods_per_year)
        
        logger.info(f"Sharpe ratio: {sharpe_ratio:.4f}")
        return sharpe_ratio
    
    def calculate_max_drawdown(self) -> Tuple[float, Dict[str, Any]]:
        """
        Calculate maximum drawdown and related statistics.
        
        Returns:
            Tuple of (max_drawdown_pct, drawdown_info)
        """
        if not self.portfolio_history:
            self.calculate_portfolio_history()
        
        if len(self.portfolio_history) < 2:
            logger.warning("Insufficient data for drawdown calculation")
            return 0.0, {}
        
        values = [snapshot['total_value'] for snapshot in self.portfolio_history]
        
        max_drawdown = 0.0
        peak_value = values[0]
        peak_index = 0
        trough_index = 0
        drawdown_start = self.portfolio_history[0]['timestamp']
        drawdown_end = self.portfolio_history[0]['timestamp']
        
        for i, value in enumerate(values):
            if value > peak_value:
                peak_value = value
                peak_index = i
            
            drawdown = (peak_value - value) / peak_value if peak_value > 0 else 0
            
            if drawdown > max_drawdown:
                max_drawdown = drawdown
                trough_index = i
                drawdown_start = self.portfolio_history[peak_index]['timestamp']
                drawdown_end = self.portfolio_history[i]['timestamp']
        
        drawdown_info = {
            'max_drawdown_pct': max_drawdown,
            'peak_value': peak_value,
            'trough_value': values[trough_index] if trough_index < len(values) else values[-1],
            'drawdown_start': drawdown_start,
            'drawdown_end': drawdown_end,
            'duration_days': (drawdown_end - drawdown_start).days if drawdown_end > drawdown_start else 0
        }
        
        logger.info(f"Maximum drawdown: {max_drawdown:.4f} ({max_drawdown*100:.2f}%)")
        return max_drawdown, drawdown_info
    
    def calculate_all_metrics(self) -> Dict[str, Any]:
        """Calculate all performance metrics and store in metrics dict."""
        logger.info("Calculating comprehensive performance metrics...")
        
        # Basic statistics
        total_signals = len(self.engine.signals)
        total_orders = len(self.engine.orders)
        successful_orders = sum(1 for order in self.engine.orders if order.status == "FILLED")
        failed_orders = total_orders - successful_orders
        
        # Performance metrics
        total_return = self.calculate_total_return()
        sharpe_ratio = self.calculate_sharpe_ratio()
        max_drawdown, drawdown_info = self.calculate_max_drawdown()
        
        # Portfolio analysis
        final_portfolio_value = self.portfolio_history[-1]['total_value'] if self.portfolio_history else self.initial_capital
        
        # Return statistics
        if self.returns_series:
            avg_return = statistics.mean(self.returns_series)
            return_volatility = statistics.stdev(self.returns_series) if len(self.returns_series) > 1 else 0
            best_return = max(self.returns_series)
            worst_return = min(self.returns_series)
        else:
            avg_return = return_volatility = best_return = worst_return = 0
        
        # Win/Loss analysis for trades
        winning_trades = 0
        losing_trades = 0
        for i in range(1, len(self.returns_series)):
            if self.returns_series[i] > 0:
                winning_trades += 1
            elif self.returns_series[i] < 0:
                losing_trades += 1
        
        win_rate = winning_trades / (winning_trades + losing_trades) if (winning_trades + losing_trades) > 0 else 0
        
        self.metrics = {
            'execution_summary': {
                'total_signals': total_signals,
                'total_orders': total_orders,
                'successful_orders': successful_orders,
                'failed_orders': failed_orders,
                'success_rate': (successful_orders / total_orders * 100) if total_orders > 0 else 0
            },
            'portfolio_performance': {
                'initial_capital': self.initial_capital,
                'final_value': final_portfolio_value,
                'total_return': total_return,
                'total_return_pct': total_return * 100,
                'profit_loss': final_portfolio_value - self.initial_capital
            },
            'risk_metrics': {
                'sharpe_ratio': sharpe_ratio,
                'max_drawdown': max_drawdown,
                'max_drawdown_pct': max_drawdown * 100,
                'volatility': return_volatility,
                'best_period_return': best_return,
                'worst_period_return': worst_return
            },
            'trading_statistics': {
                'total_periods': len(self.returns_series),
                'winning_periods': winning_trades,
                'losing_periods': losing_trades,
                'win_rate': win_rate * 100,
                'average_return': avg_return,
                'average_return_pct': avg_return * 100
            },
            'drawdown_analysis': drawdown_info,
            'positions': self.engine.positions,
            'time_period': {
                'start': self.engine._market_data[0].timestamp.isoformat() if self.engine._market_data else None,
                'end': self.engine._market_data[-1].timestamp.isoformat() if self.engine._market_data else None,
                'total_ticks': len(self.engine._market_data)
            }
        }
        
        logger.info("Performance metrics calculation completed")
        return self.metrics
    
    def generate_ascii_equity_curve(self, width: int = 80, height: int = 20) -> str:
        """
        Generate ASCII art equity curve.
        
        Args:
            width: Chart width in characters
            height: Chart height in characters
            
        Returns:
            ASCII art string representing the equity curve
        """
        if not self.portfolio_history:
            return "No portfolio data available for equity curve"
        
        values = [snapshot['total_value'] for snapshot in self.portfolio_history]
        
        if len(values) < 2:
            return "Insufficient data for equity curve"
        
        # Normalize values to fit chart dimensions
        min_val = min(values)
        max_val = max(values)
        
        if max_val == min_val:
            return "No variation in portfolio value"
        
        # Create the chart
        chart_lines = []
        
        # Header
        chart_lines.append("Portfolio Equity Curve")
        chart_lines.append("=" * width)
        chart_lines.append(f"Initial: ${self.initial_capital:,.0f} | Final: ${values[-1]:,.0f} | Return: {((values[-1]/self.initial_capital-1)*100):+.1f}%")
        chart_lines.append("")
        
        # Y-axis labels and chart body
        for row in range(height):
            y_pos = height - 1 - row
            y_value = min_val + (max_val - min_val) * (y_pos / (height - 1))
            
            line = f"{y_value:8.0f} │"
            
            for col in range(width - 10):
                data_index = int((col / (width - 10)) * (len(values) - 1))
                data_value = values[data_index]
                
                # Determine if this position should have a point
                normalized_value = (data_value - min_val) / (max_val - min_val)
                chart_y = normalized_value * (height - 1)
                
                if abs(chart_y - y_pos) < 0.5:
                    line += "●"
                elif row == height - 1:  # Bottom line
                    line += "─"
                else:
                    line += " "
            
            chart_lines.append(line)
        
        # X-axis
        x_axis = "         └" + "─" * (width - 10)
        chart_lines.append(x_axis)
        
        # Time labels
        start_time = self.portfolio_history[0]['timestamp'].strftime("%H:%M:%S")
        end_time = self.portfolio_history[-1]['timestamp'].strftime("%H:%M:%S")
        time_label = f"         {start_time}" + " " * (width - 20 - len(start_time) - len(end_time)) + end_time
        chart_lines.append(time_label)
        
        return "\\n".join(chart_lines)
    
    def generate_markdown_report(self, filename: str = "performance.md") -> str:
        """
        Generate comprehensive performance report in Markdown format.
        
        Args:
            filename: Output filename for the report
            
        Returns:
            Path to generated report file
        """
        if not self.metrics:
            self.calculate_all_metrics()
        
        logger.info(f"Generating performance report: {filename}")
        
        # Generate report content
        report_content = self._create_markdown_content()
        
        # Write to file
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Performance report generated: {filename}")
        return filename
    
    def _create_markdown_content(self) -> str:
        """Create the markdown content for the performance report."""
        execution = self.metrics['execution_summary']
        portfolio = self.metrics['portfolio_performance']
        risk = self.metrics['risk_metrics']
        trading = self.metrics['trading_statistics']
        drawdown = self.metrics['drawdown_analysis']
        time_info = self.metrics['time_period']
        
        content = f"""# Trading Strategy Performance Report

Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Executive Summary

This report analyzes the performance of the algorithmic trading strategy over the period from {time_info['start']} to {time_info['end']}.

**Key Highlights:**
- **Total Return**: {portfolio['total_return_pct']:+.2f}%
- **Sharpe Ratio**: {risk['sharpe_ratio']:.3f}
- **Maximum Drawdown**: {risk['max_drawdown_pct']:.2f}%
- **Win Rate**: {trading['win_rate']:.1f}%

## Portfolio Performance

| Metric | Value |
|--------|-------|
| Initial Capital | ${portfolio['initial_capital']:,.2f} |
| Final Portfolio Value | ${portfolio['final_value']:,.2f} |
| Total Return | {portfolio['total_return_pct']:+.2f}% |
| Absolute Profit/Loss | ${portfolio['profit_loss']:+,.2f} |

## Risk Metrics

| Metric | Value |
|--------|-------|
| Sharpe Ratio | {risk['sharpe_ratio']:.3f} |
| Maximum Drawdown | {risk['max_drawdown_pct']:.2f}% |
| Volatility (Daily) | {risk['volatility']*100:.2f}% |
| Best Period Return | {risk['best_period_return']*100:+.2f}% |
| Worst Period Return | {risk['worst_period_return']*100:+.2f}% |

## Trading Statistics

| Metric | Value |
|--------|-------|
| Total Signals Generated | {execution['total_signals']} |
| Total Orders Placed | {execution['total_orders']} |
| Successful Orders | {execution['successful_orders']} |
| Failed Orders | {execution['failed_orders']} |
| Order Success Rate | {execution['success_rate']:.1f}% |
| Total Trading Periods | {trading['total_periods']} |
| Winning Periods | {trading['winning_periods']} |
| Losing Periods | {trading['losing_periods']} |
| Win Rate | {trading['win_rate']:.1f}% |
| Average Period Return | {trading['average_return_pct']:+.3f}% |

## Drawdown Analysis

| Metric | Value |
|--------|-------|
| Maximum Drawdown | {risk['max_drawdown_pct']:.2f}% |
| Peak Portfolio Value | ${drawdown.get('peak_value', 0):,.2f} |
| Trough Portfolio Value | ${drawdown.get('trough_value', 0):,.2f} |
| Drawdown Start | {drawdown.get('drawdown_start', 'N/A')} |
| Drawdown End | {drawdown.get('drawdown_end', 'N/A')} |
| Drawdown Duration | {drawdown.get('duration_days', 0)} days |

## Current Positions

"""

        # Add current positions
        positions = self.metrics['positions']
        if positions:
            content += "| Symbol | Quantity | Average Price | Market Value |\n"
            content += "|--------|----------|---------------|-------------|\n"
            for symbol, pos in positions.items():
                if pos['quantity'] != 0:
                    market_value = pos['quantity'] * pos['avg_price']
                    content += f"| {symbol} | {pos['quantity']} | ${pos['avg_price']:.2f} | ${market_value:,.2f} |\n"
        else:
            content += "No open positions.\n"

        content += f"""

## Equity Curve

```
{self.generate_ascii_equity_curve()}
```

## Performance Interpretation

### Overall Assessment
"""

        # Add narrative interpretation
        if portfolio['total_return'] > 0:
            content += f"The strategy generated a positive return of {portfolio['total_return_pct']:.2f}% over the testing period, "
        else:
            content += f"The strategy experienced a negative return of {portfolio['total_return_pct']:.2f}% over the testing period, "

        if risk['sharpe_ratio'] > 1.0:
            content += "with an excellent risk-adjusted return as indicated by a Sharpe ratio above 1.0."
        elif risk['sharpe_ratio'] > 0.5:
            content += "with a reasonable risk-adjusted return as indicated by a moderate Sharpe ratio."
        else:
            content += "with poor risk-adjusted returns as indicated by a low Sharpe ratio."

        content += f"""

### Risk Assessment
The strategy experienced a maximum drawdown of {risk['max_drawdown_pct']:.2f}%, which represents the largest peak-to-trough decline in portfolio value. """

        if risk['max_drawdown_pct'] < 10:
            content += "This relatively low drawdown suggests good risk management."
        elif risk['max_drawdown_pct'] < 20:
            content += "This moderate drawdown is within acceptable ranges for most strategies."
        else:
            content += "This high drawdown indicates significant risk and potential for large losses."

        content += f"""

### Trading Effectiveness
The strategy achieved a win rate of {trading['win_rate']:.1f}% with an order execution success rate of {execution['success_rate']:.1f}%. """

        if trading['win_rate'] > 60:
            content += "The high win rate suggests the strategy is effective at identifying profitable opportunities."
        elif trading['win_rate'] > 50:
            content += "The win rate is above break-even, indicating the strategy has positive edge."
        else:
            content += "The win rate below 50% suggests the strategy may need refinement, though this could be acceptable if winning trades are larger than losing ones."

        content += f"""

### Recommendations
Based on the analysis:

1. **Risk Management**: """
        
        if risk['max_drawdown_pct'] > 15:
            content += "Consider implementing stricter risk controls to reduce drawdown."
        else:
            content += "Current risk levels appear manageable."

        content += f"""
2. **Strategy Optimization**: """
        
        if trading['win_rate'] < 50:
            content += "Focus on improving trade selection criteria to increase win rate."
        else:
            content += "The strategy shows good trade selection. Consider position sizing optimization."

        content += f"""
3. **Performance**: """
        
        if portfolio['total_return'] > 0 and risk['sharpe_ratio'] > 0.5:
            content += "The strategy shows promise and may be suitable for live trading with proper risk management."
        else:
            content += "The strategy requires further development before considering live implementation."

        content += f"""

---

*This report was generated automatically by the CSV-Based Algorithmic Trading Backtester.*
*Past performance does not guarantee future results.*
"""

        return content

    def print_summary(self):
        """Print a brief performance summary to console."""
        if not self.metrics:
            self.calculate_all_metrics()
        
        portfolio = self.metrics['portfolio_performance']
        risk = self.metrics['risk_metrics']
        trading = self.metrics['trading_statistics']
        
        print("\\n" + "="*60)
        print("           PERFORMANCE SUMMARY")
        print("="*60)
        print(f"Initial Capital:     ${portfolio['initial_capital']:,.2f}")
        print(f"Final Value:         ${portfolio['final_value']:,.2f}")
        print(f"Total Return:        {portfolio['total_return_pct']:+.2f}%")
        print(f"Sharpe Ratio:        {risk['sharpe_ratio']:.3f}")
        print(f"Max Drawdown:        {risk['max_drawdown_pct']:.2f}%")
        print(f"Win Rate:            {trading['win_rate']:.1f}%")
        print("="*60)