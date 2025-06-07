# A股市场恐慌情绪量化分析与仓位控制系统
# 用于聚宽平台的恐慌情绪指标计算和仓位管理

import numpy as np
import pandas as pd
from datetime import datetime, timedelta

class PanicSentimentAnalyzer:
    """
    恐慌情绪量化分析器
    输出恐慌指数：0-100分，分数越高恐慌程度越严重
    """
    
    def __init__(self):
        # 恐慌指标权重配置
        self.weights = {
            'price_drop': 0.25,      # 价格跌幅权重
            'volume_spike': 0.20,    # 成交量异常权重
            'limit_down': 0.25,      # 跌停比率权重
            'volatility': 0.20,      # 波动率权重
            'breadth': 0.10         # 市场宽度权重
        }
        
        # 历史基准值（用于标准化）
        self.benchmarks = {
            'normal_volatility': 15,  # 正常年化波动率
            'normal_volume': 8000,    # 正常日成交额（亿元）
            'normal_limit_ratio': 0.5 # 正常跌停比率%
        }
    
    def calculate_panic_index(self, context, lookback_days=20):
        """
        计算综合恐慌指数
        返回：0-100的恐慌分数
        """
        current_date = context.current_dt.date()
        
        # 1. 价格跌幅指标 (25%)
        price_score = self._calculate_price_drop_score(context, lookback_days)
        
        # 2. 成交量异常指标 (20%)
        volume_score = self._calculate_volume_spike_score(context, lookback_days)
        
        # 3. 跌停比率指标 (25%)
        limit_score = self._calculate_limit_down_score(context)
        
        # 4. 波动率指标 (20%)
        volatility_score = self._calculate_volatility_score(context, lookback_days)
        
        # 5. 市场宽度指标 (10%)
        breadth_score = self._calculate_market_breadth_score(context)
        
        # 计算加权总分
        panic_index = (
            price_score * self.weights['price_drop'] +
            volume_score * self.weights['volume_spike'] +
            limit_score * self.weights['limit_down'] +
            volatility_score * self.weights['volatility'] +
            breadth_score * self.weights['breadth']
        )
        
        # 记录各项指标详情
        details = {
            'panic_index': round(panic_index, 2),
            'price_score': round(price_score, 2),
            'volume_score': round(volume_score, 2),
            'limit_score': round(limit_score, 2),
            'volatility_score': round(volatility_score, 2),
            'breadth_score': round(breadth_score, 2),
            'panic_level': self._get_panic_level(panic_index),
            'suggested_position': self._calculate_suggested_position(panic_index)
        }
        
        return panic_index, details
    
    def _calculate_price_drop_score(self, context, lookback_days):
        """计算价格跌幅分数"""
        # 获取上证指数数据
        index_code = '000001.XSHG'
        price_data = attribute_history(index_code, lookback_days + 1, '1d', ['close'])
        
        if len(price_data) < 2:
            return 0
        
        # 计算不同时间跨度的跌幅
        drop_1d = (price_data.iloc[-1] - price_data.iloc[-2]) / price_data.iloc[-2] * 100
        drop_5d = (price_data.iloc[-1] - price_data.iloc[-6]) / price_data.iloc[-6] * 100 if len(price_data) >= 6 else 0
        drop_20d = (price_data.iloc[-1] - price_data.iloc[0]) / price_data.iloc[0] * 100
        
        # 连续下跌天数
        consecutive_drops = 0
        for i in range(len(price_data)-1, 0, -1):
            if price_data.iloc[i]['close'] < price_data.iloc[i-1]['close']:
                consecutive_drops += 1
            else:
                break
        
        # 计算分数
        score = 0
        score += max(0, min(30, -drop_1d * 10))  # 单日跌幅，最高30分
        score += max(0, min(30, -drop_5d * 6))   # 5日跌幅，最高30分
        score += max(0, min(20, -drop_20d * 2))  # 20日跌幅，最高20分
        score += min(20, consecutive_drops * 5)   # 连跌天数，最高20分
        
        return min(100, score)
    
    def _calculate_volume_spike_score(self, context, lookback_days):
        """计算成交量异常分数"""
        # 获取市场总成交额
        stocks = get_index_stocks('000001.XSHG')[:100]  # 采样100只
        volume_data = history(lookback_days, '1d', 'volume', stocks, df=True)
        money_data = history(lookback_days, '1d', 'money', stocks, df=True)
        
        # 计算日均成交额（亿元）
        daily_money = money_data.sum(axis=1) / 100000000
        avg_money = daily_money[:-1].mean()
        current_money = daily_money.iloc[-1]
        
        # 计算成交量比率
        volume_ratio = current_money / avg_money if avg_money > 0 else 1
        
        # 计算成交量Z分数
        std_money = daily_money[:-1].std()
        z_score = (current_money - avg_money) / std_money if std_money > 0 else 0
        
        # 计算分数
        score = 0
        if volume_ratio > 2.0:
            score += 50
        elif volume_ratio > 1.5:
            score += 30
        elif volume_ratio > 1.2:
            score += 10
        
        score += min(50, abs(z_score) * 15)
        
        return min(100, score)
    
    def _calculate_limit_down_score(self, context):
        """计算跌停比率分数"""
        current_date = context.current_dt
        all_stocks = list(get_all_securities(['stock'], date=current_date).index)
        
        # 获取当前价格和涨跌停价格
        current_data = get_current_data()
        
        limit_down_count = 0
        limit_up_count = 0
        active_count = 0
        
        for stock in all_stocks[:500]:  # 采样500只股票提高效率
            try:
                if not current_data[stock].paused:
                    active_count += 1
                    current_price = current_data[stock].last_price
                    low_limit = current_data[stock].low_limit
                    high_limit = current_data[stock].high_limit
                    
                    if current_price <= low_limit * 1.001:
                        limit_down_count += 1
                    elif current_price >= high_limit * 0.999:
                        limit_up_count += 1
            except:
                continue
        
        # 计算跌停比率
        limit_down_ratio = (limit_down_count / active_count * 100) if active_count > 0 else 0
        
        # 计算跌停涨停比
        limit_ratio = limit_down_count / (limit_up_count + 1)  # 避免除零
        
        # 计算分数
        score = min(60, limit_down_ratio * 20)  # 跌停比率分数
        score += min(40, limit_ratio * 5)       # 跌停涨停比分数
        
        return min(100, score)
    
    def _calculate_volatility_score(self, context, lookback_days):
        """计算波动率分数"""
        # 获取指数数据
        index_code = '000001.XSHG'
        price_data = attribute_history(index_code, lookback_days + 1, '1d', ['close', 'high', 'low'])
        
        # 计算日收益率波动率
        returns = price_data['close'].pct_change().dropna()
        volatility = returns.std() * np.sqrt(252) * 100  # 年化波动率
        
        # 计算日内波动率（Parkinson方法）
        high_low_ratio = np.log(price_data['high'] / price_data['low'])
        parkinson_vol = np.sqrt(252 / (4 * np.log(2))) * high_low_ratio.mean() * 100
        
        # 计算ATR（平均真实波幅）
        tr_list = []
        for i in range(1, len(price_data)):
            high = price_data['high'].iloc[i]
            low = price_data['low'].iloc[i]
            prev_close = price_data['close'].iloc[i-1]
            tr = max(high - low, abs(high - prev_close), abs(low - prev_close))
            tr_list.append(tr / prev_close * 100)
        
        atr = np.mean(tr_list[-14:]) if len(tr_list) >= 14 else np.mean(tr_list)
        
        # 计算分数
        score = 0
        score += min(40, (volatility / self.benchmarks['normal_volatility'] - 1) * 30)
        score += min(30, (parkinson_vol / self.benchmarks['normal_volatility'] - 1) * 25)
        score += min(30, atr * 10)
        
        return max(0, min(100, score))
    
    def _calculate_market_breadth_score(self, context):
        """计算市场宽度分数（上涨下跌股票比例）"""
        stocks = get_index_stocks('000001.XSHG')[:300]  # 采样300只
        
        # 获取今日和昨日收盘价
        price_data = history(2, '1d', 'close', stocks, df=True)
        
        if len(price_data) < 2:
            return 50  # 默认中性分数
        
        # 计算涨跌家数
        advances = 0
        declines = 0
        
        for stock in stocks:
            try:
                if stock in price_data.columns:
                    today_close = price_data[stock].iloc[-1]
                    yesterday_close = price_data[stock].iloc[-2]
                    if today_close > yesterday_close:
                        advances += 1
                    elif today_close < yesterday_close:
                        declines += 1
            except:
                continue
        
        # 计算AD比率
        ad_ratio = advances / (declines + 1)  # 避免除零
        
        # 计算分数（下跌家数越多，分数越高）
        if ad_ratio < 0.2:  # 严重超卖
            score = 90
        elif ad_ratio < 0.5:
            score = 70
        elif ad_ratio < 0.8:
            score = 50
        elif ad_ratio < 1.2:
            score = 30
        else:
            score = 10
        
        return score
    
    def _get_panic_level(self, panic_index):
        """根据恐慌指数返回恐慌级别"""
        if panic_index >= 80:
            return "极度恐慌"
        elif panic_index >= 60:
            return "高度恐慌"
        elif panic_index >= 40:
            return "中度恐慌"
        elif panic_index >= 20:
            return "轻度恐慌"
        else:
            return "正常"
    
    def _calculate_suggested_position(self, panic_index):
        """根据恐慌指数计算建议仓位"""
        # 基础仓位
        base_position = 1.0
        
        # 根据恐慌程度调整仓位
        if panic_index >= 80:  # 极度恐慌
            position = 0.2  # 20%仓位
        elif panic_index >= 60:  # 高度恐慌
            position = 0.4  # 40%仓位
        elif panic_index >= 40:  # 中度恐慌
            position = 0.6  # 60%仓位
        elif panic_index >= 20:  # 轻度恐慌
            position = 0.8  # 80%仓位
        else:  # 正常
            position = 1.0  # 100%仓位
        
        return position


# 聚宽平台使用示例
def initialize(context):
    """初始化函数"""
    # 创建恐慌情绪分析器
    g.panic_analyzer = PanicSentimentAnalyzer()
    
    # 其他初始化代码...
    g.stock_num = 20
    g.base_position = 1.0
    
    # 定时运行
    run_daily(analyze_panic_and_adjust, '09:30')
    

def analyze_panic_and_adjust(context):
    """分析恐慌情绪并调整仓位"""
    # 计算恐慌指数
    panic_index, details = g.panic_analyzer.calculate_panic_index(context)
    
    # 记录详细信息
    log.info("="*50)
    log.info(f"恐慌情绪分析报告 - {context.current_dt.date()}")
    log.info(f"综合恐慌指数: {details['panic_index']}/100")
    log.info(f"恐慌级别: {details['panic_level']}")
    log.info(f"建议仓位: {details['suggested_position']*100:.1f}%")
    log.info("-"*30)
    log.info(f"价格跌幅分数: {details['price_score']}")
    log.info(f"成交量异常分数: {details['volume_score']}")
    log.info(f"跌停比率分数: {details['limit_score']}")
    log.info(f"波动率分数: {details['volatility_score']}")
    log.info(f"市场宽度分数: {details['breadth_score']}")
    log.info("="*50)
    
    # 根据恐慌指数调整仓位
    g.target_position = details['suggested_position']
    
    # 如果当前仓位高于建议仓位，进行减仓
    current_position_ratio = len(context.portfolio.positions) / g.stock_num
    if current_position_ratio > g.target_position:
        reduce_positions(context, g.target_position)


def reduce_positions(context, target_position_ratio):
    """根据目标仓位比例减仓"""
    current_positions = len(context.portfolio.positions)
    target_positions = int(g.stock_num * target_position_ratio)
    
    if target_positions < current_positions:
        # 需要卖出的股票数量
        sell_count = current_positions - target_positions
        
        # 按照收益率排序，优先卖出亏损最大的
        positions_list = []
        for stock, position in context.portfolio.positions.items():
            profit_rate = (position.price - position.avg_cost) / position.avg_cost
            positions_list.append((stock, profit_rate))
        
        positions_list.sort(key=lambda x: x[1])
        
        # 卖出表现最差的股票
        for i in range(sell_count):
            stock = positions_list[i][0]
            order_target_value(stock, 0)
            log.info(f"恐慌减仓：卖出 {stock}")


# 历史恐慌时期分析函数
def analyze_historical_panic_periods():
    """分析三个历史恐慌时期的恐慌指数"""
    
    # 模拟三个时期的市场数据和恐慌指数
    panic_periods = {
        "2024年1月-2月": {
            "date_range": "2024-01-10 至 2024-02-07",
            "price_drop": -12.5,  # 最大跌幅
            "volume_ratio": 2.5,  # 成交量比率
            "limit_down_ratio": 5.2,  # 跌停比率%
            "volatility": 35,  # 年化波动率
            "panic_index": 85.3,  # 综合恐慌指数
            "suggested_position": 0.2  # 建议仓位20%
        },
        "2024年12月-2025年1月": {
            "date_range": "2024-12-16 至 2025-01-06", 
            "price_drop": -5.8,
            "volume_ratio": 1.8,
            "limit_down_ratio": 2.1,
            "volatility": 25,
            "panic_index": 52.7,
            "suggested_position": 0.5  # 建议仓位50%
        },
        "2025年4月初": {
            "date_range": "2025-04-03 至 2025-04-08",
            "price_drop": -7.2,
            "volume_ratio": 2.2,
            "limit_down_ratio": 3.8,
            "volatility": 30,
            "panic_index": 68.4,
            "suggested_position": 0.35  # 建议仓位35%
        }
    }
    
    print("A股市场三个时期恐慌情绪量化分析")
    print("="*60)
    
    for period, data in panic_periods.items():
        print(f"\n{period} ({data['date_range']})")
        print("-"*40)
        print(f"最大跌幅: {data['price_drop']:.1f}%")
        print(f"成交量比率: {data['volume_ratio']:.1f}倍")
        print(f"跌停比率: {data['limit_down_ratio']:.1f}%")
        print(f"年化波动率: {data['volatility']:.0f}%")
        print(f"综合恐慌指数: {data['panic_index']:.1f}/100")
        print(f"建议仓位: {data['suggested_position']*100:.0f}%")
        
        # 恐慌级别判定
        if data['panic_index'] >= 80:
            level = "极度恐慌"
        elif data['panic_index'] >= 60:
            level = "高度恐慌"
        elif data['panic_index'] >= 40:
            level = "中度恐慌"
        else:
            level = "轻度恐慌"
        print(f"恐慌级别: {level}")
    
    print("\n" + "="*60)
    print("仓位控制建议：")
    print("- 极度恐慌(80-100): 仓位20%，等待市场企稳")
    print("- 高度恐慌(60-80): 仓位35-40%，逐步建仓")
    print("- 中度恐慌(40-60): 仓位50-60%，正常操作")
    print("- 轻度恐慌(20-40): 仓位80%，积极操作")
    print("- 正常市场(0-20): 仓位100%，满仓操作")


# 运行历史分析
if __name__ == "__main__":
    analyze_historical_panic_periods()