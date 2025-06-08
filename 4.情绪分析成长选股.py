#导入函数库
from jqdata import *
from jqfactor import *
import numpy as np
import pandas as pd



#初始化函数 
def initialize(context):
    # 设定基准
    set_benchmark('000905.XSHG')
    # 用真实价格交易
    set_option('use_real_price', True)
    # 打开防未来函数
    set_option("avoid_future_data", True)
    # 将滑点设置为0
    set_slippage(FixedSlippage(0.02))
    # 设置交易成本万分之三，不同滑点影响可在归因分析中查看
    set_order_cost(OrderCost(open_tax=0, close_tax=0.001, open_commission=0.0001, close_commission=0.0001, close_today_commission=0, min_commission=5),type='stock')
    # 过滤order中低于error级别的日志
    log.set_level('order', 'error')
    #初始化全局变量
    g.no_trading_today_signal = False
    g.stock_num = 20
    g.hold_list = [] #当前持仓的全部股票    
    g.yesterday_HL_list = [] #记录持仓中昨日涨停的股票
    g.factor_list = [
        (#ARBR-SGAI-NPtTORttm-RPps
            [
               'circulating_market_cap', 
               'book_to_price_ratio', 
               'non_linear_size', 
               'residual_volatility' 
            ],
            [
               -103.18798601126404,
               50.61847706789803,
               -420.7751809609919,
               2.8591316971197074e-08
            ]
        ),
    ]
    
    # 初始化恐慌情绪分析器
    g.panic_analyzer = PanicSentimentAnalyzer()
    g.target_position_ratio = 1.0  # 目标仓位比例
    g.panic_index = 0  # 当前恐慌指数
    
    # 设置交易运行时间
    run_daily(prepare_stock_list, '9:05')
    run_daily(analyze_panic_sentiment, '9:25')  # 分析恐慌情绪
    run_daily(daily_adjustment, '9:30')  # 每日调仓
    run_daily(check_limit_up, '10:00')  # 检查涨停股
    run_daily(close_account, '14:30')
    #run_daily(print_position_info, '15:10')



#恐慌情绪分析器类
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
        try:
            # 获取上证指数数据
            index_code = '000001.XSHG'
            price_data = attribute_history(index_code, lookback_days + 1, '1d', ['close'])
            
            if len(price_data) < 2:
                return 0
            
            # 计算不同时间跨度的跌幅
            drop_1d = (price_data['close'].iloc[-1] - price_data['close'].iloc[-2]) / price_data['close'].iloc[-2] * 100
            drop_5d = (price_data['close'].iloc[-1] - price_data['close'].iloc[-6]) / price_data['close'].iloc[-6] * 100 if len(price_data) >= 6 else 0
            drop_20d = (price_data['close'].iloc[-1] - price_data['close'].iloc[0]) / price_data['close'].iloc[0] * 100
            
            # 连续下跌天数
            consecutive_drops = 0
            for i in range(len(price_data)-1, 0, -1):
                if price_data['close'].iloc[i] < price_data['close'].iloc[i-1]:
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
        except:
            return 0
    
    def _calculate_volume_spike_score(self, context, lookback_days):
        """计算成交量异常分数"""
        try:
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
        except:
            return 0
    
    def _calculate_limit_down_score(self, context):
        """计算跌停比率分数"""
        try:
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
        except:
            return 0
    
    def _calculate_volatility_score(self, context, lookback_days):
        """计算波动率分数"""
        try:
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
        except:
            return 0
    
    def _calculate_market_breadth_score(self, context):
        """计算市场宽度分数（上涨下跌股票比例）"""
        try:
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
        except:
            return 50
    
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



#1-1 准备股票池
def prepare_stock_list(context):
    #获取已持有列表
    g.hold_list= []
    for position in list(context.portfolio.positions.values()):
        stock = position.security
        g.hold_list.append(stock)
    #获取昨日涨停列表
    if g.hold_list != []:
        df = get_price(g.hold_list, end_date=context.previous_date, frequency='daily', fields=['close','high_limit'], count=1, panel=False, fill_paused=False)
        df = df[df['close'] == df['high_limit']]
        g.yesterday_HL_list = list(df.code)
    else:
        g.yesterday_HL_list = []
    #判断今天是否为账户资金再平衡的日期（4月5日-30日为财报季，空仓规避风险）
    g.no_trading_today_signal = today_is_between(context, '04-05', '04-30')

#1-2 恐慌情绪分析
def analyze_panic_sentiment(context):
    """分析市场恐慌情绪并设置目标仓位"""
    # 计算恐慌指数
    panic_index, details = g.panic_analyzer.calculate_panic_index(context)
    g.panic_index = panic_index
    g.target_position_ratio = details['suggested_position']
    
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
    
#1-3 选股模块
def get_stock_list(context):
    #指定日期防止未来数据
    yesterday = context.previous_date
    today = context.current_dt
    #获取初始列表
    initial_list = get_all_securities('stock', today).index.tolist()
    initial_list = filter_new_stock(context, initial_list)
    initial_list = filter_kcbj_stock(initial_list)
    initial_list = filter_st_stock(initial_list)
    final_list = []
    #MS
    for factor_list,coef_list in g.factor_list:
        factor_values = get_factor_values(initial_list,factor_list, end_date=yesterday, count=1)
        df = pd.DataFrame(index=initial_list, columns=factor_values.keys())
        for i in range(len(factor_list)):
            df[factor_list[i]] = list(factor_values[factor_list[i]].T.iloc[:,0])
        df = df.dropna()
        df['total_score'] = 0
        for i in range(len(factor_list)):
            df['total_score'] += coef_list[i]*df[factor_list[i]]
        df = df.sort_values(by=['total_score'], ascending=False) #分数越高即预测未来收益越高，排序默认降序
        complex_factor_list = list(df.index)[:int(0.1*len(list(df.index)))]
        lst = complex_factor_list[:]
        lst = filter_paused_stock(lst)
        lst = lst[:min(g.stock_num, len(lst))]
        for stock in lst:
            if stock not in final_list:
                final_list.append(stock)
    return final_list

#1-4 每日调整持仓
def daily_adjustment(context):
    if g.no_trading_today_signal == False:
        # 根据恐慌指数调整目标持仓数量
        target_num = int(g.stock_num * g.target_position_ratio)
        log.info(f"当前恐慌指数: {g.panic_index:.1f}, 目标持仓数: {target_num}")
        
        #获取应买入列表 
        target_list = get_stock_list(context)[:target_num]  # 根据仓位比例调整持仓数
        log.info("选择股票：")
        log.info(target_list)
        
        #调仓卖出
        for stock in g.hold_list:
            if (stock not in target_list) and (stock not in g.yesterday_HL_list):
                log.info("卖出[%s]" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
            else:
                log.info("已持有[%s]" % (stock))
        
        # 如果当前持仓数超过目标，优先卖出表现最差的
        current_positions = len(context.portfolio.positions)
        if current_positions > target_num:
            positions_list = []
            for stock, position in context.portfolio.positions.items():
                if stock not in g.yesterday_HL_list:  # 保护涨停股
                    profit_rate = (position.price - position.avg_cost) / position.avg_cost
                    positions_list.append((stock, profit_rate))
            
            positions_list.sort(key=lambda x: x[1])  # 按收益率排序
            
            # 卖出多余的股票
            for i in range(current_positions - target_num):
                if i < len(positions_list):
                    stock = positions_list[i][0]
                    order_target_value(stock, 0)
                    log.info(f"恐慌减仓：卖出 {stock}")
        
        #调仓买入
        position_count = len(context.portfolio.positions)
        if target_num > position_count:
            value = context.portfolio.cash / (target_num - position_count)
            for stock in target_list:
                # 检查是否已持有该股票，避免访问不存在的position
                if stock not in context.portfolio.positions:
                    if open_position(stock, value):
                        if len(context.portfolio.positions) == target_num:
                            break

#1-5 调整昨日涨停股票（上午10点执行）
def check_limit_up(context):
    now_time = context.current_dt
    if g.yesterday_HL_list != []:
        #对昨日涨停股票观察，如不涨停则卖出，如果涨停即使不在应买入列表仍暂时持有
        for stock in g.yesterday_HL_list:
            current_data = get_price(stock, end_date=now_time, frequency='1m', fields=['close','high_limit'], skip_paused=False, fq='pre', count=1, panel=False, fill_paused=True)
            if current_data.iloc[0,0] < current_data.iloc[0,1]:
                log.info("[%s]涨停打开，卖出" % (stock))
                position = context.portfolio.positions[stock]
                close_position(position)
            else:
                log.info("[%s]涨停，继续持有" % (stock))



#2-1 过滤停牌股票
def filter_paused_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list if not current_data[stock].paused]

#2-2 过滤ST及其他具有退市标签的股票
def filter_st_stock(stock_list):
    current_data = get_current_data()
    return [stock for stock in stock_list
            if not current_data[stock].is_st
            and 'ST' not in current_data[stock].name
            and '*' not in current_data[stock].name
            and '退' not in current_data[stock].name]

#2-3 过滤科创北交股票
def filter_kcbj_stock(stock_list):
    for stock in stock_list[:]:
        if stock[0] == '4' or stock[0] == '8' or stock[:2] == '68':
            stock_list.remove(stock)
    return stock_list

#2-4 过滤涨停的股票
def filter_limitup_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] < current_data[stock].high_limit]

#2-5 过滤跌停的股票
def filter_limitdown_stock(context, stock_list):
    last_prices = history(1, unit='1m', field='close', security_list=stock_list)
    current_data = get_current_data()
    return [stock for stock in stock_list if stock in context.portfolio.positions.keys()
            or last_prices[stock][-1] > current_data[stock].low_limit]

#2-6 过滤次新股
def filter_new_stock(context,stock_list):
    yesterday = context.previous_date
    return [stock for stock in stock_list if not yesterday - get_security_info(stock).start_date < datetime.timedelta(days=375)]



#3-1 交易模块-自定义下单
def order_target_value_(security, value):
    if value == 0:
        log.debug("Selling out %s" % (security))
    else:
        log.debug("Order %s to value %f" % (security, value))
    return order_target_value(security, value)

#3-2 交易模块-开仓
def open_position(security, value):
    order = order_target_value_(security, value)
    if order != None and order.filled > 0:
        return True
    return False

#3-3 交易模块-平仓
def close_position(position):
    security = position.security
    order = order_target_value_(security, 0)  # 可能会因停牌失败
    if order != None:
        if order.status == OrderStatus.held and order.filled == order.amount:
            return True
    return False



#4-1 判断今天是否为账户资金再平衡的日期（财报季空仓）
def today_is_between(context, start_date, end_date):
    today = context.current_dt.strftime('%m-%d')
    if (start_date <= today) and (today <= end_date):
        return True
    else:
        return False

#4-2 财报季清仓（4月5日-30日）
def close_account(context):
    if g.no_trading_today_signal == True:
        if len(g.hold_list) != 0:
            for stock in g.hold_list:
                position = context.portfolio.positions[stock]
                close_position(position)
                log.info("财报季清仓卖出[%s]" % (stock))

#4-3 打印每日持仓信息
def print_position_info(context):
    #打印当天成交记录
    trades = get_trades()
    for _trade in trades.values():
        print('成交记录：'+str(_trade))
    #打印账户信息
    for position in list(context.portfolio.positions.values()):
        securities=position.security
        cost=position.avg_cost
        price=position.price
        ret=100*(price/cost-1)
        value=position.value
        amount=position.total_amount    
        print('代码:{}'.format(securities))
        print('成本价:{}'.format(format(cost,'.2f')))
        print('现价:{}'.format(price))
        print('收益率:{}%'.format(format(ret,'.2f')))
        print('持仓(股):{}'.format(amount))
        print('市值:{}'.format(format(value,'.2f')))
        print('———————————————————————————————————')
    print('———————————————————————————————————————分割线————————————————————————————————————————')