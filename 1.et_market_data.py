def get_market_data(stock_list, start_date, end_date):
    data = get_price(
        stock_list, 
        start_date=start_date, 
        end_date=end_date,
        fields=['open', 'close', 'high', 'low', 'volume', 'money'],
        frequency='daily',
        fq='pre'
    )