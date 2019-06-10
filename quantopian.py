"""
This is a template algorithm on Quantopian for you to adapt and fill in.
"""
from quantopian.algorithm import attach_pipeline, pipeline_output
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.factors import AverageDollarVolume, SimpleMovingAverage
import talib
from sklearn import preprocessing, cross_validation, svm
from sklearn.linear_model import LinearRegression, Ridge
import numpy as np
def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    context.last = context.portfolio.portfolio_value
    log.info(str(context.portfolio.cash))
    #set_long_only()
    # Rebalance every day, 1 hour after market open.
    schedule_function(my_rebalance, date_rules.every_day(), time_rules.market_close(minutes=180))
    schedule_function(weekly_fuckoff, date_rules.every_day(), time_rules.market_close(minutes=30))
    schedule_function(my_rebalance_two, date_rules.every_day(), time_rules.market_open(minutes=90))
    '''total_minutes = 6*60 + 30
    for i in range(1, total_minutes):
     Every 30 minutes run schedule
      if i % 15 == 0:
       This will start at 9:31AM and will run every 30 minutes
        schedule_function(
        my_rebalance,
          date_rules.every_day(),
          time_rules.market_open(minutes=i),
          True
      )'''
     
    # Create our dynamic stock selector.

    attach_pipeline(make_pipeline(), 'my_pipeline')
    #context.cant_sell=[]
    #context.can_sell=[]
    #context.take_gains = 1.03
    #context.take_loss = -0.8     
def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation on
    pipeline can be found here: https://www.quantopian.com/help#pipeline-title
    """

    #sma_short = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=20)
    #sma_long = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=80)
    sma_filter = SimpleMovingAverage(inputs=[USEquityPricing.close], window_length=28)
    avrange = (4<=sma_filter<=33)
    
    # Create a dollar volume factor.
    dollar_volume = AverageDollarVolume(window_length=16)
    #over_under = talib.WILLR(USEquityPricing.high, USEquityPricing.low, USEquityPricing.close, timeperiod=7)
    #over_sold = over_under < -92
    #trix_val = talib.TRIX(close, timeperiod = 18)
    #trix_filter = trix_val >0.05
    # Pick the top 1% of stocks ranked by dollar volume.
    mid_dollar_volume = dollar_volume.percentile_between(90,95)
    pipe_screen = (mid_dollar_volume & avrange)
    pipe_columns = {
        #'over_under':over_under,
        #'trix_val':trix_val,
        #'upT':upT,
        #'over_sold':over_sold,
        #'trix_filter': trix_filter,
        'mid_dollar_volume':mid_dollar_volume,
        'sma_filter': sma_filter
    }
    pipe = Pipeline(
        columns = pipe_columns,
        screen = pipe_screen
    )
    return pipe
 
def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    
    context.output = pipeline_output('my_pipeline')
  
    # These are the securities that we are interested in trading each day.
    
    
    context.security_list = context.output.index
    '''
    for symbol in context.cant_sell:
        if symbol in context.portfolio.positions:
            context.can_sell.append(symbol)
            context.cant_sell.remove(symbol)
        else: context.cant_sell.remove(symbol)
       ''' 
    
    
def weekly_fuckoff(context, data):
    log.info('returns = %d' %(context.portfolio.portfolio_value-context.portfolio.starting_cash))
    for symbol in get_open_orders():
        cancel_order(symbol)

def my_assign_weights(context, data):
    """
    Assign weights to securities that we want to order.
    """
    pass
 
def my_rebalance(context,data):
    """
    Execute orders according to our schedule_function() timing.
    
    """
    
    for symbol in context.security_list:
        cash = context.account.buying_power
        if symbol not in context.portfolio.positions:
            '''
            volume= data.history(
                symbol,
                'volume',
                bar_count=100,
                frequency='1d'
                )
                '''
            priceData = data.history(
                symbol,
                ['high','low','close','volume'],
                bar_count= 246,
                frequency='1d'
                )
            lastClose = data.history(symbol, 'close', bar_count = 2, frequency = '1d')[-1]
            lastplus = lastClose*1.04
            #mom = talib.ULTOSC(priceData['high'], priceData['low'], priceData['close'], timeperiod1=7,timeperiod2=14,timeperiod3=28)[-1]
            #ma_15 = moving_15.mean()
            
            #long_ma = moving_45.mean()
            #vol = talib.OBV(priceData['close'], volume)[-1]
            priceData['willr'] = talib.WILLR(priceData['high'], priceData['low'], priceData['close'], timeperiod=4)[-1]
            #rsi = talib.TRIX(priceData['close'], timeperiod = 14)[-1]
            #current_price = data.current(symbol, 'price')
            
            
            if data.can_trade(symbol):
                scores = getML(priceData, 1)
                if scores[0]>.97 and scores[1] > lastplus:
                    log.info("%s con: %f next: %f fuck yeah bro" %(symbol, scores[0], scores[1]))
                    if cash > 10000.0:
                        order_value(symbol, 7000)
                        '''
                        pcount = 0
                        while get_open_orders(symbol) != None:
                            if pcount == 0:
                                log.info("buying %s" %symbol)
                                pcount += 1
                            if pcount > 50:
                                break
                                '''
                        
                        
def my_rebalance_two(context, data):
    highLim = .04
    lowLim = -.03
    if context.portfolio.returns > highLim or context.portfolio.returns< lowLim:
        for sym in context.portfolio.positions:
            holding = context.portfolio.positions[sym].amount
            order(sym, -holding)
            log.info("selling %s" %sym)
            highLim += context.portfolio.returns
    '''                                     
    for sym in context.portfolio.positions:
        costB = float((context.portfolio.positions[sym].cost_basis))*1.2
        costLow = float((context.portfolio.positions[sym].cost_basis))*.90
        now = data.current(sym, 'price')
        if  now > costB or now < costLow:
            holding = context.portfolio.positions[sym].amount
            order(sym, -holding)
            log.info("selling %s" %sym)
          

            scount = 0
            while get_open_orders(symbol) != None:
                if scount == 0:
                    log.info("selling %s" % symbol)
                    scount += 1
                if scount > 0:
                    pass
                '''
def my_record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    pass


def getML(df, days_ahead):
    #df['rollinghigh'] = df['close'].rolling(window = 21, center=False).mean()
    #df['rollinglow'] = df['close'].rolling(window = 8, center=False).mean()
    forecast_val = days_ahead
    forcastCol = 'close'
    df.fillna(value = -99999, inplace = True)
    df['label'] = df[forcastCol].shift(-forecast_val)
    x = np.array(df.drop(['label'], 1))
    x = preprocessing.scale(x)
    futurex = x[-1:]
    x = x[:-forecast_val]
    df.dropna(inplace=True)
    y=np.array(df['label'])
    
    X_train, X_test, y_train, y_test = cross_validation.train_test_split(x, y, test_size=0.05)
    clf = LinearRegression(n_jobs = 1)
    clf.fit(X_train, y_train)
    confidence = clf.score(X_test, y_test)
    future = clf.predict(futurex)
    return(confidence,future)
def handle_data(context,data):

    pass