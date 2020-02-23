
import ccxt


class CcxtWrapper(object):

    keys = []
    all_exchanges = []

    def __init__(self, exchange):
        self.current_exchange = exchange
        return

    def get_exchange_markets(self):
        return self.current_exchange.load_markets()