
import ccxt


class CcxtWrapper(object):

    keys = []
    all_exchanges = []
    current_exchange = ""

    def __init__(self):
        return