import hist_service as hs
import datetime, time
import pandas as pd
import numpy as np


class CryptoMarginFolio:
    
    #assume we 
    fees = .002
    buys = 0
    sells = 0
    target_amount = 0.1
    ledger = {}
    start = 0
    def __init__(self, start_amount, coins, base="BTC", save_trades=False):
        self.base_sym = base
        self.ledger[self.base_sym] = start_amount
        for ix in range(len(coins)):
            self.ledger[coins[ix]] = 0.0
        #print(self.ledger)
        self.start = start_amount
        self.hs = hs.HistWorker()
        self.save_trades = save_trades


    def buy_contract(self, c_name, price, leverage):
        amount = self.start * self.target_amount
        if(amount > self.ledger[self.base_sym]):
            return False
        else:
            coin_amount = amount/(price* 1.01)
            the_fee = self.fees * amount
            self.ledger[self.base_sym] -= (amount + the_fee)
            self.ledger[c_name] += coin_amount
            self.buys += 1
            return True


    def sell_coin(self, c_name, price):
        price = price -(price*.01)
        if self.ledger[c_name] != 0.0:
            amount = self.ledger[c_name]
            self.ledger[self.base_sym] += ((amount*price) - ((amount * price)*self.fees))
            self.ledger[c_name] = 0.0
            self.sells += 1
            return True
        else:
            return False

    
    def get_total_btc_value(self, e_prices):
        for c in self.ledger.keys():
            if self.ledger[c] != 0.0 and c != self.base_sym:
                current_price = e_prices[c]
                self.sell_coin(c, current_price)
        return self.ledger[self.base_sym], self.buys, self.sells
        
    def get_total_btc_value_no_sell(self, e_prices):
        btcval = self.ledger[self.base_sym]
        for c in self.ledger.keys():
            if self.ledger[c] != 0.0 and c != self.base_sym:
                current_price = e_prices[c]
                btc_amt = current_price * self.ledger[c]
                btcval += btc_amt
                #print(c, " ", btc_amt)
        return btcval, self.ledger[self.base_sym]

    def evaluate_output(self, out, coin, price):
        if (out == 1.0):
            self.buy_coin(coin, price)
        elif(out==.5):
            return
        else:
            self.sell_coin(price,coin)

