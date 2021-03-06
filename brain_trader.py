# -*- coding: utf-8 -*-
"""
Created on Sun Aug 13 08:07:30 2017

@author: nick
"""
import pickle
import time
import pandas as pd
import numpy as np
from poloniex import Poloniex
from datetime import date, timedelta, datetime
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
from random import randint, shuffle
import requests
from pytorch_neat.cppn import create_cppn
# Local
import neat.nn
import neat
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
#polo = Poloniex('key', 'secret')

class LiveTrader:
    params = {"initial_depth": 3,
            "max_depth": 4,
            "variance_threshold": 0.00013,
            "band_threshold": 0.00013,
            "iteration_level": 3,
            "division_threshold": 0.00013,
            "max_weight": 8.0,
            "activation": "tanh"}


    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')

    def __init__(self, ticker_len, target_percent, hd, base_sym="BTC"):
        self.base_sym = base_sym
        self.load_polo_client()
        self.hd = hd
        self.target_percent = target_percent
        self.ticker_len = ticker_len
        self.end_ts = datetime.now()+timedelta(seconds=(ticker_len*55))
        self.hs = HistWorker()
        self.refresh_data()
        self.tickers = self.polo.returnTicker()
        self.refresh_balances()
        self.sellCoins()
        self.set_target()
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1])
        self.outputs = self.hs.hist_shaped.shape[0]
        self.end_idx = len(self.hs.hist_shaped[0]) -1
        self.make_shapes()
        self.load_net()
        self.poloTrader()

    def load_polo_client(self):
        keys = self.get_keys()
        self.polo = Poloniex(keys[0], keys[1])

    def purge_polo_client(self):
        self.polo = None

    def load_net(self):
        champ_file = open("./champ_data/latest_greatest.pkl",'rb')
        g = pickle.load(champ_file)
        #file.close()
        the_cppn = neat.nn.FeedForwardNetwork.create(g, self.config)
        self.cppn = the_cppn

    def refresh_data(self):
        try:
            self.hs.pull_polo_usd_live(21)
            self.hs.combine_live_usd_frames()
        except Exception as e:
            print(e)
            time.sleep(360)
            self.refresh_data()

    def refresh_balances(self):
        try:
            self.bal = self.polo.returnCompleteBalances()
        except Exception as e:
            print(e)
            time.sleep(360)
            self.refresh_balances()

    def get_one_bar_input_2d(self):
        master_active = []
        try:
            for x in range(0, self.hd):
                active = []
                #print(self.outputs)
                for y in range(0, self.outputs):
                    sym_data = self.hs.hist_shaped[y][self.end_idx-x]
                    #print(len(sym_data))
                    active += sym_data.tolist()
                master_active.append(active)
        except:
            print("error getting look back data")
            self.refresh_data()
            self.get_one_bar_input_2d()
        #print(active)
        return master_active

    def closeOrders(self):
        try:
            orders = self.polo.returnOpenOrders()
        except Exception as  e:
            print(e)
            print('error getting open orers')
            time.sleep(360)
            self.closeOrders()
        for o in orders:
            if orders[o] != []:
                try:
                    ordnum = orders[o][0]['orderNumber']
                    self.polo.cancelOrder(ordnum)
                except Exception as  e:
                    print(e)
                    print('error closing')



    def sellCoins(self):
        for b in self.tickers:
            if(b.split("_")[0] == self.base_sym):
                price = self.get_price(b)
                price = price - (price * .005)
                self.sell_coin(b, price)

    def buy_coin(self, coin, price):
        amt = self.target / price
        if(self.bal[self.base_sym]["available"] > self.target):
            try:
                self.polo.buy(coin, price, amt)
                print("buying: ", coin)
            except Exception as e:
                print("error buying ", coin)
                print(e)
        return

    def sell_coin(self, coin, price):
        if (self.base_sym != "BTC"):
            amt = self.bal[coin.split("_")[1]]["available"]
        else:
            amt = self.bal[coin.split("_")[1]]["btcValue"]
        if (amt*price > .0001):
            try:
                self.polo.sell(coin, price, amt)
                print("selling this shit: ", coin)
            except Exception as e:
                print("error selling ", coin)
                print(e)
        return


    def reset_tickers(self):
        try:
            self.tickers = self.polo.returnTicker()
            self.bal = self.polo.returnCompleteBalances()
        except Exception as  e:
            print(e)
            time.sleep(360)
            self.reset_tickers()
        return
    def get_keys(self):
        with open("./godsplan.txt") as f:
            content = f.readlines()
            content[0] = content[0][:-1]
            if (content[1][-1:] == "\n"):
                content[1] = content[1][:-1]
            return content

    def make_shapes(self):
        sign = 1
        self.out_shapes = []
        self.in_shapes = []
        for ix in range(1,self.outputs+1):
            sign = sign *-1
            self.out_shapes.append((0.0-(sign*.005*ix), 0.0, -1.0))
            for ix2 in range(1,(self.inputs//self.outputs)+1):
                self.in_shapes.append((0.0+(sign*.01*ix2), 0.0-(sign*.01*ix2), 0.0))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)

    def get_price(self, coin):
        return self.tickers[coin]['last']

    def set_target(self):
        total = 0
        full_bal = self.polo.returnCompleteBalances()
        for x in full_bal:
            total += full_bal[x]["btcValue"]
        if(self.base_sym != "BTC"):
            total = total * self.get_price(self.base_sym +"_"+"BTC") * self.target_percent
        print(total)
        self.target = total

    def poloTrader(self):
        self.refresh_balances()
        end_prices = {}
        active = self.get_one_bar_input_2d()
        self.load_net()
        network = ESNetwork(self.subStrate, self.cppn, self.params, self.hd)
        net = network.create_phenotype_network_nd('paper_net.png')
        net.reset()
        sell_syms = []
        buy_syms = []
        buy_signals = []
        sell_signals = []
        self.closeOrders()
        for n in range(1, self.hd):
            net.activate(active[self.hd-n])
        out = net.activate(active[0])
        for x in range(len(out)):
            sym = self.hs.coin_dict[x]
            end_prices[sym] = self.get_price(self.base_sym+"_"+sym)
            if(out[x] > .5):
                buy_signals.append(out[x])
                buy_syms.append(sym)
            if(out[x] < -.5):
                sell_signals.append(out[x])
                sell_syms.append(sym)
        #rng = iter(shuffle(rng))
        sorted_buys = np.argsort(buy_signals)[::-1]
        sorted_sells = np.argsort(sell_signals)
        self.reset_tickers()
        for x in sorted_sells:
            sym = sell_syms[x]
            p = self.get_price(self.base_sym + "_" +sym)
            price = p -(p*.005)
            self.sell_coin(self.base_sym + "_" + sym, price)
        for x in sorted_buys:
            sym = buy_syms[x]
            self.target_percent = .1 + out[x] - .45
            p = self.get_price(self.base_sym + "_" +sym)
            price = p*1.005
            self.buy_coin(self.base_sym + "_" +sym, price)
        if datetime.now() >= self.end_ts:
            return
        else:
            self.purge_polo_client()
            time.sleep(self.ticker_len)
            self.load_polo_client()
        self.refresh_data()
        self.make_shapes()
        #self.closeOrders()
        self.poloTrader()

class PaperTrader:
    params = {"initial_depth": 3,
            "max_depth": 4,
            "variance_threshold": 0.00013,
            "band_threshold": 0.00013,
            "iteration_level": 3,
            "division_threshold": 0.00013,
            "max_weight": 8.0,
            "activation": "tanh"}


    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')
    in_shapes = []
    out_shapes = []
    def __init__(self, ticker_len, start_amount, histdepth, base_sym):
        self.trade_hist = {}
        self.polo = Poloniex()
        self.hd = histdepth
        self.ticker_len = ticker_len
        self.end_ts = datetime.now()+timedelta(seconds=(ticker_len*24))
        self.start_amount = start_amount
        self.hs = HistWorker()
        self.hs.combine_live_usd_frames()
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.hist_shaped[0])-1
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1])
        self.outputs = self.hs.hist_shaped.shape[0]
        self.folio = CryptoFolio(start_amount, list(self.hs.currentHists.keys()), base_sym)
        self.base_sym = base_sym
        sign = 1
        for ix in range(1,self.outputs+1):
            sign = sign *-1
            self.out_shapes.append((0.0-(sign*.005*ix), 0.0, -1.0))
            for ix2 in range(1,(self.inputs//self.outputs)+1):
                self.in_shapes.append((0.0+(sign*.01*ix2), 0.0-(sign*.01*ix2), 0.0))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
        self.load_net()
        print(self.hs.coin_dict)
        self.poloTrader()

    def refresh_data(self):
        try:
            self.hs.pull_polo_usd_live(21)
            self.hs.combine_live_usd_frames()
            self.end_idx = len(self.hs.hist_shaped[0])-1
        except Exception as e:
            print(e)
            time.sleep(360)
            self.refresh_data()
        return

    def load_net(self):
        champ_file = open("./champ_data/latest_greatest.pkl",'rb')
        g = pickle.load(champ_file)
        #file.close()
        the_cppn = neat.nn.FeedForwardNetwork.create(g, self.config)
        self.cppn = the_cppn

    def make_shapes(self):
        self.in_shapes = []
        self.out_shapes = []
        sign = 1
        for ix in range(1,self.outputs+1):
            sign = sign *-1
            self.out_shapes.append((0.0-(sign*.005*ix), -1.0, -1.0))
            for ix2 in range(1,(self.inputs//self.outputs)+1):
                self.in_shapes.append((0.0+(sign*.01*ix2), 0.0-(sign*.01*ix2), 1.0))

    def reset_tickers(self):
        try:
            self.tickers = self.polo.returnTicker()
        except Exception as e:
            time.sleep(360)
            self.reset_tickers()
        return
    def get_price(self, coin):
        return self.tickers[coin]['last']

    def get_current_balance(self):
        #self.refresh_data()
        self.reset_tickers()
        c_prices = {}
        for s in self.hs.currentHists.keys():
            if s != self.base_sym:
                c_prices[s] = self.get_price(self.base_sym + "_"+s)
        return self.folio.get_total_btc_value_no_sell(c_prices)

    def get_one_bar_input_2d(self):
        master_active = []
        try:
            for x in range(0, self.hd):
                active = []
                #print(self.outputs)
                for y in range(0, self.outputs):
                    sym_data = self.hs.hist_shaped[y][self.end_idx-x]
                    #print(len(sym_data))
                    active += sym_data.tolist()
                master_active.append(active)
        except:
            self.refresh_data()
            self.get_one_bar_input_2d()
        #print(active)
        return master_active

    def poloTrader(self):
        try:
            trade_df = pd.read_json("./live_hist/json_hist.json")
        except Exception as e:
            trade_df = pd.DataFrame()
        end_prices = {}
        active = self.get_one_bar_input_2d()
        self.load_net()
        sub = Substrate(self.in_shapes, self.out_shapes)
        net = ESNetwork(sub, self.cppn, self.params, self.hd)
        network = net.create_phenotype_network_nd('paper_net.png')
        sell_syms = []
        buy_syms = []
        buy_signals = []
        sell_signals = []
        for n in range(1, self.hd):
            network.activate(active[self.hd-n])
        out = network.activate(active[0])
        self.reset_tickers()
        for x in range(len(out)):
            sym = self.hs.coin_dict[x]
            end_prices[sym] = self.get_price(self.base_sym+"_"+sym)
            if(out[x] > .5):
                buy_signals.append(out[x])
                buy_syms.append(sym)
            if(out[x] < -.5):
                sell_signals.append(out[x])
                sell_syms.append(sym)
        #rng = iter(shuffle(rng))
        sorted_buys = np.argsort(buy_signals)[::-1]
        sorted_sells = np.argsort(sell_signals)
        for x in sorted_sells:
            try:
                sym = sell_syms[x]
                p = end_prices[sym]
                print("selling: ", sym)
                self.folio.sell_coin(sym, p)
            except Exception as e:
                print("error placing order")
        for x in sorted_buys:
            try:
                sym = buy_syms[x]
                p = end_prices[sym]
                print("buying: ", sym)
                self.folio.buy_coin(sym, p)
            except Exception as e:
                print("error placing order")
        '''
        self.trade_hist["date"] = datetime.now()
        self.trade_hist["portfoliovalue"] = self.folio.get_total_btc_value_no_sell(end_prices)[0] 
        self.trade_hist["portfolio"] = self.folio.ledger
        self.trade_hist["percentchange"] = ((self.trade_hist["portfoliovalue"] - self.folio.start)/self.folio.start)*100
        trade_df.append(self.trade_hist)
        trade_df.to_json("./live_hist/json_hist.json")
        
        if(self.trade_hist["portfoliovalue"] > self.folio.start *1.1):
            self.folio.start = self.folio.get_total_btc_value(end_prices)[0]
        '''
        if datetime.now() >= self.end_ts:
            port_info = self.folio.get_total_btc_value(end_prices)
            print("total val: ", port_info[0], "btc balance: ", port_info[1])
            return
        
        else:
            print(self.get_current_balance())
            for t in range(2):
                p_vals = self.get_current_balance()
                print("current value: ", p_vals[0], "current holdings: ", p_vals[1])
                time.sleep(self.ticker_len/2)
        self.refresh_data()
        self.poloTrader()



LiveTrader(7200, .2, 34, "USDT")
#PaperTrader(7200, 1000.0 , 34, "USDT")
