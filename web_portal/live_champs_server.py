from flask import request
from flask import Flask, url_for
import json
import pandas as pd
import random
import urllib.request 
import sys, os
from functools import partial
from itertools import product
import socket
from trading_purples import PurpleTrader
# Libs
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
from random import randint, shuffle
# Local
import neat.nn
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat_torch import ESNetwork



class LiqMaster2000:
    app = Flask(__name__, static_url_path='champ')
    peer_data = {}
    config = {}
    
    '''
    def get_ip(self):
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            # doesn't even have to be reachable
            s.connect(('10.255.255.255', 1))
            IP = s.getsockname()[0]
        except:
            IP = '127.0.0.1'
        finally:
            s.close()
        return IP
    '''
    @app.route("/trade_hist")
    def get_trade_hist(self, request):
        frame = pd.read_csv('./live_hist/latest_hist')
        return frame.to_json()

    @app.route("/exchanges")
    def get_exchanges(self):
        current_exchanges = os.listdir("./trade_hists")
        return json.dumps(current_exchanges)


    @app.route("/store_champ")
    def store_net_json(self, request):
        return 
    


# If run as script.

if __name__ == '__main__':
    cs = LiqMaster2000()
    '''
    task = PurpleTrader(13)
    winner = run_pop(task, 21)[0]
    print('\nBest genome:\n{!s}'.format(winner))
    # Verify network output against training data.
    print('\nOutput:')
    cppn = neat.nn.FeedForwardNetwork.create(winner, task.config)
    network = ESNetwork(task.subStrate, cppn, task.params)
    with open('local_winner.pkl', 'wb') as output:
        pickle.dump(cppn, output)
    #draw_net(cppn, filename="es_trade_god")
    winner_net = network.create_phenotype_network_nd('dabestest.png')  # This will also draw winner_net.
    
    # Save CPPN if wished reused and draw it to file.
    #draw_net(cppn, filename="es_trade_god")
'''
