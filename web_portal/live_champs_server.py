from flask import request
from flask import Flask, url_for, render_template
import json
import pandas as pd
import random
import urllib.request 
import sys, os
from functools import partial
from itertools import product
import socket
# Libs
import numpy as np
from random import randint, shuffle
# Local
import neat.nn
import _pickle as pickle

app = Flask(__name__)
peer_data = {}
config = {}

@app.route('/')
def root():
    return render_template("trade_hist.html")
    
@app.route("/trade_hist")
def get_trade_hist(request):
    frame = pd.read_csv('./live_hist/latest_hist')
    return frame.to_json()


@app.route("/test_net_balance")
def test_trade_hist_chart():
    frame = pd.read_csv("../trade_hists/8604_hist.txt")
    print(frame.to_json())
    return frame.to_json()

@app.route("/exchanges")
def get_exchanges():
    current_exchanges = os.listdir("../trade_hists")
    print(current_exchanges)
    return json.dumps(current_exchanges)


@app.route("/store_champ")
def store_net_json(request):
    return 
    


# If run as script.

if __name__ == '__main__':
    app.run()
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
