from flask import request, jsonify
from flask import Flask, url_for, render_template
from flask_cors import CORS
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
cors = CORS(app, resources={r"*": {"origins": "*"}})
peer_data = {}
config = {}

@app.route('/')
def root():
    return render_template("trade_hist.html")

@app.route('/high_chart')
def high_chart():
    hist_files = os.listdir("../trade_hists/ftx_live/")
    print(hist_files)
    return render_template("hc_per_symbol.html", data=hist_files)

@app.route('/high_chart_single')
def high_chart_single():
    return render_template("hc_single.html")

@app.route("/bootstrap_sample")
def bootstrap_sample():
    return render_template("bstrap_template.html")

@app.route("/trade_hist")
def get_trade_hist(request):
    frame = pd.read_csv('./live_hist/latest_hist')
    return frame.to_json()

@app.route("/trade_hist_all/<genome>/all")
def get_all_trade_hist_genome(genome):
    return jsonify(get_genome_performance_backtest(genome))

@app.route("/trade_hist_avg/<genome>/all")
def get_avg_trade_hist_genome(genome):
    return jsonify(get_genome_performance_live(genome))


@app.route("/test_single_portfolio")
def test_trade_hist_chart():
    hist_files = os.listdir("../trade_hists/binance")
    data_dict = {}
    for f in hist_files:
        frame = pd.read_csv("../trade_hists/binance/" + f)
        data_dict[f] = [list(v.values()) for v in frame.T.to_dict().values()]
    return jsonify(data_dict)

@app.route("/exchanges")
def get_exchanges():
    current_exchanges = os.listdir("../trade_hists")
    print(current_exchanges)
    return json.dumps(current_exchanges)

@app.route("/exchange")
def get_exchange(exchange="binance"):
    hist_files = os.listdir("../trade_hists/binance/")
    hist_data = []
    return hist_files

def get_genome_performance_backtest(g_name, data_set):
    hist_files = os.listdir("../trade_hists/ftx_" + data_set + "/" + g_name)
    data_dict = {}
    for f in hist_files:
        if f != ".DS_Store":
            total_balances += 1
            frame = pd.read_csv("../trade_hists/ftx_" + data_set + "/" + g_name +"/" + f)
            if (frame["1"][0]) < frame["1"][len(frame) - 1]:
                print(f, " ", frame["1"][0], frame["1"][len(frame) - 1])
            data_dict[f] = [list(v.values()) for v in frame.T.to_dict().values()]
    return data_dict

def get_genome_performance_live(g_name, data_set):
    hist_files = os.listdir("../trade_hists/ftx_"+ data_set+"/" + g_name)
    data_dict = {}
    first_loop = True
    total_balances = 0
    for f in hist_files:
        if f != ".DS_Store":
            total_balances += 1
            frame = pd.read_csv("../trade_hists/ftx_"+data_set+"/"+ g_name +"/" + f)
            if first_loop == True:
                df_avg = frame.copy()
                first_loop = False
            else:
                df_avg["1"] = (df_avg["1"] + frame["1"])
            if (frame["1"][0]) < frame["1"][len(frame) - 1]:
                print(f, " ", frame["1"][0], frame["1"][len(frame) - 1])
            #data_dict[f] = [list(v.values()) for v in frame.T.to_dict().values()]
    df_avg["1"] = df_avg["1"] / total_balances
    df_avg.dropna(inplace=True)
    data_dict["total"] = [list(v.values()) for v in df_avg.T.to_dict().values()]
    return data_dict

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
