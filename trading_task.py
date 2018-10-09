
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product

# Libs
import numpy as np
# Libraries
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
# Local
from peas.peas.networks.rnn import NeuralNetwork
from peas.peas.methods import hyperneat.HyperNEATDeveloper, hyperneat.Substrate

from peas.peas.methods.neat import NEATPopulation, NEATGenotype
from peas.peas.methods.evolution import SimplePopulation

class Trading_Task:

    EPSILON = 1e-100

    start_idx = 0
    
    portfolio_list = []


    def __init__(self):
        self.hs = HistWorker()
        self.end_idx = len(self.hs.currentHists["DASH"])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*self.hist_shaped[0].shape[1]
        self.outputs = self.hs.hist_shaped.shape[0] # times by three for buy | sell | hodl(pass)
        #self.port = CryptoFolio(1)

    
    def set_portfolio_keys(folio):
        for k in self.hs.currentHists.keys:
            folio.ledger[k] = 0

    def eval_out(sigmoid_out):
        if 

    def get_one_bar_input_2d(self, end_idx):
        active = {}
        for x in range(0, self.outputs):
            active[x] = self.hs.hist_shaped[x][idx]
        return active

    def evaluate(self, network, verbose=False):
        portfolio = CryptoFolio(1)
        active = {}
        if not isinstance(network, NeuralNetwork):
            network = NeuralNetwork(network)
        for z in range(0, 14):
            '''
            if(z == 0):
                old_idx = 1
            else:
                old_idx = z * 5
            new_idx = (z + 1) * 5
            '''
            active = self.get_one_bar_input(z)
            results[z] = network.feed(active)

        for i in range(0, 14):
            out = results[i]
            for x in range(0, self.outputs):
                sym = self.hs.coin_dict[x]
                if(out[x] == 1.0):
                    portfolio.buy_coin(sym, self.hs.currentHists[sym][x]['close'])
                elif(out[x] == 0.0):
                    portfolio.sell_coin(sym)
        end_ts = self.hs.hist_shaped[0][14][0]
        result_val = portfolio.get_total_btc_value(int(end_ts))
        print(results)


    def run(generations=100, popsize=100):
                
        substrate = Substrate((self.hs.hist_shaped.shape[0],))
        #substrate.add_nodes(, 'l')
        substrate.add_connections('l', 'l')
        geno = lambda: NEATGenotype(feedforward=True, inputs=self.inputs, weight_range=(-3.0, 3.0), 
                                       prob_add_conn=0.3, prob_add_node=0.03,
                                       types=['sin', 'ident', 'gauss', 'sigmoid', 'abs'])
        pop = NEATPopulation(geno, popsize=popsize, target_species=8)
        developer = HyperNEATDeveloper(substrate=substrate, add_deltas=False, sandwich=False, node_type="sigmoid")
        results = pop.epoch(generations=generations, evaluator=partial(evaluate, task=self, developer=developer), solution=self)
        return results



if __name__ == '__main__':
    do_it = run()

