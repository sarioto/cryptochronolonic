
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product
from pytorch_neat.cppn import create_cppn
# Libs
import numpy as np
from hist_service import HistWorker
from crypto_evolution import CryptoFolio
from random import randint, shuffle
# Local
import neat.nn
import neat
import _pickle as pickle
from pureples.shared.substrate import Substrate
from pureples.shared.visualize import draw_net
from pureples.es_hyperneat.es_hyperneat import ESNetwork
# Local
class PurpleTrader:

    #needs to be initialized so as to allow for 62 outputs that return a coordinate

    # ES-HyperNEAT specific parameters.
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

    start_idx = 0
    highest_returns = 0
    portfolio_list = []
    def __init__(self, hist_depth, num_gens, gen_count = 1):
        self.hd = hist_depth
        self.num_gens = num_gens + num_gens
        self.gen_count = gen_count
        self.refresh()

    def refresh(self):
        self.in_shapes = []
        self.out_shapes = []
        self.hs = HistWorker()
        self.hs.pull_polo_usd(144)
        self.hs.combine_polo_usd_frames()
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.hist_shaped[0])
        self.but_target = .1
        self.inputs = self.hs.hist_shaped.shape[0]*(self.hs.hist_shaped[0].shape[1])
        self.outputs = self.hs.hist_shaped.shape[0]
        sign = 1
        for ix in range(1,self.outputs+1):
            sign = sign *-1
            self.out_shapes.append((0.0-(sign*.005*ix), 0.0, -1.0))
            for ix2 in range(1,(self.inputs//self.outputs)+1):
                self.in_shapes.append((0.0+(sign*.01*ix2), 0.0-(sign*.01*ix2), 0.0))
        self.subStrate = Substrate(self.in_shapes, self.out_shapes)
        #self.leaf_names.append('bias')
    def set_portfolio_keys(self, folio):
        for k in self.hs.currentHists.keys():
            folio.ledger[k] = 0

    def get_one_epoch_input(self,end_idx):
        master_active = []
        for x in range(0, self.hd):
            active = []
            #print(self.outputs)
            for y in range(0, self.outputs):
                try:
                    sym_data = self.hs.hist_shaped[y][end_idx-x]
                    #print(len(sym_data))
                    active += sym_data.tolist()
                except:
                    print('error')
            master_active.append(active)
        #print(active)
        return master_active

    def evaluate(self, network, es, rand_start, g, verbose=False):
        portfolio_start = 1.0
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict, "USDT")
        end_prices = {}
        buys = 0
        sells = 0
        for z in range(rand_start, rand_start+self.epoch_len):
            #TODO add comments to clarify all the 
            #shit im doing here
            active = self.get_one_epoch_input(z)
            buy_signals = []
            buy_syms = []
            sell_syms = []
            sell_signals = []
            network.reset()
            for n in range(1, self.hd+1):
                network.activate(active[self.hd-n])
            out = network.activate(active[0])
            for x in range(len(out)):
                if(z > (self.epoch_len+rand_start)-2):
                    sym = self.hs.coin_dict[x]
                    end_prices[sym] = self.hs.currentHists[sym]['close'][self.epoch_len+rand_start]
                if(out[x] > .5):
                    buy_signals.append(out[x])
                    buy_syms.append(self.hs.coin_dict[x])
                if(out[x] < -.5):
                    sell_signals.append(out[x])
                    sell_syms.append(self.hs.coin_dict[x])
            #rng = iter(shuffle(rng))
            sorted_buys = np.argsort(buy_signals)[::-1]
            sorted_sells = np.argsort(sell_signals)
            #print(len(sorted_shit), len(key_list))
            for x in sorted_sells:
                sym = sell_syms[x]
                portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][z])
            for x in sorted_buys:
                sym = buy_syms[x]
                portfolio.target_amount = .1 + (out[x] * .1)
                portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][z])
        result_val = portfolio.get_total_btc_value(end_prices)
        print(g.key, " : ")
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        if result_val[1] == 0:
            ft = .7
        else:
            ft = result_val[0]
        return ft

    def solve(self, network):
        return self.evaluate(network) >= self.highest_returns

    def trial_run(self):
        r_start = 0
        file = open("es_trade_god_cppn_3d.pkl",'rb')
        [cppn] = pickle.load(file)
        network = ESNetwork(self.subStrate, cppn, self.params)
        net = network.create_phenotype_network_nd()
        fitness = self.evaluate(net, network, r_start)
        return fitness

    def eval_fitness(self, genomes, config):
        self.epoch_len = 89
        r_start = randint(0+self.hd, self.hs.hist_full_size - self.epoch_len)
        r_start_2 = self.hs.hist_full_size - self.epoch_len-1
        best_g_fit = 0.0
        champ_counter = self.gen_count % 10 
        #img_count = 0
        for idx, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(self.subStrate, cppn, self.params, self.hd)
            net = network.create_phenotype_network_nd()
            train_ft = self.evaluate(net, network, r_start, g)
            validate_ft = self.evaluate(net, network, r_start_2, g)
            g.fitness = (train_ft+validate_ft)/2
            if(g.fitness > best_g_fit):
                best_g_fit = g.fitness
                with open("./champ_data/latest_greatest"+str(champ_counter)+".pkl", 'wb') as output:
                    pickle.dump(g, output)
            #img_count += 1
        if(champ_counter == 0):
            self.refresh()
            self.compare_champs()
        self.gen_count += 1
        return

    def compare_champs(self):
        self.epoch_len = 233
        r_start = self.hs.hist_full_size - self.epoch_len-1
        champ_current = open("./champ_data/latest_greatest.pkl",'rb')
        g = pickle.load(champ_current)
        champ_current.close()
        cppn = neat.nn.FeedForwardNetwork.create(g, self.config)
        network = ESNetwork(self.subStrate, cppn, self.params, self.hd)
        net = network.create_phenotype_network_nd()
        champ_fit = self.evaluate(net, network, r_start, g)
        for f in os.listdir("./champ_data"):
            if(f != "lastest_greatest.pkl"):
                champ_file = open("./champ_data/"+f,'rb')
                g = pickle.load(champ_file)
                champ_file.close()
                cppn = neat.nn.FeedForwardNetwork.create(g, self.config)
                network = ESNetwork(self.subStrate, cppn, self.params, self.hd)
                net = network.create_phenotype_network_nd()
                g.fitness = self.evaluate(net, network, r_start, g)
                if (g.fitness > champ_fit):
                    with open("./champ_data/latest_greatest.pkl", 'wb') as output:
                        pickle.dump(g, output)
        return

    def validate_fitness(self):
        config = self.config
        genomes = neat.Checkpointer.restore_checkpoint("./pkl_pops/pop-checkpoint-27").population
        self.epoch_len = 233
        r_start = self.hs.hist_full_size - self.epoch_len-1
        best_g_fit = 1.0
        for idx in genomes:
            g = genomes[idx]
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(self.subStrate, cppn, self.params, self.hd)
            net = network.create_phenotype_network_nd()
            g.fitness = self.evaluate(net, network, r_start, g)
            if(g.fitness > best_g_fit):
                best_g_fit = g.fitness
                with open('./champ_data/latest_greatest.pkl', 'wb') as output:
                    pickle.dump(g, output)
        return

# Create the population and run the XOR task by providing the above fitness function.
    def run_pop(self, checkpoint=""):
        if(checkpoint == ""):
            pop = neat.population.Population(self.config)
        else:
            pop = neat.Checkpointer.restore_checkpoint("./pkl_pops/pop-checkpoint-" + checkpoint)
        checkpoints = neat.Checkpointer(generation_interval=2, time_interval_seconds=None, filename_prefix='./pkl_pops/pop-checkpoint-')
        stats = neat.statistics.StatisticsReporter()
        pop.add_reporter(stats)
        pop.add_reporter(checkpoints)
        pop.add_reporter(neat.reporting.StdOutReporter(True))

        winner = pop.run(self.eval_fitness, self.num_gens)
        return winner, stats


# If run as script.

    def run_training(self, checkpoint = ""):
        #print(task.trial_run())
        winner = self.run_pop(checkpoint)[0]
        print('\nBest genome:\n{!s}'.format(winner))
        checkpoint_string = str(self.num_gens-1)
        self.num_gens += self.num_gens
        self.run_training(checkpoint_string)


    def run_validation(self):
        self.validate_fitness()
pt = PurpleTrader(34, 144, 211)
pt.run_training("211")
#run_validation()
