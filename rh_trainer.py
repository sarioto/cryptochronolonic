
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product
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
# ES-HyperNEAT specific parameters.
    params = {"initial_depth": 1,
            "max_depth": 3,
            "variance_threshold": 0.3,
            "band_threshold": 0.3,
            "iteration_level": 1,
            "division_threshold": 0.5,
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
        if gen_count == 1:
            self.num_gens = num_gens
        else:
            self.num_gens = gen_count + num_gens
        self.gen_count = gen_count
        self.refresh()

    def refresh(self, reload_data=False):
        print("refreshing")
        self.in_shapes = []
        self.out_shapes = []
        self.hs = HistWorker()
        if(reload_data != False):
            self.hs.pull_robinhood_train_data()
        self.hs.get_robinhood_train()
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.hist_shaped[0])
        self.but_target = 1.0
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
        for x in range(1, self.hd+1):
            active = []
            #print(self.outputs)
            for y in range(0, self.outputs):
                try:
                    sym_data = self.hs.hist_shaped[y][end_idx+x]
                    #print(len(sym_data))
                    active += sym_data.tolist()
                except:
                    print('error')
            master_active.append(active)
        #print(active)
        return master_active

    def get_net_input(self, idx):
        active = []
        for y in range(0, self.outputs):
            try:
                sym_data = self.hs.hist_shaped[y][idx - 1]
                active += sym_data.tolist()
            except:
                print("error loading input data")
        return active

    def evaluate_champ(self, network, es, rand_start, g, verbose=False):
        portfolio_start = 10000.0
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict, "USDT")
        end_prices = {}
        buys = 0
        sells = 0
        with open("./trade_hists/" + str(g.key) + "_hist.txt", "w") as ft:
            ft.write('date,current_balance \n')
            for z_minus in range(0, self.epoch_len - 1):
                #TODO add comments to clarify all the 
                #shit im doing here
                z = rand_start - z_minus
                active = self.get_one_epoch_input(z)
                buy_signals = []
                buy_syms = []
                sell_syms = []
                sell_signals = []
                network.reset()
                for n in range(1, self.hd+1):
                    out = network.activate(active[self.hd-n])
                for x in range(len(out)):
                    sym = self.hs.coin_dict[x]
                    if (out[x] > .5):
                        portfolio.buy_coin(sym, self.hs.currentHists[sym]['open_price'][z])
                    if (out[x] < -.5):
                        portfolio.sell_coin(sym, self.hs.currentHists[sym]['open_price'][z])
                    end_prices[sym] = self.hs.currentHists[sym]['open_price'][z]
                ft.write(self.hs.currentHists[sym]['begins_at'][z] + ",")
                ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
            result_val = portfolio.get_total_btc_value(end_prices)
            print("genome id ", g.key, " : ")
            print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
            if result_val[1] == 0:
                ft = .7
            else:
                ft = result_val[0]
            return ft

    def evaluate(self, network, es, rand_start, g, verbose=False):
        portfolio_start = 10000.0
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict, "USDT")
        end_prices = {}
        buys = 0
        sells = 0
        loss_factor = 0
        ft = 0
        last_val = portfolio_start
        for z_minus in range(0, self.epoch_len - 1):
            #TODO add comments to clarify all the 
            #shit im doing here
            z = rand_start - z_minus
            active = self.get_one_epoch_input(z)
            buy_signals = []
            buy_syms = []
            sell_syms = []
            sell_signals = []
            network.reset()
            for n in range(1, self.hd+1):
                out = network.activate(active[self.hd-n])
            for x in range(len(out)):
                sym = self.hs.coin_dict[x]
                if (out[x] > .5):
                    #print("buying " + sym)
                    portfolio.buy_coin(sym, self.hs.currentHists[sym]['open_price'][z])
                if (out[x] < -.5):
                    #print("selling " + sym)
                    portfolio.sell_coin(sym, self.hs.currentHists[sym]['open_price'][z])
                
                end_prices[sym] = self.hs.currentHists[sym]['open_price'][z]
            bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
            ft += bal_now - last_val 
            last_val = bal_now
        result_val = portfolio.get_total_btc_value(end_prices)
        print("genome id ", g.key, " : ")
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        '''
        if result_val[1] == 0:
            ft = .5
        else:
            ft = result_val[0] - loss_factor
        '''
        return ft

    def trial_run(self):
        r_start = 0
        file = open("es_trade_god_cppn_3d.pkl",'rb')
        [cppn] = pickle.load(file)
        network = ESNetwork(self.subStrate, cppn, self.params)
        net = network.create_phenotype_network_nd()
        fitness = self.evaluate(net, network, r_start)
        return fitness

    def eval_fitness(self, genomes, config):
        r_start = randint(20,self.hs.hist_full_size - self.hd)
        self.epoch_len = randint(5,20)
        best_g_fit = 0.0
        champ_counter = self.gen_count % 10
        #print(champ_counter) 
        #img_count = 0
        for idx, g in genomes:
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(self.subStrate, cppn, self.params, self.hd)
            net = network.create_phenotype_network_nd()
            train_ft = self.evaluate(net, network, r_start, g)
            #validate_ft = self.evaluate(net, network, r_start_2, g)
            #g.fitness = (train_ft+validate_ft)/2
            g.fitness = train_ft
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
        r_start = 40
        self.epoch_len = 40
        print(self.end_idx)
        champ_fit = 0
        for f in os.listdir("./champ_data"):
            if(f != "lastest_greatest.pkl"):
                champ_file = open("./champ_data/"+f,'rb')
                g = pickle.load(champ_file)
                champ_file.close()
                cppn = neat.nn.FeedForwardNetwork.create(g, self.config)
                network = ESNetwork(self.subStrate, cppn, self.params, self.hd)
                net = network.create_phenotype_network_nd()
                g.fitness = self.evaluate_champ(net, network, r_start, g)
                if (g.fitness > champ_fit):
                    with open("./champ_data/latest_greatest.pkl", 'wb') as output:
                        pickle.dump(g, output)
        print(champ_fit)
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
        print(self.num_gens)
        winner = pop.run(self.eval_fitness, self.num_gens)
        return winner, stats


# If run as script.

    def run_training(self, checkpoint = ""):
        #print(task.trial_run())
        if checkpoint == "":
            winner = self.run_pop()[0]
        else:
            winner = self.run_pop(checkpoint)[0]
        print('\nBest genome:\n{!s}'.format(winner))
        checkpoint_string = str(self.num_gens-1)
        self.num_gens += self.num_gens
        self.run_training(checkpoint_string)

    def run_validation(self):
        self.validate_fitness()
        

pt = PurpleTrader(5, 144, 1)
pt.compare_champs()
#pt.run_training()


#run_validation()
