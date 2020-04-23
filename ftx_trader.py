
### IMPORTS ###
import random
import sys, os
from functools import partial
from itertools import product
# Libs
import numpy as np
from exchange_wrappers.ftx_wrapper import FtxWrapper
from new_hist_service import HistWorker
from crypto_folio import CryptoFolio
from random import randint, shuffle
import pathlib
import statistics
# Local
import neat.nn
import neat
import _pickle as pickle
#from pureples.es_hyperneat.es_hyperneat import ESNetwork
from pytorch_neat.cppn_safe import create_cppn
from pytorch_neat.substrate import Substrate
from pytorch_neat.safe_es_hyperneat import ESNetwork
import torch.nn.functional as F
import time
# Local
class PurpleTrader:

    # ES-HyperNEAT specific parameters.

    params = {"initial_depth": 2,
            "max_depth": 3,
            "variance_threshold": 0.89,
            "band_threshold": 0.055,
            "iteration_level": 3,
            "division_threshold": 0.21,
            "max_weight": 34.0,
            "activation": "tanh",
            "safe_baseline_depth": 3,
            "grad_steps": 3}



    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')

    start_idx = 0
    highest_returns = 0
    leaf_names = []
    current_champs = {}
    executions = 0
    phenotypes = {}

    def __init__(self, hist_depth, num_gens, genome_filename, gen_count = 1):
        self.genome_filename = genome_filename
        self.hd = hist_depth
        if gen_count != 1:
            self.num_gens = num_gens
        else:
            self.num_gens = gen_count + num_gens
        self.gen_count = gen_count
        self.last_ts = ""
        self.port_dict = {}
        self.refresh()

    def refresh(self):
        self.in_shapes = []
        self.out_shapes = [(0.0, -1.0, 0.0)]
        self.hs = HistWorker(FtxWrapper())
        self.hs.get_wrapper_live_frames_all_syms()
        print(self.hs.currentHists.keys())
        hist_lengths = {}
        self.end_idx = len(self.hs.hist_shaped[0])
        self.but_target = .1
        self.num_syms = self.hs.hist_shaped.shape[0]
        self.inputs = self.hs.hist_shaped[0].shape[1]
        self.outputs = len(self.out_shapes)
        sign = 1
        for ix2 in range(1,self.inputs+1):
            sign *= -1
            self.in_shapes.append((0.0+(sign*.01*ix2), 0.0-(sign*.01*ix2), 0.0))
        self.substrate = Substrate(self.in_shapes, self.out_shapes)
        self.set_leaf_names()

    # informing the substrate
    def reset_substrate(self, input_row):
        current_inputs = self.substrate.input_coordinates
        new_input = []
        for ix,t in enumerate(current_inputs):
            t = list(t)
            offset = input_row[ix] * .5
            t[2] = t[2] + .5
            new_input.append(tuple(t))
        #self.in_shapes = new_input
        self.substrate = Substrate(new_input, self.out_shapes)

    def set_portfolio_keys(self, folio):
        for k in self.hs.currentHists.keys():
            folio.ledger[k] = 0

    def set_leaf_names(self):   
        for l in range(len(self.in_shapes[0])):
            self.leaf_names.append('leaf_one_'+str(l))
            self.leaf_names.append('leaf_two_'+str(l))
        print(self.leaf_names)

    def get_one_epoch_input(self,end_idx):
        master_active = []
        for x in range(1, self.hd+1):
            active = []
            #print(self.outputs)
            for y in range(0, self.outputs):
                sym_data = self.hs.hist_shaped[y][end_idx + x]
                #print(len(sym_data))
                active += sym_data.tolist()

            master_active.append(active)
        #print(active)
        return master_active


    def get_single_symbol_epoch_recurrent_with_position_size(self, end_idx, symbol_idx, current_positions, pnl_hist):
        master_active = []
        for x in range(0, self.hd):
            next_index = end_idx-x
            sym_data = self.hs.hist_shaped[symbol_idx][next_index]
            #print(len(sym_data))
            sym_data = sym_data.tolist()
            '''
            sym_data.append(current_positions[0])
            sym_data.append(current_positions[1])
            if next_index in pnl_hist.keys():
                sym_data.append(pnl_hist[next_index])
            else:
                sym_data.append(0.0)
            '''
            master_active.append(sym_data)
        return master_active

    def evaluate_live(self, builder, rand_start, g, champ_num, verbose=False):
        end_prices = {}
        buys = 0
        sells = 0
        pathlib.Path(str(pathlib.Path(__file__).parent.absolute()) + '/trade_hists/ftx_live/champ_' + str(champ_num)).mkdir(exist_ok=True)
        balances = [] 
        sym_bull = "BULL"
        sym_bear = "BEAR"
        for s in self.hs.coin_dict:
            portfolio = None
            sym = s
            x = self.hs.coin_dict[sym]
            if self.executions == 0:
                self.port_dict[s] = CryptoFolio(1000.0, {0: "BULL", 1: "BEAR"}, "USD", target_amt=1.0)
            portfolio = self.port_dict[s]
            last_val = 1000.0
            port_hist = {}
            ft = 0.0
            if self.executions == 0:
                with open("./trade_hists/ftx_live/champ_" + str(champ_num) + "/" + sym + "_hist.txt", "w+") as f:
                    f.write('0,1\n')
            with open("./trade_hists/ftx_live/champ_" + str(champ_num) + "/" + sym + "_hist.txt", "a+") as f:
                z = self.hs.hist_sizes[s] - 1
                pos_size = []
                active = self.get_single_symbol_epoch_recurrent_with_position_size(z, x, pos_size, port_hist)
                if(self.executions % 13 == 0):
                    self.reset_substrate(active[0])
                    builder.substrate = self.substrate
                    self.phenotypes[s] = builder.create_phenotype_network_nd()
                    network = self.phenotypes[s]
                    network.reset()
                else:
                    network = self.phenotypes[s]
                for n in range(1, self.hd):
                    network.activate([active[self.hd-n]])
                out = network.activate([active[0]])
                bull_open = self.hs.wrapper.get_last_price(s + sym_bull)
                bear_open = self.hs.wrapper.get_last_price(s + sym_bear)
                if(out[0] > .25):
                    portfolio.sell_coin(sym_bull, bull_open)
                    did_buy = portfolio.buy_coin(sym_bear, bear_open)
                    print("short ", s)
                elif(out[0] < -.25):
                    portfolio.sell_coin(sym_bear, bear_open)
                    did_buy = portfolio.buy_coin(sym_bull, bull_open)
                    print("long ", s)
                else:
                    portfolio.sell_coin(sym_bull, bull_open)
                    portfolio.sell_coin(sym_bear, bear_open)
                    print("no position ", s)
                end_prices[sym_bull] = bull_open
                end_prices[sym_bear] = bear_open
                bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                f.write(str(self.hs.currentHists[s][sym_bull]['time'][z]) + ",")
                f.write(str(bal_now)+ " \n")
                ft += bal_now - last_val
                last_val = bal_now
                port_hist[x] = ft / last_val
            result_val = portfolio.get_total_btc_value_no_sell(end_prices)
            balances.append(result_val[0])
            print("genome id ", g.key, " : ")
            print(result_val[0])
        return np.asarray(balances, dtype=np.float32).mean()

    def evaluate_champ(self, builder, rand_start, g, champ_num, verbose=False):
        end_prices = {}
        phenotypes = {}
        buys = 0
        sells = 0
        pathlib.Path(str(pathlib.Path(__file__).parent.absolute()) + '/trade_hists/ftx_full/champ_' + str(champ_num)).mkdir(exist_ok=True)
        balances = [] 
        sym_bull = "BULL"
        sym_bear = "BEAR"
        for s in self.hs.coin_dict:
            self.epoch_len = self.hs.hist_sizes[s] - (self.hd+1)
            start_index = self.hs.currentHists[s][sym_bull].index[0]
            sym = s
            x = self.hs.coin_dict[sym]
            portfolio = CryptoFolio(1000.0, {0: "BULL", 1: "BEAR"}, "USD")
            portfolio.target_amount = .25
            last_val = 1000.0
            port_hist = {}
            ft = 0.0
            with open("./trade_hists/ftx_full/champ_" + str(champ_num) + "/" + sym + "_hist.txt", "w") as f:
                f.write('0,1\n')
                for z_minus in range(start_index, self.epoch_len - 1):
                    z = z_minus
                    pos_size = []
                    active = self.get_single_symbol_epoch_recurrent_with_position_size(z, x, pos_size, port_hist)
                    if(z_minus == start_index or (z_minus + 1) % 13 == 0):
                        self.reset_substrate(active[0])
                        builder.substrate = self.substrate
                        phenotypes[sym] = builder.create_phenotype_network_nd()
                        network = phenotypes[sym]
                        network.reset()
                    for n in range(1, self.hd):
                        network.activate([active[self.hd-n]])
                    out = network.activate([active[0]])
                    bull_open = self.hs.wrapper.get
                    bear_open = self.hs.currentHists[s][sym_bear]['open'][z+1]
                    if(out[0] > .25):
                        portfolio.sell_coin(sym_bull, bull_open)
                        did_buy = portfolio.buy_coin(sym_bear, bear_open)
                        #print("bought ", sym
                    elif(out[0] < -.25):
                        portfolio.sell_coin(sym_bear, bear_open)
                        did_buy = portfolio.buy_coin(sym_bull, bull_open)
                    else:
                        portfolio.sell_coin(sym_bull, bull_open)
                        portfolio.sell_coin(sym_bear, bear_open)
                    end_prices[sym_bull] = bull_open
                    end_prices[sym_bear] = bear_open
                    bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                    f.write(str(self.hs.currentHists[s][sym_bull]['time'][z+1]) + ",")
                    f.write(str(bal_now)+ " \n")
                    ft += bal_now - last_val
                    last_val = bal_now
                    port_hist[x] = ft / last_val
            result_val = portfolio.get_total_btc_value(end_prices)
            balances.append(result_val[0])
            print("genome id ", g.key, " : ")
            print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        return np.asarray(balances, dtype=np.float32).mean()

    def evaluate_relu(self, builder, sym_index, rand_starts, g, verbose=False):
        portfolio_start = 1000.0
        fits = []
        phenotypes = {}
        buys = 0
        sells = 0
        sym_bull = "BULL"
        sym_bear = "BEAR"
        for s in self.hs.coin_dict:
            ft = 0.0
            end_prices = {}
            last_val = portfolio_start
            rand_start = rand_starts[s]
            x = self.hs.coin_dict[s]
            portfolio = CryptoFolio(portfolio_start, {0: "BULL", 1: "BEAR"}, "USD")
            portfolio.target_amount = .25
            port_hist = {}
            for z_minus in range(rand_start, rand_start + self.epoch_len):
                z = z_minus
                pos_sizes = (portfolio.ledger[sym_bull], portfolio.ledger[sym_bear])
                active = self.get_single_symbol_epoch_recurrent_with_position_size(z, x, pos_sizes, port_hist)
                if(z_minus == rand_start or (z_minus + 1) % 13 == 0):
                    self.reset_substrate(active[0])
                    builder.substrate = self.substrate
                    phenotypes[s] = builder.create_phenotype_network_nd()
                    network = phenotypes[s]
                    network.reset()
                for n in range(1, self.hd):
                    network.activate([active[self.hd-n]])
                out = network.activate([active[0]])
                #out = F.softmax(out[0], dim=0)
                #max_output = torch.max(out, 0)[1]
                bull_open = self.hs.currentHists[s][sym_bull]['open'][z+1]
                bear_open = self.hs.currentHists[s][sym_bear]['open'][z+1]
                if(out[0] > .25):
                    portfolio.sell_coin(sym_bull, bull_open)
                    did_buy = portfolio.buy_coin(sym_bear, bear_open)
                    #print("bought ", sym
                elif(out[0] < -.25):
                    portfolio.sell_coin(sym_bear, bear_open)
                    did_buy = portfolio.buy_coin(sym_bull, bull_open)
                else:
                    portfolio.sell_coin(sym_bull, bull_open)
                    portfolio.sell_coin(sym_bear, bear_open)
                #rng = iter(shuffle(rng))
                end_prices[sym_bull] = bull_open
                end_prices[sym_bear] = bear_open
                bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                ft += (bal_now - last_val) / last_val
                last_val = bal_now
                port_hist[z] = ft / last_val
            fits.append(ft)
        avg_returns = statistics.mean(fits)
        print("avg pnl", avg_returns)
        if avg_returns == 0.0:
            avg_returns = -.1
        return avg_returns       

    def trial_run(self):
        r_start = 0
        file = open("es_trade_god_cppn_3d.pkl",'rb')
        [cppn] = pickle.load(file)
        network = ESNetwork(self.subStrate, cppn, self.params)
        net = network.create_phenotype_network_nd()
        fitness = self.evaluate(net, network, r_start)
        return fitness

    def execute_back_prop(self, genome_dict, champ_key, config):
        [cppn] = create_cppn(genome_dict[champ_key], config, self.leaf_names, ['cppn_out'])
        net_builder = ESNetwork(Substrate(self.in_shapes, self.out_shapes), cppn, self.params)
        champ_output = net_builder.safe_baseline(False)
        for key in genome_dict:
            if key != champ_key:
                [cppn_2] = create_cppn(genome_dict[key], config, self.leaf_names, ['cppn_out'])
                es_net = ESNetwork(Substrate(self.in_shapes, self.out_shapes), cppn_2, self.params)
                output = es_net.safe_baseline(True)
                es_net.optimizer.zero_grad()
                if output.requires_grad == True:
                    loss_val = (champ_output - output).pow(2).mean()
                    loss_val.backward()
                    es_net.optimizer.step()
                    es_net.map_back_to_genome(genome_dict[key], config, self.leaf_names, ['cppn_out'])
                else:
                    print("error less fit has no grad attached")
        return

    def execute_eval(self, genome):
        if self.executions > 0:
            self.hs.get_wrapper_live_frames_all_syms()
        [cppn] = create_cppn(genome, self.config, self.leaf_names, ["cppn_out"])
        builder = ESNetwork(self.substrate, cppn, self.params)
        current_performance = self.evaluate_live(builder, 0, genome, 3)
        self.executions += 1        

    def eval_fitness(self, genomes, config, grad_step=0):
        self.epoch_len = 55
        r_starts = {}
        for s in self.hs.currentHists:
            r_starts[s] = randint(self.hs.wrapper.start_idxs[s] + self.hd, self.hs.hist_sizes[s] - self.epoch_len)
        champ_counter = self.gen_count % 10
        sym_idx = randint(0,self.num_syms - 1)
        genome_dict = {}
        champ_key = 0
        best_g_fit = -10000
        #img_count = 0
        for idx, g in genomes:
            genome_dict[g.key] = g
            [cppn] = create_cppn(g, config, self.leaf_names, ["cppn_out"])
            net_builder = ESNetwork(self.substrate, cppn, self.params)
            #net = net_builder.create_phenotype_network_nd()
            train_ft = self.evaluate_relu(net_builder, 0, r_starts, g, 0)
            g.fitness = train_ft
            if(g.fitness > best_g_fit):
                best_g_fit = g.fitness
                champ_key = g.key              
                with open("./champ_data/ftx_full/latest_greatest"+str(champ_counter)+".pkl", 'wb') as output:
                    pickle.dump(g, output)
        if grad_step == self.params["grad_steps"]:
            #self.full_backtest(genome_dict[champ_key])
            self.gen_count += 1
            return
        else:
            self.execute_back_prop(genome_dict, champ_key, config)
            grad_step += 1
            self.eval_fitness(genomes, config, grad_step)

    def compare_champs(self):
        r_start = 0
        champ_fit = 0
        for ix, f in enumerate(os.listdir("./champ_data/ftx_full")):
            if f != ".DS_Store":
                champ_file = open("./champ_data/ftx_full/"+f,'rb')
                g = pickle.load(champ_file)
                champ_file.close()
                [cppn] = create_cppn(g, self.config, self.leaf_names, ["cppn_out"])
                net_builder = ESNetwork(self.substrate, cppn, self.params)
                #net = net_builder.create_phenotype_network_nd()
                g.fitness = self.evaluate_champ(net_builder, r_start, g, champ_num = int(f.split(".")[0][-1]))
        return

    def full_backtest(self, genome):
        self.epoch_len = self.hs.hist_full_size - (self.hd+1)
        r_start = self.epoch_len
        [cppn] = create_cppn(genome, self.config, self.leaf_names, ["cppn_out"])
        builder = ESNetwork(self.substrate, cppn, self.params)
        champ_fit = self.evaluate_champ(builder, r_start, genome, 11)
        for ix, f in enumerate(os.listdir("./champ_data/alt_bull_bear")):
            if f != ".DS_Store":
                champ_file = open("./champ_data/alt_bull_bear/"+f,'rb')
                g = pickle.load(champ_file)
                champ_file.close()
                if g.key not in self.current_champs.keys():
                    [cppn] = create_cppn(g, self.config, self.leaf_names, ["cppn_out"])
                    net_builder = ESNetwork(self.substrate, cppn, self.params)
                    g.fitness = self.evaluate_champ(net_builder, r_start, g, champ_num = ix)
                    self.current_champs[g.key] = g.fitness
                if (self.current_champs[g.key] < champ_fit):
                    del self.current_champs[g.key]
                    self.current_champs[genome.key] = champ_fit
                    with open("./champ_data/alt_bull_bear/" + f, 'wb') as output:
                        pickle.dump(genome, output)
                    return
        return

    def run_live(self):
        genome_file = open("./champ_data/archive/"+self.genome_filename,'rb')
        g = pickle.load(genome_file)
        genome_file.close()
        self.execute_eval(g)
        time.sleep(3600)
        self.run_live()

pt = PurpleTrader(34, 255, "latest_greatest3.pkl", 1)
pt.run_live()
#pt.run_training("69")
