
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
import torch
# Local
class PurpleTrader:

    # ES-HyperNEAT specific parameters.

    params = {"initial_depth": 2,
            "max_depth": 3,
            "variance_threshold": 0.8,
            "band_threshold": 0.05,
            "iteration_level": 3,
            "division_threshold": 0.3,
            "max_weight": 34.0,
            "activation": "relu",
            "safe_baseline_depth": 3,
            "grad_steps": 1}



    # Config for CPPN.
    config = neat.config.Config(neat.genome.DefaultGenome, neat.reproduction.DefaultReproduction,
                                neat.species.DefaultSpeciesSet, neat.stagnation.DefaultStagnation,
                                'config_trader')

    start_idx = 0
    highest_returns = 0
    portfolio_list = []
    leaf_names = []
    def __init__(self, hist_depth, num_gens, gen_count = 1):
        self.hd = hist_depth
        if gen_count != 1:
            self.num_gens = num_gens
        else:
            self.num_gens = gen_count + num_gens
        self.gen_count = gen_count
        self.refresh()

    def refresh(self):
        self.in_shapes = []
        self.out_shapes = [(0.0, -1.0, 1.0), (0.0, -1.0, 0.0), (0.0, -1.0, -1.0)]
        self.hs = HistWorker()
        self.hs.get_binance_train()
        print(self.hs.currentHists.keys())
        self.end_idx = len(self.hs.hist_shaped[0])
        self.but_target = .1
        print(self.hs.hist_shaped.shape)
        self.num_syms = self.hs.hist_shaped.shape[0]
        self.inputs = self.hs.hist_shaped[0].shape[1] + 2
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

    def get_single_symbol_epoch_recurrent(self, end_idx, symbol_idx):
        master_active = []
        for x in range(0, self.hd):
            try:
                sym_data = self.hs.hist_shaped[symbol_idx][end_idx-x]
                #print(len(sym_data))
                master_active.append(sym_data.tolist())
            except:
                print('error')
        return master_active

    def get_single_symbol_epoch_recurrent_with_position_size(self, end_idx, symbol_idx, current_position, pnl_hist):
        master_active = []
        for x in range(0, self.hd):
            try:
                next_index = end_idx-x
                sym_data = self.hs.hist_shaped[symbol_idx][next_index]
                #print(len(sym_data))
                sym_data = sym_data.tolist()
                sym_data.append(current_position)
                if next_index in pnl_hist.keys():
                    sym_data.append(pnl_hist[next_index])
                else:
                    sym_data.append(0.0)
                master_active.append(sym_data)
            except:
                print('error')
        return master_active
    def evaluate_champ_one_balance(self, builder, rand_start, g, champ_num, verbose=False):
        portfolio_start = 1000.0
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict, "USD")
        end_prices = {}
        phenotypes = {}
        port_hist = {}
        buys = 0
        sells = 0
        last_val = 1000.0
        with open("./trade_hists/binance/" + str(champ_num) + "_hist.txt", "w") as ft:
            ft.write('date,current_balance \n')
            for z_minus in range(0, self.epoch_len - 1):
                for x in range(self.num_syms):
                    z = rand_start - z_minus
                    sym = self.hs.coin_dict[x]
                    z = rand_start - z_minus
                    pos_size = portfolio.ledger[sym]
                    active = self.get_single_symbol_epoch_recurrent_with_position_size(z, x, pos_size, port_hist)
                    if(z_minus == 0 or (z_minus + 1) % 21 == 0):
                        self.reset_substrate(active[0])
                        builder.substrate = self.substrate
                        phenotypes[sym] = builder.create_phenotype_network_nd()
                        network = phenotypes[sym]
                    network.reset()
                    for n in range(1, self.hd+1):
                        network.activate([active[self.hd-n]])
                    out = network.activate([active[0]])
                    out = F.softmax(out[0], dim=0)
                    #print(out)
                    max_output = torch.max(out, 0)[1]
                    end_prices[sym] = self.hs.currentHists[sym]['open'][z-1]
                    if(max_output == 2):
                        portfolio.sell_coin(sym, self.hs.currentHists[sym]['open'][z-1])
                    if(max_output == 0):
                        portfolio.buy_coin(sym, self.hs.currentHists[sym]['open'][z-1])
                    ft.write(str(self.hs.currentHists[sym]['date'][z]) + ",")
                    ft.write(str(portfolio.get_total_btc_value_no_sell(end_prices)[0])+ " \n")
                    end_prices[sym] = self.hs.currentHists[sym]['open'][z-1]
                bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                ft += bal_now - last_val
                last_val = bal_now
                port_hist[x] = ft
            result_val = portfolio.get_total_btc_value(end_prices)
            print("genome id ", g.key, " : ")
            print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
            ft = result_val[0]
            return ft

    def evaluate_champ(self, builder, rand_start, g, champ_num, verbose=False):
        end_prices = {}
        phenotypes = {}
        buys = 0
        sells = 0
        pathlib.Path(str(pathlib.Path(__file__).parent.absolute()) + '/trade_hists/binance_per_symbol_new/champ_' + str(champ_num)).mkdir(exist_ok=True)
        balances = [] 
        for x in range(self.num_syms):
            sym = self.hs.coin_dict[x]
            portfolio = CryptoFolio(1000.0, self.hs.coin_dict, "USD")
            portfolio.target_amount = .25
            last_val = 1000.0
            port_hist = {}
            ft = 0.0
            with open("./trade_hists/binance_per_symbol_new/champ_" + str(champ_num) + "/" + sym + "_hist.txt", "w") as f:
                f.write('0,1\n')
                for z_minus in range(0, self.epoch_len - 1):
                    z = rand_start - z_minus
                    pos_size = portfolio.ledger[sym]
                    active = self.get_single_symbol_epoch_recurrent_with_position_size(z, x, pos_size, port_hist)
                    if(z_minus == 0 or (z_minus + 1) % 21 == 0):
                        self.reset_substrate(active[0])
                        builder.substrate = self.substrate
                        phenotypes[sym] = builder.create_phenotype_network_nd()
                        network = phenotypes[sym]
                    network.reset()
                    for n in range(1, self.hd):
                        network.activate([active[self.hd-n]])
                    out = network.activate([active[0]])
                    out = F.softmax(out[0], dim=0)
                    #print(out)
                    max_output = torch.max(out, 0)[1]
                    end_prices[sym] = self.hs.currentHists[sym]['close'][z]
                    if(max_output == 2):
                        portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][z])
                    if(max_output == 0):
                        portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][z])
                    balance = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                    f.write(str(self.hs.currentHists[sym]['date'][z]) + ",")
                    f.write(str(balance)+ " \n")
                    end_prices[sym] = self.hs.currentHists[sym]['close'][z]
                    bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                    ft += bal_now - last_val
                    last_val = bal_now
                    port_hist[x] = ft / last_val
            result_val = portfolio.get_total_btc_value(end_prices)
            balances.append(result_val[0])
            print("genome id ", g.key, " : ")
            print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        return np.asarray(balances, dtype=np.float32).mean()

    def evaluate(self, builder, rand_start, g, verbose=False):
        portfolio_start = 1000.0
        end_prices = {}
        phenotypes = {}
        balances = []
        buys = 0
        sells = 0
        last_val = portfolio_start
        ft = 0.0
        for x in range(self.num_syms):
            portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict, "USD")
            portfolio.target_amount = .25
            sym = self.hs.coin_dict[x]
            for z_minus in range(0, self.epoch_len):
                z = rand_start - z_minus
                pos_size = portfolio.ledger[sym]
                active = self.get_single_symbol_epoch_recurrent_with_position_size(z, x, pos_size)
                #print(active)
                if(z_minus == 0 or (z_minus + 1) % 8 == 0):
                    self.reset_substrate(active[0])
                    builder.substrate = self.substrate
                    phenotypes[sym] = builder.create_phenotype_network_nd()
                    network = phenotypes[sym]
                network.reset()
                for n in range(1, self.hd+1):
                    network.activate([active[self.hd-n]])
                out = network.activate([active[0]])
                if(out[0] < -0.5):
                    portfolio.sell_coin(sym, self.hs.currentHists[sym]['close'][z])
                    #print("bought ", sym)
                elif(out[0] > 0.5):
                    did_buy = portfolio.buy_coin(sym, self.hs.currentHists[sym]['close'][z])
                #rng = iter(shuffle(rng))
                end_prices[sym] = self.hs.currentHists[sym]['close'][z]
                bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
                if bal_now == last_val:
                    ft += -.01
                else: 
                    ft += bal_now - last_val
                last_val = bal_now
        print(g.key, " : ")
        result_val = portfolio.get_total_btc_value(end_prices)
        print(result_val[0], "buys: ", result_val[1], "sells: ", result_val[2])
        return ft

    def evaluate_relu(self, builder, rand_start, g, sym_index, verbose=False):
        portfolio_start = 1000.0
        end_prices = {}
        phenotypes = {}
        balances = []
        buys = 0
        sells = 0
        last_val = portfolio_start
        ft = 0.0
        x = sym_index
        portfolio = CryptoFolio(portfolio_start, self.hs.coin_dict, "USD")
        portfolio.target_amount = .25
        sym = self.hs.coin_dict[x]
        port_hist = {}
        for z_minus in range(0, self.epoch_len):
            z = rand_start - z_minus
            pos_size = portfolio.ledger[sym]
            active = self.get_single_symbol_epoch_recurrent_with_position_size(z, x, pos_size, port_hist)
            if(z_minus == 0 or (z_minus + 1) % 21 == 0):
                self.reset_substrate(active[0])
                builder.substrate = self.substrate
                phenotypes[sym] = builder.create_phenotype_network_nd()
                network = phenotypes[sym]
            network.reset()
            for n in range(1, self.hd):
                network.activate([active[self.hd-n]])
            out = network.activate([active[0]])
            out = F.softmax(out[0], dim=0)
            max_output = torch.max(out, 0)[1]
            if(max_output == 2):
                portfolio.sell_coin(sym, self.hs.currentHists[sym]['open'][z-1])
                #print("bought ", sym)
            elif(max_output == 0):
                did_buy = portfolio.buy_coin(sym, self.hs.currentHists[sym]['open'][z-1])
            #rng = iter(shuffle(rng))
            end_prices[sym] = self.hs.currentHists[sym]['open'][z-1]
            bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
            ft += bal_now - last_val
            last_val = bal_now
            port_hist[z] = ft / last_val
        bal_now = portfolio.get_total_btc_value_no_sell(end_prices)[0]
        balances.append(bal_now)
        print("sym ", sym, " end balance: ", bal_now)
        return ft       

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

    def eval_fitness(self, genomes, config, grad_step=0):
        self.epoch_len = 89
        r_start = randint(0+self.epoch_len, self.hs.hist_full_size - self.hd)
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
            train_ft = self.evaluate_relu(net_builder, r_start, g, sym_idx)
            g.fitness = train_ft
            if(g.fitness > best_g_fit):
                best_g_fit = g.fitness
                champ_key = g.key
                with open("./champ_data/binance_per_symbol/latest_greatest"+str(champ_counter)+".pkl", 'wb') as output:
                    pickle.dump(g, output)
        if grad_step== self.params["grad_steps"]:
            return
        else:
            self.execute_back_prop(genome_dict, champ_key, config)
            grad_step += 1
            self.eval_fitness(genomes, config, grad_step)
        self.gen_count += 1
        return

    def compare_champs(self):
        self.epoch_len = self.hs.hist_full_size - (self.hd+1)
        r_start = self.epoch_len
        champ_fit = 0
        for ix, f in enumerate(os.listdir("./champ_data/binance_per_symbol")):
            if f != ".DS_Store":
                champ_file = open("./champ_data/binance_per_symbol/"+f,'rb')
                g = pickle.load(champ_file)
                champ_file.close()
                [cppn] = create_cppn(g, self.config, self.leaf_names, ["cppn_out"])
                net_builder = ESNetwork(self.substrate, cppn, self.params)
                g.fitness = self.evaluate_champ(net_builder, r_start, g, champ_num = ix)
                '''
                if (g.fitness > champ_fit):
                    with open("./champ_data/binance_per_symbol/latest_greatest.pkl", 'wb') as output:
                        pickle.dump(g, output)
                '''
        return

    def validate_fitness(self):
        config = self.config
        genomes = neat.Checkpointer.restore_checkpoint("./pkl_pops/pop-checkpoint-27").population
        self.epoch_len = 233
        r_start = self.hs.hist_full_size - self.epoch_len-1
        best_g_fit = -100.0
        best_fit_idx = 0
        for idx in genomes:
            g = genomes[idx]
            cppn = neat.nn.FeedForwardNetwork.create(g, config)
            network = ESNetwork(self.subStrate, cppn, self.params, self.hd)
            net = network.create_phenotype_network_nd()
            g.fitness = self.evaluate(net, network, r_start, g)
            if(g.fitness > best_g_fit):
                best_g_fit = g.fitness
                best_fit_idx = g.key
                with open('./champ_data/binance_per_symbol/latest_greatest.pkl', 'wb') as output:
                    pickle.dump(g, output)
        
        return

# Create the population and run the XOR task by providing the above fitness function.
    def run_pop(self, checkpoint=""):
        pop = neat.population.Population(self.config)
        stats = neat.statistics.StatisticsReporter()
        pop.add_reporter(stats)
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

pt = PurpleTrader(21, 255, 1)
#pt.run_training("")
pt.compare_champs()
#run_validation()