from turtle import st

import neat
import pandas as pd
import numpy as np
import visualizer
import graphviz

class NEATmodel():
    def __init__(self, model_config_path, prices, balance = 1000.0, max_generations=300):
        self.config_path = model_config_path
        self.prices = prices
        self.balance = balance
        self.max_steps = max_generations

        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            self.config_path,
        )
        self.config = config

        population = neat.Population(config)

        population.add_reporter(neat.StdOutReporter(True))
        stats = neat.StatisticsReporter()
        population.add_reporter(stats)

        self.population = population

    def _build_inputs(self, df, t):
        inputs =  {
            "Normalized_Open_Price": df.loc[t,"Normalized_Open_Price"],
            "Normalized_Close_Price": df.loc[t,"Normalized_Close_Price"],
            "ADX": df.loc[t,"ADX"],
            "CCI": df.loc[t,"CCI"],
            "VI+": df.loc[t,"VI+"],
            "VI-": df.loc[t,"VI-"],
            "RSI": df.loc[t,"RSI"],
            "Percent_R": df.loc[t,"Percent_R"],
            "FI": df.loc[t,"FI"],
            "OBV": df.loc[t,"OBV"],
            "VPT": df.loc[t,"VPT"],
            "ADI": df.loc[t,"ADI"]
        }
        return list(inputs.values())

    def _interpret_outputs(self, o_buy, o_sell, o_volume):
        return np.argmax([o_buy,o_sell]), o_volume

    def _run_simulation(self, net):
        """
        Runs the agent for a fixed number of steps.
        Returns the accumulated score (fitness).
        """
        fitnesses = np.zeros(len(self.prices))
        for i in range(0, len(self.prices)):
            price_series = self.prices[i]
    #        drift_series = train_drifts[i]
    #        volatility_series = train_volatilities[i]
            shares = 0
            last_buy = 0
            last_sell = 0
            in_stock_balance = self.balance
            portfolio_value = in_stock_balance
            event_inactive = True
            for t in range(0, min(1024,len(price_series.index))):
                if t%10 == 0:
                    inputs = self._build_inputs(price_series, t)
                    o_buy, o_sell, o_volume = net.activate(inputs)

                    action, volume = self._interpret_outputs(o_buy, o_sell, o_volume)

                price = price_series.loc[t,"Close"]

                if action == 0:
                    shares += in_stock_balance * volume / price
                    in_stock_balance *= (1-volume)
                    last_buy = t

                elif action == 1:
                    shares *= (1-volume)
                    in_stock_balance += price * shares * volume
                    last_sell = t

                # reward = portfolio value
                portfolio_value = in_stock_balance + shares * price

            fitnesses[i] = portfolio_value
        return fitnesses, np.mean(fitnesses)

    def _eval_genome(self, genome, config):
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        fitnesses, fitness = self._run_simulation(net)
        #print(fitness)
        return fitnesses, fitness


    def _eval_genomes(self, genomes, config):
        genome_fitnesses = []
        for genome_id, genome in genomes:
            fitnesses, fitness = self._eval_genome(genome, config)
            genome.fitness = fitness
            genome_fitnesses.append(fitnesses)
            
        print(f"Best fitness: {genome_fitnesses[np.argmax([genome.fitness for genome_id, genome in genomes])]}")

    def _create_graph(self, winner):
        import os, sys
        print("Python executable:", sys.executable)
        try:
            import graphviz as gv
            print("graphviz module version:", gv.__version__)
            print("graphviz module path:", gv.__file__)
        except Exception as e:
            print(f"graphviz import failed inside Streamlit process: {e}")

        try:
            dot = visualizer.draw_net(self.config, winner, view=False, filename="winner-net", fmt="png")
            print("draw_net returned:", dot)
        except Exception as e:
            print("draw_net raised an exception:", repr(e))

        print("winner-net.png exists:", os.path.exists("winner-net.png"))

        visualizer.plot_stats(self.population.reporters.reporters[1], ylog=False, view=False)
        visualizer.plot_species(self.population.reporters.reporters[1], view=False)

    def train(self):

        winner = self.population.run(self._eval_genomes, n=self.max_steps)

        return winner

def setup_neat_model(config_path,prices):
    neat_model = NEATmodel(model_config_path=config_path,prices=prices)

    winner = neat_model.train()

    return winner
    