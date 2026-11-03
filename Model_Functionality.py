from turtle import st

import neat
import pandas as pd
import numpy as np
import visualizer
import graphviz

class BasicNEATmodel():
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
        return np.argmax([o_buy,o_sell]), o_volume * 2.0 - 1

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
                    in_stock_balance *= (1 - volume)
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

def setup_basic_neat_model(config_path,prices):
    neat_model = BasicNEATmodel(model_config_path=config_path,prices=prices)

    winner = neat_model.train()

    return winner

class LiveBestNetworkReporter(neat.reporting.BaseReporter):
    """Renders the best genome after each generation and passes the PNG bytes to a callback."""
    def __init__(self, config, callback, filename="live-net", every=1):
        self.config = config
        self.callback = callback
        self.filename = filename
        self.every = every
        self.generation = 0

    def start_generation(self, generation):
        self.generation = generation

    def post_evaluate(self, config, population, species, best_genome):
        if self.generation % self.every != 0:
            return
        try:
            visualizer.draw_net(self.config, best_genome, view=False, filename=self.filename, fmt="png")
            with open(self.filename + ".png", "rb") as f:
                image = f.read()
        except Exception as e:
            print("Live network rendering failed:", repr(e))
            return
        self.callback(self.generation, best_genome, image)

class TargetExposureNEATModel():
    def __init__(self, model_config_path, prices, balance = 1000.0, max_generations=300, exposure_threshold=0.05, on_generation=None, live_every=1):
        self.config_path = model_config_path
        self.prices = prices
        self.balance = balance
        self.max_steps = max_generations
        self.exposure_threshold = exposure_threshold

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
        if on_generation is not None:
            population.add_reporter(LiveBestNetworkReporter(config, on_generation, every=live_every))

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

    def _interpret_outputs(self, out):
        return out[0]

    def _run_simulation(self, net):
        """
        Runs the agent for a fixed number of steps.
        Returns the accumulated score (fitness).
        """
        fitnesses = np.zeros(len(self.prices))
        for i in range(0, len(self.prices)):
            price_series = self.prices[i]
            units = 0
            cash = self.balance

            current_exposure = 0.0
            current_equity = cash

            for t in range(0, min(1024,len(price_series.index))):
                if t%10 == 0:
                    inputs = self._build_inputs(price_series, t)
                    out = net.activate(inputs)

                    target_exposure = self._interpret_outputs(out)

                price = price_series.loc[t,"Close"]

                if current_equity <= 0:
                    break # bankrupt

                current_exposure = units * price / current_equity

                if abs(target_exposure - current_exposure) >= self.exposure_threshold:
                    # target_exposure measures the proportion of your equity that is at risk (positive indicates a long position, negative indicates a short position)

                    target_units = target_exposure * current_equity / price
                    units_to_trade = target_units - units

                    cash -= units_to_trade * price
                    units += units_to_trade

                current_equity = cash + units * price

            pnl = (current_equity - self.balance) / self.balance
            initial_price = price_series.loc[0,"Close"]
            final_price = price_series.loc[min(1024,len(price_series.index))-1,"Close"]
            buy_and_hold_return = final_price / initial_price - 1
            pnl_relative = pnl - buy_and_hold_return

            fitnesses[i] = pnl + 1.5 * pnl_relative
            
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

        try:
            visualizer.draw_net(self.config, winner, view=False, filename="winner_net", fmt="png")
        except Exception as e:
            print("Winner network rendering failed:", repr(e))

        return winner

def setup_target_exposure_neat_model(config_path,prices,on_generation=None,live_every=1):
    neat_model = TargetExposureNEATModel(model_config_path=config_path,prices=prices,on_generation=on_generation,live_every=live_every)

    winner = neat_model.train()

    return winner

def backtest_target_exposure_genome(genome, config_path, price_df, balance=1000.0, exposure_threshold=0.05):
    """
    Runs a trained genome over a full price dataframe using the same trading logic as training.
    Returns a dataframe with the model's equity, buy-and-hold equity and exposure per step, plus summary stats.
    """
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path,
    )
    net = neat.nn.FeedForwardNetwork.create(genome, config)
    helper = TargetExposureNEATModel.__new__(TargetExposureNEATModel)

    price_df = price_df.reset_index(drop=True)
    units = 0.0
    cash = balance
    current_equity = cash
    target_exposure = 0.0
    first_price = price_df.loc[0, "Close"]

    rows = []
    for t in range(len(price_df.index)):
        if t % 10 == 0:
            target_exposure = helper._interpret_outputs(net.activate(helper._build_inputs(price_df, t)))

        price = price_df.loc[t, "Close"]

        if current_equity <= 0:
            rows.append((price_df.loc[t, "Date"], 0.0, balance * price / first_price, 0.0))
            continue

        current_exposure = units * price / current_equity
        if abs(target_exposure - current_exposure) >= exposure_threshold:
            units_to_trade = target_exposure * current_equity / price - units
            cash -= units_to_trade * price
            units += units_to_trade

        current_equity = cash + units * price
        rows.append((price_df.loc[t, "Date"], current_equity, balance * price / first_price, units * price / current_equity if current_equity > 0 else 0.0))

    result = pd.DataFrame(rows, columns=["Date", "Model", "Buy and Hold", "Exposure"])
    final_model = result["Model"].iloc[-1]
    final_bh = result["Buy and Hold"].iloc[-1]
    summary = {
        "model_return": final_model / balance - 1,
        "buy_and_hold_return": final_bh / balance - 1,
        "final_equity": final_model,
        "max_drawdown": (result["Model"] / result["Model"].cummax() - 1).min(),
    }
    return result, summary