import copy
import math
import random
import numpy as np
from operator import attrgetter

from player import Player


class Evolution:
    def __init__(self):
        self.game_mode = "Neuroevolution"

    def next_population_selection(self, players, num_players, selection_method="q_tournament", tournament_size=3):
        """
        Gets list of previous and current players (μ + λ) and returns num_players number of players based on their
        fitness value.

        :param players: list of players in the previous generation
        :param num_players: number of players that we return
        """    
        if selection_method == "q_tournament":
            selected_players = self.q_tournament(players, num_players, tournament_size)
        elif selection_method == "roulette_wheel":
            selected_players = self.roulette_wheel(players, num_players)
        elif selection_method == "sus":
            selected_players = self.sus(players, num_players)
        else:
            raise ValueError("Invalid selection method")

        selected_fitnesses = [p.fitness for p in selected_players]
        average_fitness = sum(selected_fitnesses) / num_players
        fittest_fitness = selected_fitnesses[0]
        least_fit_fitness = selected_fitnesses[-1]

        print("Statistics for Selected Players:")
        print(f"  Average Fitness: {average_fitness:.2f}")
        print(f"  Fittest Fitness: {fittest_fitness}")
        print(f"  Least Fit Fitness: {least_fit_fitness}")

        return selected_players

    def generate_new_population(self, num_players, prev_players=None):
        """
        Gets survivors and returns a list containing num_players number of children.

        :param num_players: Length of returning list
        :param prev_players: List of survivors
        :return: A list of children
        """
        if prev_players is None:
            new_population = [Player(self.game_mode) for _ in range(num_players)]
        else:
            selected_parents = self.roulette_wheel(prev_players, num_players)
            offspring = self.crossover(selected_parents, num_players)
            new_population = [self.mutation(player, 0.2, 0.4, 0.2) for player in offspring]
        return new_population

    def clone_player(self, player):
        """
        Gets a player as an input and produces a clone of that player.
        """
        new_player = Player(self.game_mode)
        new_player.nn = copy.deepcopy(player.nn)
        new_player.fitness = player.fitness
        return new_player

    def roulette_wheel(self, players, num_players):
        total_fitness = sum(player.fitness for player in players)
        return [next(player for player in players if (total_fitness := total_fitness - player.fitness) <= 0) for _ in range(num_players)]

    def sus(self, players, num_players):
        total = sum(player.fitness for player in players)
        thresholds = np.cumsum([player.fitness for player in players]) / total * num_players
        chosen_players = [next(players[i] for i, val in enumerate(thresholds) if val >= np.random.uniform(0, 1)) for _ in range(num_players)]
        return chosen_players

    def q_tournament(self, players, num_players, q):
        selected_players = []
        for _ in range(num_players):
            tournament_players = random.sample(players, q)  # Select q random players for the tournament
            winner = max(tournament_players, key=attrgetter('fitness'))  # Select the player with the highest fitness
            selected_players.append(winner)
        return selected_players

    def mutation(self, child, Weight_probability, Bias_probability, Mutation_deviation):
        child = self.clone_player(child)
        for layer in range(2):
            for attr in ['weight_', 'bias_']:
                if np.random.uniform(0, 1) <= (Weight_probability if attr == 'weight_' else Bias_probability):
                    setattr(child.nn, f'{attr}{layer + 1}', getattr(child.nn, f'{attr}{layer + 1}') + np.random.normal(0, Mutation_deviation, getattr(child.nn, f'{attr}{layer + 1}').shape))
        return child

    def crossover(self, players, num_players):
        inheritance = []
        for integer in range(num_players // 2):
            cross_over_prob = np.random.uniform(0, 1)
            if cross_over_prob < 0.8:
                for num in range(2):
                    parent_num = players[integer * 2 + num]
                    child_num = self.clone_player(parent_num)
                    num_temp = [(parent_num.nn.layer_sizes[layer] // 2) for layer in range(2)]
                    for j in range(2):
                        for attr in ['weight_', 'bias_']:
                            child_params = getattr(parent_num.nn, f'{attr}{j+1}').copy()
                            setattr(child_num.nn, f'{attr}{j+1}', np.concatenate((child_params[:num_temp[j]], getattr(players[integer * 2 + 1 - num].nn, f'{attr}{j+1}')[num_temp[j]:]), axis=0))
                    inheritance.append(child_num)
            else:
                    inheritance.extend(players[integer * 2:integer * 2 + 2])
        return inheritance
