import itertools
from multiprocessing import Pool
from collections import defaultdict

import numpy as np
from sklearn import random_projection
from sklearn.metrics import pairwise_distances_argmin
from sklearn.cluster import KMeans


class EMMC:

    def cluster(self, data, diameter, k, epsilon, delta):

        self.diameter = diameter

        epsilon_JL = 0.99
        epsilon_L = epsilon * 0.2
        epsilon_G = epsilon * 0.6
        epsilon_E = epsilon * 0.2
        delta_G = delta * 0.5
        delta_E = delta * 0.5

        # Step 1
        print("[emmc] Starting step 1...")
        D_ = self.step_1(D=data, epsilon_JL=epsilon_JL)

        # Step 2
        print("[emmc] Starting step 2...")
        C = self.step_2(
            D_=D_, epsilon_JL=epsilon_JL, epsilon_E=epsilon_E, delta_E=delta_E, k=k
        )

        # step 3
        print("[emmc] Starting step 3...")
        D__ = self.step_3(C=C, D_=D_, epsilon_L=epsilon_L)

        # step 4
        print("[emmc] Starting step 4...")
        F = self.step_4(
            D__=D__,
            D_=D_,
            D=data,
            k=k,
            epsilon_G=epsilon_G,
            delta_G=delta_G,
        )

        F = np.asarray([cl for cl in F if len(cl) > 0])

        return F

    def step_1(self, D, epsilon_JL):
        n, d = D.shape  # no privacy violation, because only used for JLT
        d_ = int(np.log10(n) / epsilon_JL**2)

        D_ = D

        print("[emmc] \tProjecting...")
        if d_ < d:
            T = random_projection.GaussianRandomProjection(n_components=d_)
            D_ = T.fit_transform(X=D)

        print("[emmc] \tScaling...")
        D_ = self.scale_D_to_unit_ball(D=D_)

        return D_

    def scale_D_to_unit_ball(self, D: np.ndarray) -> np.ndarray:
        max_vector_norm_in_data = np.max(np.sqrt(np.square(D).sum(axis=1)))
        return np.divide(D, max_vector_norm_in_data)

    def step_2(self, D_, epsilon_JL, epsilon_E, delta_E, k):
        n, d_ = D_.shape

        r_i = 1 / n
        t_i = (epsilon_JL * r_i) / np.sqrt(d_)
        m = np.ceil(np.emath.logn(1 + epsilon_JL, 2 * n)).astype(int)
        k_ = np.ceil(k * np.log(np.ceil(1 / epsilon_JL))).astype(int)
        if k_ <= 0:
            k_ = 1

        distance_budget = d_ * ((1 / epsilon_JL) + 1) ** 2

        print("[emmc] \tGenerating offset...")
        V_groups = SophisticatedOffsetSetGenerator.generate_offset_groups(
            dimensions=d_, distance_budget=distance_budget
        )

        C = None

        print("[emmc] \tIterate Grids...", end="\t")
        for _ in range(1, m + 1):
            C_i, D_ = self.dp_grid_max_cover(
                D_=D_,
                t_i=t_i,
                r_i=r_i,
                k_=k_,
                epsilon_E=epsilon_E,
                delta_E=delta_E,
                V_groups=V_groups,
            )
            C = C_i if C is None else np.append(C, C_i, axis=0)
            C = np.unique(C, axis=0)

            if len(D_) == 0:
                break
            r_i = (1 + epsilon_JL) * r_i
            t_i = (1 + epsilon_JL) * t_i
        return C

    def dp_grid_max_cover(self, D_, t_i, r_i, k_, epsilon_E, delta_E, V_groups):
        C_i = []
        C_i_already_seen = set()
        d_ = D_.shape[1]

        epsilon_E_ = epsilon_E / (2 * np.log(np.e / delta_E))

        G_i_size = np.floor(1 / t_i) ** d_

        cover = SophisticatedCoverSetsGenerator.get_cover_sets(
            data=D_, t_i=t_i, r_i=r_i, V_groups=V_groups, dimensions=d_
        )

        cover_altered = True
        total_cover = 0
        covered_demand_points_this_iteration = []

        for _ in range(k_):
            if cover_altered:
                total_cover = 0
                for value in cover.values():
                    total_cover += np.exp(epsilon_E_ * len(value))

                total_cover += G_i_size - len(cover)

            bias = 1 - (G_i_size / total_cover)
            if np.random.uniform() < bias:
                elements = list(cover.keys())
                weights = np.array(
                    [epsilon_E_ * len(value) for value in cover.values()]
                )
                weights = np.exp(weights)
                weights = np.subtract(weights, 1)

                weights = np.divide(weights, weights.sum())
                indices = np.arange(len(elements))
                ii = np.random.choice(a=indices, p=weights)
                g = elements[ii]
            else:
                g = self.sample_random_grid_point(t_i, d_)
            g = tuple(g)

            if g not in C_i_already_seen:
                grid_point = np.array(g)
                C_i.append(list(np.multiply(grid_point, t_i)))
                C_i_already_seen.add(g)

                if (cover.get(g, False)) is not False:
                    covered_demand_points = list(cover[g])
                    covered_demand_points_this_iteration.extend(covered_demand_points)
                    for key in cover.keys():
                        cover[key].difference_update(covered_demand_points)
                    del cover[g]
                    cover_altered = True
                else:
                    cover_altered = False
            else:
                cover_altered = False

            if len(cover) == 0:
                break

        if len(covered_demand_points_this_iteration) != 0:
            D_ = np.delete(D_, covered_demand_points_this_iteration, axis=0)
        return C_i, D_

    def sample_random_grid_point(self, t_i, dimensions):
        max_grid_index = np.floor(1 / t_i)

        random_vector_length_in_ball = np.random.uniform(low=0.0, high=1)
        vector = np.random.uniform(size=dimensions)
        vector_norm = np.linalg.norm(vector)
        if vector_norm == 0.0:
            return vector
        vector = (
            np.multiply(np.divide(vector, vector_norm), random_vector_length_in_ball)
            / t_i
        )

        for i in range(len(vector)):
            if np.random.uniform() < 0.5:
                vector[i] = np.ceil(vector[i])
            else:
                vector[i] = np.floor(vector[i])

            if vector[i] > max_grid_index:
                vector[i] = max_grid_index
            elif vector[i] < -max_grid_index:
                vector[i] = -max_grid_index

        return list(vector)

    def step_3(self, D_, C, epsilon_L):
        if C is None:
            raise ValueError("C has no candidates.")

        array_of_indexes_closest_to_points = pairwise_distances_argmin(D_, C)
        N = {
            c: np.maximum(
                np.ceil(
                    np.count_nonzero(array_of_indexes_closest_to_points == c)
                    + np.random.laplace(0.0, (1 / epsilon_L))
                ).astype(int),
                0,
            )
            for c in range(len(C))
        }

        D__ = []
        for c in range(len(C)):
            count = N[c]
            if count > 0:
                point_at_index_c_taken_count_times = [list(C[c])] * count
                D__.extend(point_at_index_c_taken_count_times)

        return np.array(D__)

    def step_4(self, D__, D_, D, k, epsilon_G, delta_G):

        S__ = KMeans(n_clusters=k, init="k-means++").fit(D__).cluster_centers_

        labels = pairwise_distances_argmin(D_, S__)
        F = []

        for i in range(len(S__)):
            D_i = D[np.where(labels == i)]
            mu_i = self.noisy_average(D_i=D_i, epsilon_G=epsilon_G, delta_G=delta_G)
            if mu_i is not None:
                F.append(mu_i)

        return F

    def noisy_average(self, D_i, epsilon_G, delta_G):
        d = D_i.shape[1]
        m = len(D_i)
        if m < 2:
            return []

        diameter_D_i = self.diameter
        m_roof = (
            m
            + np.random.laplace(scale=(5 / epsilon_G))
            - ((5 / epsilon_G) * np.log(2 / delta_G))
        )
        if m_roof <= 0:
            return []

        sigma = ((5 * diameter_D_i) / (4 * epsilon_G * m_roof)) * np.sqrt(
            2 * np.log(3.5 / delta_G)
        )
        noise = np.random.normal(scale=sigma**2, size=(1, d))
        average = D_i.mean(axis=0)
        return np.add(noise, average)[0, :].tolist()


class OffsetSetGenerator(object):
    def __init__(self, dimensions, current_dimension):
        self.dimensions = dimensions
        self.current_dimension = current_dimension

    def __call__(self, distance_budget_possible_solution_tuple) -> np.ndarray:
        distance_budget, possible_solution = distance_budget_possible_solution_tuple

        return self._generate_offset_set_recursion(
            dimensions=self.dimensions,
            current_dimension=self.current_dimension,
            distance_budget=distance_budget,
            possible_solution=possible_solution,
        )

    @classmethod
    def _generate_offset_set_recursion(
        cls, possible_solution, dimensions, current_dimension, distance_budget
    ) -> np.ndarray:
        if current_dimension == dimensions - 1:
            i = 0
            solutions = []
            while i**2 < distance_budget:
                possible_solution[current_dimension] = i
                solutions.append(possible_solution.copy())
                i += 1

            return np.array(solutions)

        else:
            i = 0
            solution = np.empty(shape=(0, dimensions))
            while i**2 < distance_budget:
                possible_solution[current_dimension] = i
                new_distance_budget = distance_budget - (i**2)
                if new_distance_budget < 0:
                    break
                solution_rec = cls._generate_offset_set_recursion(
                    possible_solution=possible_solution,
                    dimensions=dimensions,
                    current_dimension=current_dimension + 1,
                    distance_budget=new_distance_budget,
                )
                solution = np.append(solution, solution_rec, axis=0)
                i += 1
            return solution

    @classmethod
    def generate_offset_set_multithread(cls, dimensions, distance_budget) -> np.ndarray:
        if dimensions == 1:
            i = 0
            solutions = []
            while i**2 < distance_budget:
                solutions.append([i])
                i += 1

            return np.array(solutions)
        else:
            next_integers = []
            i = 0
            while i**2 < distance_budget:
                next_integers.append(i)
                i += 1

            distance_budget_possible_solution_tuples = [
                (distance_budget - (i**2), list(itertools.repeat(i, dimensions)))
                for i in next_integers
            ]

            pool = Pool(5)
            callable_offset_generator = cls(dimensions=dimensions, current_dimension=1)

            res = pool.map(
                callable_offset_generator, distance_budget_possible_solution_tuples
            )
            pool.close()
            pool.join()

            return np.concatenate(res)

    @classmethod
    def _generate_offset_set_counter_for_testing_multithread(
        cls, dimensions, epsilon
    ) -> int:
        distance_budget = dimensions * (((1 / epsilon) + 1) ** 2)

        if dimensions == 1:
            i = 0
            solutions = []
            while i**2 < distance_budget:
                solutions.append([i])
                i += 1

            return len(solutions)
        else:
            next_integers = []
            i = 0
            while i**2 < distance_budget:
                next_integers.append(i)
                i += 1

            distance_budget_possible_solution_tuples = [
                (distance_budget - (i**2), list(itertools.repeat(i, dimensions)))
                for i in next_integers
            ]

            pool = Pool(5)
            callable_offset_generator = cls(dimensions=dimensions, current_dimension=1)

            res = pool.map(
                callable_offset_generator, distance_budget_possible_solution_tuples
            )
            pool.close()
            pool.join()

            return len(np.concatenate(res))


class SophisticatedOffsetSetGenerator(object):
    @classmethod
    def generate_offset_groups(cls, dimensions, distance_budget):
        offset_set = OffsetSetGenerator.generate_offset_set_multithread(
            dimensions=dimensions, distance_budget=distance_budget
        )

        offset_set_sorted_index_list = np.argsort(np.linalg.norm(offset_set, axis=1))
        offset_set_sorted = offset_set[offset_set_sorted_index_list]

        groups = []
        orthants = cls.generate_possible_orthants(dimensions=dimensions)

        for orthant in orthants:
            offsets_in_orthant = np.multiply(offset_set_sorted, orthant)
            groups.append((np.array(orthant), offsets_in_orthant))

        return groups

    @classmethod
    def generate_possible_orthants(cls, dimensions):
        combinations = list(
            itertools.combinations_with_replacement([1, -1], dimensions)
        )

        orthants = []
        for i in combinations:
            permutations = list(itertools.permutations(list(i)))
            permutation_array = np.array(permutations)
            unique_permutations = np.unique(permutation_array, axis=0)
            orthants.extend(unique_permutations.tolist())

        return orthants


class SophisticatedCoverSetsGenerator(object):

    @classmethod
    def get_cover_sets(cls, data, V_groups: dict, t_i, r_i, dimensions):
        data_grid_cells = np.divide(data, t_i)
        data_grid_cells = data_grid_cells
        data_grid_cells_copy = np.copy(data_grid_cells)

        num_points = len(data_grid_cells_copy)
        proportion_to_sample = 1.0
        sample_size = int(num_points * proportion_to_sample)
        sampling_indices = np.random.choice(num_points, size=sample_size, replace=False)
        data_grid_cells_copy = data_grid_cells_copy[sampling_indices]
        data_grid_cells = data_grid_cells[sampling_indices]

        cover = defaultdict(set)

        distance_budget = r_i + t_i * np.sqrt(dimensions)

        for orthant, offset_group in V_groups:

            floor_cond = np.where(orthant == -1)
            ceil_cond = np.where(orthant == 1)

            data_grid_cells_copy[:, floor_cond] = np.floor(
                data_grid_cells[:, floor_cond]
            )
            data_grid_cells_copy[:, ceil_cond] = np.ceil(data_grid_cells[:, ceil_cond])
            data_grid_cells_copy = data_grid_cells_copy.astype(int)

            all_possible_grid_cells = (
                data_grid_cells_copy[:, np.newaxis, :] + offset_group[np.newaxis, :, :]
            ).astype(int)

            scaled_grid_cells_with_t_i = all_possible_grid_cells * t_i

            ti_dist_check_ = np.sum(scaled_grid_cells_with_t_i**2, axis=2) <= 1

            scaled_grid_cells_with_t_i = scaled_grid_cells_with_t_i.transpose(1, 0, 2)

            dists_grid_check_ = (
                np.sum(
                    (scaled_grid_cells_with_t_i - data[np.newaxis, :, :]) ** 2,
                    axis=2,
                )
                < distance_budget**2
            )

            dists_grid_check_ = dists_grid_check_.transpose((1, 0))

            relevant_grid_points_index = np.multiply(
                ti_dist_check_, dists_grid_check_
            )  #  logical and

            masked_active_grid_points = np.any(relevant_grid_points_index, axis=1)
            indices_active_grid_points = np.where(masked_active_grid_points)[0]

            for ii, active_grid_points in zip(
                indices_active_grid_points,
                all_possible_grid_cells[masked_active_grid_points],
            ):
                for grid_point in active_grid_points:
                    cover[tuple(grid_point)].add(ii)

        return cover
