import matplotlib.pyplot as plt
import numpy as np
import sklearn.cluster
import sklearn.datasets
from deap import (
    base,
    creator,
    tools,
)
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from tqdm import tqdm

import problem1

creator.create('Particle', np.ndarray, fitness=float('inf'))


def assign_clusters(centroids, X):
    clusters = tuple([] for _ in centroids)
    for i, d in enumerate(X):
        cluster_distances = np.linalg.norm(centroids - d, axis=1)
        clusters[cluster_distances.argmin()].append(i)
    return clusters


def evaluate(centroids, X, clusters):
    return sum(
        np.linalg.norm(X[cluster] - centroid, axis=1).mean()
        for centroid, cluster in zip(centroids, clusters)
        if cluster
    ) / len(clusters)


def update_particle(particle, global_best, w=0.7298, c_1=1.49618, c_2=1.49618):
    dist_personal_best = particle.best - particle
    dist_global_best = global_best - particle
    particle.velocity = (
        w * particle.velocity
        + c_1 * np.random.rand() * dist_personal_best
        + c_2 * np.random.rand() * dist_global_best
    )
    particle += particle.velocity


def run_pso(population, X, iterations=100):
    global_best = creator.Particle(population[0])
    for t in tqdm(range(iterations)):
        for particle in population:
            clusters = assign_clusters(particle, X)
            fitness = evaluate(particle, X, clusters)
            if fitness < particle.fitness:
                particle.best[:] = particle
                particle.fitness = fitness
            if fitness < global_best.fitness:
                global_best[:] = particle
                global_best.fitness = fitness
            update_particle(particle, global_best)
    return global_best


def generate_particle(toolbox):
    particle = creator.Particle(toolbox.random_vectors())
    particle.velocity = toolbox.random_vectors()
    particle.best = particle.copy()
    return particle


def plot_clustering_2d(
    X, y,
    centroids,
    *,
    feature_names,
    title,
    save_path=None,
):
    colors = ('red', 'blue', 'yellow')

    clusters = assign_clusters(centroids, X)
    classes = np.unique(y)

    plt.figure()
    plt.title(title)
    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])
    for cls in classes:
        indexes = np.arange(len(X))[y == cls]
        for cluster_id, cluster in enumerate(clusters):
            instances = X[indexes[np.isin(indexes, cluster)]]
            plt.scatter(
                *instances.T,
                alpha=0.5,
                color=colors[cluster_id],
                label=f'class {cls}, cluster {cluster_id}',
                marker='o^'[cls],
            )
    for i, centroid in enumerate(centroids):
        plt.scatter(
            *centroid,
            color=colors[i],
            label=f'centroid {i}',
            marker='x',
        )
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def exercise3_problem1():
    X, y = problem1.generate_instances()
    feature_names = ('$z_1$', '$z_2$')

    N_d = X.shape[1]
    N_c = 2

    toolbox = base.Toolbox()
    toolbox.register('random_vectors', np.random.uniform, -1, 1, size=(N_c, N_d))
    toolbox.register('particle', generate_particle, toolbox=toolbox)
    toolbox.register('population', tools.initRepeat, tuple, toolbox.particle)

    population = toolbox.population(n=100)
    solution = run_pso(population, X, 50)

    plot_clustering_2d(
        X, y,
        solution,
        feature_names=feature_names,
        title=f'PSO: Artificial Problem 1 (fitness = {solution.fitness:.4f})',
    )

    k_means = sklearn.cluster.KMeans(N_c)
    k_means.fit(X)
    k_means_fitness = evaluate(
        k_means.cluster_centers_,
        X,
        assign_clusters(k_means.cluster_centers_, X),
    )

    plot_clustering_2d(
        X, y,
        k_means.cluster_centers_,
        feature_names=feature_names,
        title=f'k-Means: Artificial Problem 1 (fitness = {k_means_fitness:.4f})',
    )


def plot_clustering_4d(
    X, y,
    centroids,
    *,
    feature_names,
    title,
    save_path=None,
):
    classes = np.unique(y)

    plt.figure()
    axes = plt.gca(projection='3d')
    axes.set_title(title)
    axes.set_xlabel(feature_names[0])
    axes.set_ylabel(feature_names[1])
    axes.set_zlabel(feature_names[2])
    for cls in classes:
        instances = X[y == cls]
        axes.scatter(
            *instances.T[:3],
            c=instances[:, 3],
            cmap=plt.cm.viridis,
            label=f'class {cls}',
            marker='os^'[cls],
        )
    axes.scatter(
        *centroids.T[:3],
        c=centroids[:, 3],
        cmap=plt.cm.viridis,
        label='centroids',
        marker='x',
    )
    plt.legend()

    if save_path:
        plt.savefig(save_path)

    plt.show()
    plt.close()


def exercise3_iris():
    iris_dataset = sklearn.datasets.load_iris()
    X = iris_dataset['data']
    y = iris_dataset['target']
    feature_names = iris_dataset['feature_names']

    N_d = X.shape[1]
    N_c = len(iris_dataset['target_names'])


exercise3_problem1()
