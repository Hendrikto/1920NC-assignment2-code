import numpy as np
from deap import creator
from tqdm import tqdm

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
