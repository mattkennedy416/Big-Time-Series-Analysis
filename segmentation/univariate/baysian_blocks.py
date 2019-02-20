

from external.astropy.stats.bayesian_blocks import bayesian_blocks as bb



def bayesian_blocks(t, x=None, sigma=None,fitness='events', **kwargs):
    return bb(t, x=x, sigma=sigma, fitness=fitness, **kwargs)















