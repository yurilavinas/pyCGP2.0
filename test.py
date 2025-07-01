import pytest
import numpy as np
import random

from pycgp_finalclass.Config import CGPConfig
from pycgp_finalclass.Genome import CGPGenome
from pycgp_finalclass.Mutation import Golden_mutation
from pycgp_finalclass.Evaluator import Regressor, MultiClassClassifier
from pycgp_finalclass.ES import ES
from pycgp_finalclass.Function import Func
from pycgp_finalclass.Function_library import *
import matplotlib
matplotlib.use('Agg')

def build_funcLib(): #Define the function used
    return [Func(f_sum, 'sum', 2, 0),
            Func(f_aminus, 'aminus', 2, 0),
            Func(f_mult, 'mult', 2, 0),
            Func(f_exp, 'exp', 2, 0),
            Func(f_abs, 'abs', 1, 0),
            Func(f_sqrt, 'sqrt', 1, 0),
            Func(f_sqrtxy, 'sqrtxy', 2, 0),
            Func(f_squared, 'squared', 1, 0),
            Func(f_pow, 'pow', 2, 0),
            Func(f_one, 'one', 0, 0),
            Func(f_zero, 'zero', 0, 0),
            Func(f_const, 'const', 0, 1),
            Func(f_inv, 'inv', 1, 0),
            Func(f_gt, 'gt', 2, 0),
            Func(f_asin, 'asin', 1, 0),
            Func(f_acos, 'acos', 1, 0),
            Func(f_atan, 'atan', 1, 0),
            Func(f_sin, 'sin', 1, 0),
            Func(f_min, 'min', 2, 0),
            Func(f_max, 'max', 2, 0),
            Func(f_round, 'round', 1, 0),
            Func(f_floor, 'floor', 1, 0),
            Func(f_ceil, 'ceil', 1, 0)
            ]
functions = build_funcLib()

# Fix seeds for reproducibility of tests
@pytest.fixture(autouse=True)
def set_seed():
    random.seed(42)
    np.random.seed(42)

# Configuration
@pytest.fixture


def config():
    return CGPConfig(
        num_inputs=10,
        num_nodes=10,
        num_outputs=3,
        function_set = functions,
        const_min=-1,
        const_max=1,
        input_node_chance=0.4
    )

# Test data
@pytest.fixture
def X(config):

    X = np.random.uniform(-1, 1, size=(50, config.num_inputs))
    X = X.astype(float)
    return X  # 50 sample, 10 features

@pytest.fixture
def y(X):
    y = X[:, 0] + 2 * X[:, 1]
    y = y.astype(float)
    return y  # simple linear relation 

# Génome de test
@pytest.fixture
def sample_genome(config):
    return CGPGenome.create_genome(config)

# Mutation
@pytest.fixture
def mutation(config):
    return Golden_mutation(config,input_node_mutation_rate=0.2, function_mutation_rate=0.4, input_mutation_rate=0.5, const_mutation_rate=0.1,output_node_mutation_rate=0.5)


# Evaluator
@pytest.fixture
def evaluator(X, y):
    return Regressor(X, y)


# Evolution Strategy
@pytest.fixture
def sample_es(sample_genome, evaluator, mutation):
    return ES(
        evaluator=evaluator,
        lam=4,
        parent=lambda: sample_genome,
        mutation=mutation,
    )

# -------------------
# TESTS Regression
# -------------------

def test_genome_is_valid(sample_genome,config):
    assert isinstance(sample_genome, CGPGenome)
    assert len(sample_genome.nodes) == config.num_nodes
    assert isinstance(sample_genome.outputs, list)

def test_mutation_changes_genome(sample_genome, mutation):
    original = sample_genome.copy()
    mutation.mutate(sample_genome)
    assert original != sample_genome, "Mutation did not change the genome"

def test_Regressor_returns_score(evaluator, sample_genome):
    score = evaluator.evaluate(sample_genome)
    assert isinstance(score, float)
    assert score <= 1.0

def test_es_process(sample_es,evaluator):
    first_genome = sample_es.parent.copy()
    best_genome = sample_es.evolve(n_generations=100, early_stopping=100, verbose=False)
    assert best_genome is not None
    assert evaluator.evaluate(best_genome) > evaluator.evaluate(first_genome)


####
@pytest.fixture
def y2(X):
    y2 = X[:, 0] + 2 * X[:, 1]
    y2 = np.select(
        [y2 < 0.33, (y2 >= 0.33) & (y2 < 0.66), y2 >= 0.66],
        [0, 1, 2]
    )
    return y2


# Evaluator
@pytest.fixture
def evaluator2(X, y2):
    return MultiClassClassifier(X, y2)

# Evolution Strategy
@pytest.fixture
def sample_es2(sample_genome, evaluator2, mutation):
    return ES(
        evaluator=evaluator2,
        lam=4,
        parent=sample_genome,
        mutation=mutation,
    )
# -------------------
# TESTS Classification


def test_Classification_returns_accuracy(evaluator2, sample_genome):
    accuracy = evaluator2.evaluate(sample_genome)
    assert isinstance(accuracy, float)
    assert 0.0 <= accuracy <= 1.0

def test_es_process_classification(sample_es2, evaluator2):
    first_genome = sample_es2.parent.copy()
    best_genome = sample_es2.evolve(n_generations=100, early_stopping=100, verbose=False)
    assert best_genome is not None
    assert evaluator2.evaluate(best_genome) > evaluator2.evaluate(first_genome)
