import cPickle
import os
from itertools import izip, product
from scipy.stats import mode
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tensorflow as tf

import billboard
from datasets import dump_datasets, load_datasets

# TODO: remove
if 'sess' in globals():
    tf.ops.reset_default_graph()
    #sess = tf.InteractiveSession()

if not 'flags' in globals():
    flags = tf.flags
    FLAGS = flags.FLAGS
    flags.DEFINE_string('billboard_path', None, 'Path to McGill/Billboard ground truth files')
    flags.DEFINE_string('backend', 'numpy', 'numpy/tensorflow')
    flags.DEFINE_string('feature_type', 'cqt', 'cqt/chromagram/unaligned_chromagram')

class EchoStateNetwork(object):

    def __init__(
            self,
            n_inputs=12,
            n_hidden=1000,
            n_outputs=26,
            n_steps=200,
            spectral_radius=1.0,
            connectivity=0.1,
            min_leakage=0.3,
            max_leakage=0.9,
            ridge_beta=0.0,
            iter_learning_rate=0.05,
            input_scaling=0.01,
            input_shift=0,
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.min_leakage = min_leakage
        self.max_leakage = max_leakage
        self.ridge_beta = ridge_beta
        self.iter_learning_rate = iter_learning_rate
        self.input_scaling = input_scaling
        self.input_shift = input_shift
        self.n_hidden_2 = self.n_hidden * 2 + self.n_inputs

    def generate_hh_weights(self):
        # in numpy for now
        weights = scipy.sparse.rand(
            self.n_hidden, self.n_hidden, self.connectivity).tocsr()
        weights.data -= .5
        for attempt in range(5):
            try:
                eigvals = scipy.sparse.linalg.eigs(
                    weights, k=1, which='LM',
                    return_eigenvectors=False, tol=.02, maxiter=5000)
                break
            except (scipy.sparse.linalg.ArpackNoConvergence,
                    scipy.sparse.linalg.ArpackError):
                continue
        else:
            #print 'scipy.sparse.linalg failed to converge, falling back to numpy.linalg'
            eigvals = np.linalg.eigvals(weights.todense())
        radius = np.abs(np.max(eigvals))
        weights /= radius
        weights *= self.spectral_radius

        return weights

class DirectionNumPy(object):

    def __init__(self, esn, backwards=False):
        self.esn = esn
        self.backwards = backwards
        self.w_xh = np.random.normal(0, .2, [esn.n_inputs, esn.n_hidden])
        self.w_hh = esn.generate_hh_weights()
        self.leakage = np.linspace(
            esn.min_leakage, esn.max_leakage, esn.n_hidden).reshape([1, esn.n_hidden])

    def compute_states(self, features):
        states_list = []
        state = np.zeros([features.shape[0], self.esn.n_hidden])

        timesteps = xrange(self.esn.n_steps)
        if self.backwards:
            timesteps = reversed(timesteps)

        for i in timesteps:
            prev_state = state
            state = np.tanh(features[:, i, :].dot(self.w_xh) +
                            self.w_hh.dot(prev_state.T).T)

            state = self.leakage * prev_state + (1 - self.leakage) * state
            #state = np.random.random(state.shape)
            states_list.append(state)

        if self.backwards:
            states_list.reverse()

        return np.array(states_list).swapaxes(0, 1)

class EchoStateNetworkNumPy(EchoStateNetwork):

    def __init__(self, *args, **kwargs):
        super(EchoStateNetworkNumPy, self).__init__(*args, **kwargs)

        self.forward = DirectionNumPy(self)
        self.backward = DirectionNumPy(self, backwards=True)
        self.w_hy = np.zeros([self.n_hidden_2, self.n_outputs])

    def states(self, features):
        scaled_features = features * self.input_scaling + self.input_shift
        forward_states = self.forward.compute_states(scaled_features)
        backward_states = self.backward.compute_states(scaled_features)

        return np.concatenate((forward_states, backward_states,
                               scaled_features
                           ), 2)

    def train(self, features, targets):
        states = self.states(features)
        states = np.reshape(states, [-1, self.n_hidden_2])
        outputs = np.reshape(targets, [-1, self.n_outputs])
        weights = ridge_regression(states, outputs, self.ridge_beta)
        self.w_hy = weights

    def y(self, features, trace=None):
        states = self.states(features)

        if trace is not None:
            trace.update({'states': states})

        return np.dot(
            states.reshape([-1, self.n_hidden_2]),
            self.w_hy).reshape([-1, self.n_steps, self.n_outputs])

    def accuracy(self, features, targets, trace=None):
        y = self.y(features, trace)
        predicted = np.argmax(y, 2)
        actual = np.argmax(targets, 2)
        correct = predicted == actual

        if trace is not None:
            trace.update({'predicted': predicted,
                          'actual': actual,
                          'correct': correct,
                          'y': y})

        return np.mean(correct)

    def save(self, filename):
        with open(filename, 'w') as f:
            cPickle.dump(self, f, protocol=cPickle.HIGHEST_PROTOCOL)


class DirectionTensorFlow(object):

    def __init__(self, esn, backwards=False):
        self.esn = esn
        self.backwards = backwards

        self.w_xh = tf.Variable(
            tf.random_uniform([esn.n_inputs, esn.n_hidden], -.5, .5),
            name='w_xh', trainable=False)

        self.w_hh = tf.Variable(
            tf.constant(
                esn.generate_hh_weights().todense(), dtype='float32', name='w_hh'),
            name='w_hh', trainable=False)

        self.leakage = tf.reshape(
            tf.linspace(esn.min_leakage, esn.max_leakage, esn.n_hidden),
            [1, esn.n_hidden], name='leakage')

    def compute_states(self):
        states_list = []
        state = self.zero_state()

        timesteps = xrange(self.esn.n_steps)
        if self.backwards:
            timesteps = reversed(timesteps)

        for i in timesteps:
            prev_state = state

            state = (tf.matmul(self.esn.scaled_x[:, i, :], self.w_xh) +
                     tf.matmul(prev_state, self.w_hh, b_is_sparse=True))

            state = (1 - self.leakage) * tf.tanh(state) + self.leakage * prev_state

            states_list.append(state)

        states = tf.reshape(tf.concat(
            1, states_list), [-1, self.esn.n_steps, self.esn.n_hidden])

        if self.backwards:
            states = tf.reverse(states, [False, True, False])

        return states

    def zero_state(self):
        shape = tf.pack([self.esn.batch_size(), self.esn.n_hidden])
        return tf.cast(tf.fill(shape, 0), 'float32')


class EchoStateNetworkTensorFlow(EchoStateNetwork):

    def __init__(self, *args, **kwargs):
        super(EchoStateNetworkTensorFlow, self).__init__(*args, **kwargs)

        self.x = tf.placeholder('float32', [None, self.n_steps, self.n_inputs], name='x')
        self.targets = tf.placeholder(
            'float32', [None, self.n_steps, self.n_outputs], name='targets')
        self.w_hy = tf.Variable(
            tf.random_uniform([self.n_hidden_2, self.n_outputs], -.5, .5), name='w_hy')

        self.scaled_x = self.x * self.input_scaling + self.input_shift

        self.forward = DirectionTensorFlow(self)
        self.backward = DirectionTensorFlow(self, backwards=True)

        self.forward_states = self.forward.compute_states()
        self.backward_states = self.backward.compute_states()

        self.states = tf.concat(2, [self.forward_states, self.backward_states,
                                    self.scaled_x])

        self.y = tf.reshape(
            tf.nn.softmax(
                tf.matmul(tf.reshape(self.states, [-1, self.n_hidden_2]), self.w_hy)),
            [-1, self.n_steps, self.n_outputs]
        )

        self.cost = -tf.reduce_sum(self.targets * tf.log(self.y))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        optimizer = tf.train.GradientDescentOptimizer(self.iter_learning_rate)
        self.train_iter = optimizer.apply_gradients(zip(grads, tvars))
        self.train_tf = self.make_train_tf()

        self.predicted = tf.argmax(self.y, 2)
        actual = tf.argmax(self.targets, 2)
        correct = tf.equal(self.predicted, actual)
        self.accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    def batch_size(self):
        return tf.shape(self.x)[0]

    def make_train_tf(self):
        x = tf.reshape(self.states, [-1, self.n_hidden_2])
        y = tf.reshape(self.targets, [-1, self.n_outputs])

        indices = tf.random_shuffle(tf.range(tf.shape(x)[0]))
        x_ = tf.gather(x, indices)
        y_ = tf.gather(y, indices)

        r = tf.matmul(tf.transpose(x_), x_)
        p = tf.matmul(tf.transpose(x_), y_)
        i = tf.diag(tf.ones([self.n_hidden_2]))
        w = tf.matmul(tf.matrix_inverse(r + self.ridge_beta * i), p)

        return tf.assign(self.w_hy, w)

    def train_np(self, states, targets):
        states = np.reshape(states, [-1, self.n_hidden])
        targets = np.reshape(targets, [-1, self.n_outputs])
        weights = ridge_regression(states, targets, self.ridge_beta)
        return tf.assign(self.w_hy, weights)

def ridge_regression(x, y, beta):
    r = x.T.dot(x)
    p = x.T.dot(y)
    return np.linalg.inv(
        r + beta * np.eye(x.shape[1])
    ).dot(p)

def test_esn():
    deltas = np.repeat(np.random.random(400), 100)
    x = np.sin(np.cumsum(deltas)).reshape([4, 10000]).T
    y = np.zeros((10000, 2), 'float32')
    y[np.arange(10000), np.argmax(x, 1) / 2] = 1
    x = x.reshape([10, 1000, 4])
    y = y.reshape([10, 1000, 2])

def train_tf():
    n_steps = esn.n_steps
    n_hop = n_steps // 2
    train_batch = datasets.train.single_batch(n_steps, n_hop)
    sess.run(esn.train_tf, {esn.x: train_batch.features[:, :, :84],
                            esn.targets: train_batch.targets})

    test_batch = datasets.test.single_batch(n_steps, n_hop)
    accuracy = sess.run(esn.accuracy, {esn.x: test_batch.features[:, :, :84],
                                       esn.targets: test_batch.targets})
    return accuracy, train_batch, test_batch

def train_np():
    n_steps = esn.n_steps
    n_hop = n_steps // 2
    train_batch = datasets.train.single_batch(n_steps, n_hop)
    esn.train(train_batch.features, train_batch.targets)

    test_batch = datasets.test.single_batch(n_steps, n_hop)
    accuracy = esn.accuracy(test_batch.features, test_batch.targets)

    return accuracy, train_batch, test_batch

def impulse_demo(esn):
    x = np.zeros([1, esn.n_steps, esn.n_inputs])
    x[:, ::50, :] = 1
    if isinstance(esn, EchoStateNetworkTensorFlow):
        return sess.run(esn.states, {esn.x: x})[0]
    else:
        return esn.states(x)[0][:, :-esn.n_inputs]

def input_demo(esn, datasets):
    x = datasets.train.next_batch(steps=esn.n_steps, hop=1, batch_size=1).features[:, :, :84]
    if isinstance(esn, EchoStateNetworkTensorFlow):
        return sess.run(esn.states, {esn.x: x})[0]
    else:
        return esn.states(x)[0][:, :-esn.n_inputs]

def main(unused_args):
    require_flag('billboard_path')

    feature_type = FLAGS.feature_type
    cache_filename = '../billboard-datasets-%s.cpkl' % feature_type
    if os.path.exists(cache_filename):
        datasets = load_datasets(cache_filename)
    else:
        datasets = billboard.read_billboard_datasets(FLAGS.billboard_path,
                                                     feature_type=FLAGS.feature_type)
        dump_datasets(datasets, cache_filename)

    # grid = {
    #     'n_hidden': [300, 800, 1300, 1800, 2300, 2800, 3300],
    #     'n_steps': [200],
    #     'spectral_radius': [1.],
    #     'connectivity': [.01, .0001],
    #     'max_leakage': [.99],
    #     'min_leakage': [.3],
    #     'ridge_beta': [0, .5],
    #     'input_scaling': [.2],
    #     'input_shift': [0],
    # }

    grid = {
        'n_hidden': [2000],
        'n_steps': [200],
        'spectral_radius': [1.],
        'connectivity': [.01],
        'max_leakage': [.99],
        'min_leakage': [.3],
        'ridge_beta': [.5],
        'input_scaling': [.2],
        'input_shift': [0],
    }
    grid_product = list(dict_product(grid))
    #random.shuffle(grid_product)

    for config in grid_product:

        config['n_inputs'] = datasets.train.feature_reader.num_features
        config['n_outputs'] = datasets.train.label_reader.num_labels

        print '\n%s' % config

        if FLAGS.backend == 'numpy':
            numpy_eval(datasets, config)
        elif FLAGS.backend == 'tensorflow':
            tensorflow_eval(datasets, config)
        else:
            print 'unknown backend'

def tensorflow_eval(datasets, config):

    with tf.Graph().as_default(), tf.Session() as sess:

        n_steps = config['n_steps']
        n_hop = n_steps #/ 2
        train_batch = datasets.train.single_batch(n_steps, n_hop)
        test_batch = datasets.test.single_batch(n_steps, n_hop)

        ys = []
        accuracies = []

        for _ in range(3):
            print '.',
            esn = EchoStateNetworkTensorFlow(**config)
            sess.run(tf.initialize_all_variables())

            sess.run(esn.train_tf, {esn.x: train_batch.features[:, :, :84],
                                    esn.targets: train_batch.targets})

            accuracy, y = sess.run([esn.accuracy, esn.y], {
                esn.x: test_batch.features[:, :, :84], esn.targets: test_batch.targets})

            ys.append(y)
            accuracies.append(accuracy)

        ensemble_predictions = mode(np.argmax(ys, 3), 0)[0].reshape([-1, n_steps])
        ensemble_accuracy = np.mean(ensemble_predictions == test_batch.targets.argmax(2))

        print 'ensemble accuracy: %.2f%%; mean accuracy: %.2f%%' % (
            ensemble_accuracy * 100, np.mean(accuracies) * 100)

def numpy_eval(datasets, config, iterations=1):

    ys = []
    lfn = []
    accuracies = []

    esns = []

    for i in range(iterations):

        print i

        esn = EchoStateNetworkNumPy(**config)
        esns.append(esn)

        n_steps = esn.n_steps
        n_hop = n_steps #/ 2

        train_batch = datasets.train.single_batch(n_steps, n_hop)
        test_batch = datasets.test.single_batch(n_steps, n_hop)

        esn.train(train_batch.features[:, :, :84], train_batch.targets)

        trace = {}
        accuracy = esn.accuracy(test_batch.features[:, :, :84], test_batch.targets, trace)
        learned_from_network = learned_from_network_np(esn, test_batch, trace['states'])
        y = trace['y']

        ys.append(y)
        accuracies.append(accuracy)
        lfn.append(learned_from_network)

    ensemble_predictions = mode(np.argmax(ys, 3), 0)[0].reshape([-1, n_steps])
    ensemble_accuracy = np.mean(ensemble_predictions == test_batch.targets.argmax(2))

    print 'ensemble_accuracy: %.2f%% mean_accuracy: %.2f%% mean_learned_from_network: %.2f%%' % (ensemble_accuracy * 100, np.mean(accuracies) * 100, np.mean(lfn) * 100)

    return esns
        

def learned_from_network_np(esn, batch, states):
    states = np.reshape(states, [-1, esn.n_hidden_2])
    targets = batch.targets
    def score(indices):
        y = np.reshape(states[:, indices].dot(esn.w_hy[indices, :]), [-1, esn.n_steps, esn.n_outputs])
        return np.mean(targets.argmax(2) == y.argmax(2))

    return score(np.arange(esn.n_hidden_2 - 84)) - score(np.arange(esn.n_hidden_2 - 84, esn.n_hidden_2))

def dict_product(d):
    return (dict(izip(d, x)) for x in product(*d.itervalues()))

def require_flag(flag_name):
    if FLAGS.__flags[flag_name] is None:
        raise ValueError('Must set --%s' % flag_name)

if __name__ == '__main__':
    tf.app.run()
