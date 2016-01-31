import numpy as np
import scipy.sparse
import scipy.sparse.linalg
import tensorflow as tf
import numpy as np

# TODO: remove
#ops.reset_default_graph()
#sess = tf.InteractiveSession()

class EchoStateNetwork(object):

    def __init__(
            self,
            n_inputs=4,
            n_hidden=50,
            n_outputs=2,
            n_steps=1000,
            #spectral_radius=1.0,
            spectral_radius=1.3,
            connectivity=0.1,
            min_leakage=0.0,
            max_leakage=1.0,
            feedback=0.0,
            ridge_beta=0.0,
            iter_learning_rate=0.01
    ):
        self.n_inputs = n_inputs
        self.n_hidden = n_hidden
        self.n_outputs = n_outputs
        self.n_steps = n_steps
        self.spectral_radius = spectral_radius
        self.connectivity = connectivity
        self.feedback = feedback
        self.ridge_beta = ridge_beta
        self.iter_learning_rate = iter_learning_rate
        
        self.x = tf.placeholder('float', [None, n_steps, n_inputs], name='x')
        self.y = tf.placeholder('float', [None, n_steps, n_outputs], name='y')

        self.w_xh = tf.Variable(
            tf.random_uniform([self.n_inputs, self.n_hidden], -.5, .5),
            name='w_xh', trainable=False)

        self.w_hh = tf.Variable(
            self.generate_hh_weights(),
            name='w_hh', trainable=False)

        self.w_yh = tf.Variable(
            tf.random_uniform([self.n_outputs, self.n_hidden], -.5, .5) * self.feedback,
            name='w_yh', trainable=False)

        self.w_hy = tf.Variable(
            tf.random_uniform([self.n_hidden, self.n_outputs], -.5, .5), name='w_hy')

        self.leakage = tf.Variable(
            tf.random_uniform([1, self.n_hidden], min_leakage, max_leakage),
            name='leakage', trainable=False)

        states = []
        state = self.zero_state()
        for i in range(n_steps):
            prev_state = state
            state = tf.tanh(tf.matmul(self.x[:, i, :], self.w_xh) +
                            tf.matmul(prev_state, self.w_hh) +
                            tf.matmul(self.y[:, i, :], self.w_yh))
            leaked_state = self.leakage * state + (1 - self.leakage) * prev_state
            states.append(leaked_state)
        self.states = tf.reshape(tf.concat(1, states), [-1, n_steps, n_hidden])

        self.y_ = tf.reshape(
            tf.nn.softmax(
                tf.matmul(tf.reshape(self.states, [-1, n_hidden]), self.w_hy)),
            [-1, n_steps, n_outputs]
        )

        self.cost = -tf.reduce_sum(self.y * tf.log(self.y_))

        # self.y_ = tf.reshape(
        #     tf.matmul(tf.reshape(self.states, [-1, n_hidden]), self.w_hy),
        #     [-1, n_steps, n_outputs])
        
        #cost = tf.reduce_mean(tf.pow(self.y - self.y_, 2))

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.cost, tvars), 5)
        self.train_iter = tf.train.GradientDescentOptimizer(iter_learning_rate).apply_gradients(zip(grads, tvars))
        
        #self.train_iter = tf.train.GradientDescentOptimizer(iter_learning_rate).minimize(self.cost)

    def zero_state(self):
        shape = tf.pack([self.batch_size(), self.n_hidden])
        return tf.cast(tf.fill(shape, 0), 'float32')

    def batch_size(self):
        return tf.shape(self.x)[0]

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
            print 'scipy.sparse.linalg failed to converge, falling back to numpy.linalg'
            eigvals = np.linalg.eigvals(weights.todense())
        radius = np.abs(np.max(eigvals))
        weights /= radius
        weights *= self.spectral_radius

        # no matmul for sparse matrices in tensorflow yet
        return tf.constant(weights.todense(), dtype='float32', name='w_hh')

    def train(self, states, outputs):
        states = np.reshape(states, [-1, self.n_hidden])
        outputs = np.reshape(outputs, [-1, self.n_outputs])
        weights = ridge_regression(states, outputs, self.ridge_beta)
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
    
