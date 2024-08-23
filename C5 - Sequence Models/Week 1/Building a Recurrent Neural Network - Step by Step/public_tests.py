import numpy as np
from rnn_utils import *

def rnn_cell_forward_tests(target):
    # Only bias in expression
    a_prev_tmp = np.zeros((5, 10))
    xt_tmp = np.zeros((3, 10))
    parameters_tmp = {}
    parameters_tmp['Waa'] = np.random.randn(5, 5)
    parameters_tmp['Wax'] = np.random.randn(5, 3)
    parameters_tmp['Wya'] = np.random.randn(2, 5)
    parameters_tmp['ba'] = np.random.randn(5, 1)
    parameters_tmp['by'] = np.random.randn(2, 1)
    parameters_tmp['Wya'] = np.zeros((2, 5))

    a_next_tmp, yt_pred_tmp, cache_tmp = target(xt_tmp, a_prev_tmp, parameters_tmp)
    
    assert a_next_tmp.shape == (5, 10), f"Wrong shape for a_next. Expected (5, 10) != {a_next_tmp.shape}"
    assert yt_pred_tmp.shape == (2, 10), f"Wrong shape for yt_pred. Expected (2, 10) != {yt_pred_tmp.shape}"
    assert cache_tmp[0].shape == (5, 10), "Wrong shape in cache->a_next"
    assert cache_tmp[1].shape == (5, 10), "Wrong shape in cache->a_prev"
    assert cache_tmp[2].shape == (3, 10), "Wrong shape in cache->x_t"
    assert len(cache_tmp[3].keys()) == 5, "Wrong number of parameters in cache. Expected 5"
    
    assert np.allclose(np.tanh(parameters_tmp['ba']), a_next_tmp), "Problem 1 in a_next expression. Related to ba?"
    assert np.allclose(softmax(parameters_tmp['by']), yt_pred_tmp), "Problem 1 in yt_pred expression. Related to by?"

    # Only xt in expression
    a_prev_tmp = np.zeros((5,10))
    xt_tmp = np.random.randn(3,10)
    parameters_tmp['Wax'] = np.random.randn(5,3)
    parameters_tmp['ba'] = np.zeros((5,1))
    parameters_tmp['by'] = np.zeros((2,1))

    a_next_tmp, yt_pred_tmp, cache_tmp = target(xt_tmp, a_prev_tmp, parameters_tmp)

    assert np.allclose(np.tanh(np.dot(parameters_tmp['Wax'], xt_tmp)), a_next_tmp), "Problem 2 in a_next expression. Related to xt?"
    assert np.allclose(softmax(np.dot(parameters_tmp['Wya'], a_next_tmp)), yt_pred_tmp), "Problem 2 in yt_pred expression. Related to a_next?"

    # Only a_prev in expression
    a_prev_tmp = np.random.randn(5,10)
    xt_tmp = np.zeros((3,10))
    parameters_tmp['Waa'] = np.random.randn(5,5)
    parameters_tmp['ba'] = np.zeros((5,1))
    parameters_tmp['by'] = np.zeros((2,1))

    a_next_tmp, yt_pred_tmp, cache_tmp = target(xt_tmp, a_prev_tmp, parameters_tmp)

    assert np.allclose(np.tanh(np.dot(parameters_tmp['Waa'], a_prev_tmp)), a_next_tmp), "Problem 3 in a_next expression. Related to a_prev?"
    assert np.allclose(softmax(np.dot(parameters_tmp['Wya'], a_next_tmp)), yt_pred_tmp), "Problem 3 in yt_pred expression. Related to a_next?"

    print("\033[92mAll tests passed")
    

def rnn_forward_test(target):
    np.random.seed(17)
    T_x = 13
    m = 8
    n_x = 4
    n_a = 7
    n_y = 3
    x_tmp = np.random.randn(n_x, m, T_x)
    a0_tmp = np.random.randn(n_a, m)
    parameters_tmp = {}
    parameters_tmp['Waa'] = np.random.randn(n_a, n_a)
    parameters_tmp['Wax'] = np.random.randn(n_a, n_x)
    parameters_tmp['Wya'] = np.random.randn(n_y, n_a)
    parameters_tmp['ba'] = np.random.randn(n_a, 1)
    parameters_tmp['by'] = np.random.randn(n_y, 1)

    a, y_pred, caches = target(x_tmp, a0_tmp, parameters_tmp)
    
    assert a.shape == (n_a, m, T_x), f"Wrong shape for a. Expected: ({n_a, m, T_x}) != {a.shape}"
    assert y_pred.shape == (n_y, m, T_x), f"Wrong shape for y_pred. Expected: ({n_y, m, T_x}) != {y_pred.shape}"
    assert len(caches[0]) == T_x, f"len(cache) must be T_x = {T_x}"
    
    assert np.allclose(a[5, 2, 2:6], [0.99999291, 0.99332189, 0.9921928, 0.99503445]), "Wrong values for a"
    assert np.allclose(y_pred[2, 1, 1: 5], [0.19428, 0.14292, 0.24993, 0.00119], atol=1e-4), "Wrong values for y_pred"
    assert np.allclose(caches[1], x_tmp), f"Fail check: cache[1] != x_tmp"

    
    print("\033[92mAll tests passed")
    
def lstm_cell_forward_test(target):
    np.random.seed(212)
    m = 8
    n_x = 4
    n_a = 7
    n_y = 3
    x = np.random.randn(n_x, m)
    a0 = np.random.randn(n_a, m)
    c0 = np.random.randn(n_a, m)
    params = {}
    params['Wf'] = np.random.randn(n_a, n_a + n_x)
    params['bf'] = np.random.randn(n_a, 1)
    params['Wi'] = np.random.randn(n_a, n_a + n_x)
    params['bi'] = np.random.randn(n_a, 1)
    params['Wo'] = np.random.randn(n_a, n_a + n_x)
    params['bo'] = np.random.randn(n_a, 1)
    params['Wc'] = np.random.randn(n_a, n_a + n_x)
    params['bc'] = np.random.randn(n_a, 1)
    params['Wy'] = np.random.randn(n_y, n_a)
    params['by'] = np.random.randn(n_y, 1)
    a_next, c_next, y_pred, cache = target(x, a0, c0, params)
    
    assert len(cache) == 10, "Don't change the cache"
    
    assert cache[4].shape == (n_a, m), f"Wrong shape for cache[4](ft). {cache[4].shape} != {(n_a, m)}"
    assert cache[5].shape == (n_a, m), f"Wrong shape for cache[5](it). {cache[5].shape} != {(n_a, m)}"
    assert cache[6].shape == (n_a, m), f"Wrong shape for cache[6](cct). {cache[6].shape} != {(n_a, m)}"
    assert cache[1].shape == (n_a, m), f"Wrong shape for cache[1](c_next). {cache[1].shape} != {(n_a, m)}"
    assert cache[7].shape == (n_a, m), f"Wrong shape for cache[7](ot). {cache[7].shape} != {(n_a, m)}"
    assert cache[0].shape == (n_a, m), f"Wrong shape for cache[0](a_next). {cache[0].shape} != {(n_a, m)}"
    assert cache[8].shape == (n_x, m), f"Wrong shape for cache[8](xt). {cache[8].shape} != {(n_x, m)}"
    assert cache[2].shape == (n_a, m), f"Wrong shape for cache[2](a_prev). {cache[2].shape} != {(n_a, m)}"
    assert cache[3].shape == (n_a, m), f"Wrong shape for cache[3](c_prev). {cache[3].shape} != {(n_a, m)}"
    
    assert a_next.shape == (n_a, m), f"Wrong shape for a_next. {a_next.shape} != {(n_a, m)}"
    assert c_next.shape == (n_a, m), f"Wrong shape for c_next. {c_next.shape} != {(n_a, m)}"
    assert y_pred.shape == (n_y, m), f"Wrong shape for y_pred. {y_pred.shape} != {(n_y, m)}"

    
    assert np.allclose(cache[4][0, 0:2], [0.32969833, 0.0574555]), "wrong values for ft"
    assert np.allclose(cache[5][0, 0:2], [0.0036446, 0.9806943]), "wrong values for it"
    assert np.allclose(cache[6][0, 0:2], [0.99903873, 0.57509956]), "wrong values for cct"
    assert np.allclose(cache[1][0, 0:2], [0.1352798,  0.39884899]), "wrong values for c_next"
    assert np.allclose(cache[7][0, 0:2], [0.7477249,  0.71588751]), "wrong values for ot"
    assert np.allclose(cache[0][0, 0:2], [0.10053951, 0.27129536]), "wrong values for a_next"
    
    assert np.allclose(y_pred[1], [0.417098, 0.449528, 0.223159, 0.278376,
                                   0.68453,  0.419221, 0.564025, 0.538475]), "Wrong values for y_pred"
    
    print("\033[92mAll tests passed")
    
def lstm_forward_test(target):
    np.random.seed(45)
    n_x = 4
    m = 13
    T_x = 16
    n_a = 3
    n_y = 2
    x_tmp = np.random.randn(n_x, m, T_x)
    a0_tmp = np.random.randn(n_a, m)
    parameters_tmp = {}
    parameters_tmp['Wf'] = np.random.randn(n_a, n_a + n_x)
    parameters_tmp['bf'] = np.random.randn(n_a, 1)
    parameters_tmp['Wi'] = np.random.randn(n_a, n_a + n_x)
    parameters_tmp['bi']= np.random.randn(n_a, 1)
    parameters_tmp['Wo'] = np.random.randn(n_a, n_a + n_x)
    parameters_tmp['bo'] = np.random.randn(n_a, 1)
    parameters_tmp['Wc'] = np.random.randn(n_a, n_a + n_x)
    parameters_tmp['bc'] = np.random.randn(n_a, 1)
    parameters_tmp['Wy'] = np.random.randn(n_y, n_a)
    parameters_tmp['by'] = np.random.randn(n_y, 1)

    a, y, c, caches = target(x_tmp, a0_tmp, parameters_tmp)
    
    assert a.shape == (n_a, m, T_x), f"Wrong shape for a. {a.shape} != {(n_a, m, T_x)}"
    assert c.shape == (n_a, m, T_x), f"Wrong shape for c. {c.shape} != {(n_a, m, T_x)}"
    assert y.shape == (n_y, m, T_x), f"Wrong shape for y. {y.shape} != {(n_y, m, T_x)}"
    assert len(caches[0]) == T_x, f"Wrong shape for caches. {len(caches[0])} != {T_x} "
    assert len(caches[0][0]) == 10, f"length of caches[0][0] must be 10."
    
    assert np.allclose(a[2, 1, 4:6], [-0.01606022,  0.0243569]), "Wrong values for a"
    assert np.allclose(c[2, 1, 4:6], [-0.02753855,  0.05668358]), "Wrong values for c"
    assert np.allclose(y[1, 1, 4:6], [0.70444592 ,0.70648935]), "Wrong values for y"
    
    print("\033[92mAll tests passed")