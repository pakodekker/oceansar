
import numpy as np


def factorize(n):
    """ Prime factorize a number

        :param n: Number

        :returns: Array with prime factors
    """
    result = np.array([])
    for i in np.append(2, np.arange(3, n + 1, 2)):
        s = 0
        while np.mod(n, i) == 0:
            n /= i
            s += 1
        result = np.append(result, [i]*s)
        if n == 1:
            return result


def nearest_power2(values):
    return (2**np.ceil(np.log2(values))).astype(int)


def optimize_fftsize(values, max_prime=2):
    """ Returns 'good' dimensions for FFT algorithm

        :param value: Input value/s
        :param max_prime: Maximum prime allowed (FFT is optimal for 2)

        :returns: Nearest 'good' value/s
    """
    # Force array type (if scalar was given)
    if np.isscalar(values):
        values = np.array([values], dtype=np.int)

    if max_prime == 2:
        good_values = nearest_power2(values)
        return good_values if len(good_values) > 1 else good_values[0]

    good_values = np.array([], dtype=np.int)
    for value in values:
        best_value = value
        while (np.max(factorize(best_value)) > max_prime):
            best_value += 1
        good_values = np.append(good_values, best_value)

    return good_values if len(good_values) > 1 else good_values[0]


def balance_elements(N, size):
    """ Divide N elements in size chunks
        Useful to balance arrays of size N not multiple
        of the number of processes

        :param N: Number of elements
        :param size: Number of divisions

        :returns: (counts, displ) vectors
    """
    # Counts
    count = np.round(N/size)
    counts = count*np.ones(size, dtype=np.int)
    diff = N - count*size
    counts[:diff] += 1

    # Displacements
    displ = np.concatenate(([0], np.cumsum(counts)[:-1]))

    return counts, displ