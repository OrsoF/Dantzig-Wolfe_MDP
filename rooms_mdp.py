import numpy as np

def coord(s, size):
    return s // size, s % size


def state(i, j, size):
    return i * size + j


def next_state(i, j, direction):
    assert isinstance(direction, int) and 0 <= direction < 4
    if direction == 0:
        return i - 1, j
    elif direction == 1:
        return i + 1, j
    elif direction == 2:
        return i, j + 1
    else:
        return i, j - 1


def is_in_grid(i, j, size):
    return 0 <= i < size and 0 <= j < size


def add_displacements(P, size):
    A, S, S = P.shape
    for a in range(A):
        for s in range(S):
            i, j = coord(s, size)
            next_coord = next_state(i, j, a)
            if is_in_grid(*next_coord, size):
                P[a, s, s] = 0.2
                P[a, s, state(*next_coord, size)] = 0.8
            else:
                P[a, s, s] = 1.
    return P


def add_wall_element(s1, s2, P, direction, size):
    assert direction in ['horizontal', 'vertical']
    i, j = coord(s1, size)
    k, l = coord(s2, size)
    assert (i - k) ** 2 + (j - l) ** 2 == 1
    if direction == 'vertical':
        assert j < l
        P[2, s1, s1] = 1.
        P[2, s1, s2] = 0.
        P[3, s2, s2] = 1.
        P[3, s2, s1] = 0.
    else:
        assert i < k
        P[1, s1, s1] = 1.
        P[1, s1, s2] = 0.
        P[0, s2, s2] = 1.
        P[0, s2, s1] = 0.
    return P


def add_walls(P, size):
    # Horizontal wall
    direction = 'horizontal'
    i, k = size // 2 - 1, size // 2
    for index in range(size):
        if index != size // 4 and index != 3 * size // 4:
            j, l = index, index
            s1, s2 = state(i, j, size), state(k, l, size)
            P = add_wall_element(s1, s2, P, direction, size)

    # Vertical wall
    direction = 'vertical'
    j, l = size // 2 - 1, size // 2
    for index in range(size):
        if index != size // 4 and index != 3 * size // 4:
            i, k = index, index
            s1, s2 = state(i, j, size), state(k, l, size)
            P = add_wall_element(s1, s2, P, direction, size)

    return P


def add_exit(P, R, size):
    i, j = 0, size // 4
    s = state(i, j, size)
    P[:, s, :] = 0
    P[:, s, s] = 1
    R[:, s, :] = 0

    return P, R

class Mdp:
    def __init__(self, n=5) -> None:
        self.dim = n
        self.A = 4
        self.S = 4 * self.dim * self.dim
        self.size = 2*self.dim

        self.R = -np.ones((self.A, self.S, self.S))
        self.P = np.zeros((self.A, self.S, self.S))
        self.P = add_displacements(self.P, self.size)
        self.P = add_walls(self.P, self.size)
        self.P, self.R = add_exit(self.P, self.R, self.size)

        
        self.R = self.R[:, :, 0]
        self.R = self.R.transpose()
        self.gamma = 0.999
        self.epsi = 1e-2
        self.__name__ = 'Rooms'+'_'+str(n)
