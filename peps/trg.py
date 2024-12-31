import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)

__all__ = [
    'TensorRG'
]


class TensorRG:
    
    def __init__(self, ta, tb):
        self.ta = ta
        self.tb = tb
        
    @ property
    def D(self):
        return self.ta.shape[0] 
    
    def trace(self):
        ta = self.ta
        tb = self.tb
        t1 = jnp.tensordot(tb, ta, ([1], [1]))
        res = jnp.tensordot(t1, t1, ([0, 3], [1, 2]))
        res = jnp.tensordot(res, t1, ([0, 1, 2, 3], [3, 0, 2, 1]))
        
        return res
    
    def renormalization(self, maxd):
        
        ta = self.ta
        tb = self.tb
        dim = self.D
        D = min(dim**2, maxd)
        
        T1 = jnp.tensordot(ta, tb, ([0], [0])).transpose([1, 2, 3, 0]).reshape([dim**2, dim**2])
        U, s, Vh = jnp.linalg.svd(T1)
        s = jnp.sqrt(s)
        sa1 = (U[:, :D] * s[:D]).reshape([dim, dim, D])
        sb1 = jnp.dot(jnp.diag(s[:D]), Vh[:D, :]).reshape([D, dim, dim])

        T2 = jnp.tensordot(tb, ta, ([2], [2])).transpose([1, 2, 3, 0]).reshape([dim**2, dim**2])
        U, s, Vh = jnp.linalg.svd(T2)
        s = jnp.sqrt(s)
        sb2 = (U[:, :D] * s[:D]).reshape([dim, dim, D])
        sa2 = jnp.dot(jnp.diag(s[:D]), Vh[:D, :]).reshape([D, dim, dim])

        T3 = jnp.tensordot(ta, tb, ([1], [1])).transpose([2, 1, 0, 3]).reshape([dim**2, dim**2])
        U, s, Vh = jnp.linalg.svd(T3)
        s = jnp.sqrt(s)
        sb3 = (U[:, :D] * s[:D]).reshape([dim, dim, D])
        sa3 = jnp.dot(jnp.diag(s[:D]), Vh[:D, :]).reshape([D, dim, dim])
        
        sa = jnp.einsum('lmi, kmn, jnl -> ijk', sa1, sa2, sa3)
        sb = jnp.einsum('ilm, mnk, nlj -> ijk', sb1, sb2, sb3)
    
        return sa, sb
    
    def start(self, maxd, T, ite=10):
        ca = []
        cb = []
        for i in range(ite):
            sa, sb = self.renormalization(maxd)
            ca.append(jnp.linalg.norm(sa))
            cb.append(jnp.linalg.norm(sb))
            self.ta = sa.copy() / ca[-1]
            self.tb = sb.copy() / cb[-1]
        ca = jnp.array(ca)
        cb = jnp.array(cb)
        cn = self.trace() 

        return -T * jnp.sum(jnp.log(ca*cb) * jnp.array([1 / (2 * (3**i)) for i in range(1, ite+1)]))


if __name__ == '__main__':
    # ising model in triangle lattice    
    J = 1
    D = 8
    step=0.05
    tlist = np.arange(0.1, 4.5, step)

    f = []
    for t in tlist:
        beta = 1 / t
        ta = jnp.zeros((2, 2, 2), dtype=jnp.float64)
        tb = jnp.array((2, 2, 2), dtype=jnp.float64)

        ta = ta.at[([1, 0, 1, 0], [1, 0, 0, 1], [1, 1, 0, 0])].set([
            np.exp((3/2)*beta*J), np.exp(-(1/2)*beta*J), np.exp(-(1/2)*beta*J), np.exp(-(1/2)*beta*J)
        ])
        tb = ta.copy()

        f.append(TensorRG(ta, tb).start(D, t, ite=10))