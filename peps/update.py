import jax.numpy as jnp
import numpy as np
from jax import config
config.update("jax_enable_x64", True)
import scipy

from .trg import TensorRG

__all__ = [
    'PepsTrg',
    'PEPS'
        ]


class PepsTrg(TensorRG):

    def __init__(self, ta, tb, sta, stb):
        super().__init__(ta, tb)
        self.sta = sta
        self.stb = stb

    def trace(self, uniform=True):
        ta = self.ta
        tb = self.tb
        if uniform:
            t1 = jnp.tensordot(tb, ta, ([1], [1]))
            res = jnp.tensordot(t1, t1, ([0, 3], [1, 2]))
            res = jnp.tensordot(res, t1, ([0, 1, 2, 3], [3, 0, 2, 1]))
        else:
            t1 = jnp.tensordot(self.sta, self.stb, ([0], [0]))
            res = jnp.tensordot(ta, tb, ([0], [0]))
            res = jnp.tensordot(res, res, ([0, 3], [2, 1]))
            res = jnp.tensordot(res, t1, ([0, 1, 2 ,3], [0, 3, 2, 1]))
        
        return res

    def renormalization(self, maxd):
        ta = self.ta
        tb = self.tb
        sta = self.sta
        stb = self.stb
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
        
        sT = jnp.tensordot(sta, stb, ([0], [0])).transpose([1, 2, 3, 0]).reshape([dim**2, dim**2])
        U, s, Vh = jnp.linalg.svd(sT)
        s = jnp.sqrt(s)
        ssa = (U[:, :D] * s[:D]).reshape([dim, dim, D])
        ssb = jnp.dot(jnp.diag(s[:D]), Vh[:D, :]).reshape([D, dim, dim])
        
        sa = jnp.einsum('lmi, kmn, jnl -> ijk', sa1, sa2, sa3)
        sb = jnp.einsum('ilm, mnk, nlj -> ijk', sb1, sb2, sb3)
        ssa = jnp.einsum('lmi, kmn, jnl -> ijk', ssa, sa2, sa3)
        ssb = jnp.einsum('ilm, mnk, nlj -> ijk', ssb, sb2, sb3)

        return sa, sb, ssa, ssb
    
    def start(self, maxd, ite=10):
        c = []
        for i in range(ite):
            sa, sb, ssa, ssb = self.renormalization(maxd)
            c_i = (jnp.linalg.norm(ssa) * jnp.linalg.norm(ssb)) / (jnp.linalg.norm(sa) * jnp.linalg.norm(sb))
            c.append(c_i)
            self.ta = sa.copy() / jnp.linalg.norm(sa)
            self.tb = sb.copy() / jnp.linalg.norm(sb)
            self.sta = ssa.copy() / jnp.linalg.norm(ssa)
            self.stb = ssb.copy() / jnp.linalg.norm(ssb)
            
        cn_observable = self.trace(uniform=False)
        cn_overlap = self.trace(uniform=True)
        
        return (cn_observable / cn_overlap) * np.prod(c)
    

class PEPS:
    
    def __init__(self, D, maxd):
        self.ta, self.tb, self.lamda = self.initialize(D)
        self.maxd = maxd
        self.D = D
        
    @ property
    def ta_lamda(self):
        return jnp.einsum('ijko, il, jm, kn -> lmno', self.ta, 
                          jnp.diag(self.lamda[0]), 
                          jnp.diag(self.lamda[1]), 
                          jnp.diag(self.lamda[2]))
    
    def initialize(self, D):
        d = 2
        ta = jnp.array(np.random.randn(D, D, D, d) + 1j * np.random.randn(D, D, D, d))
        tb = jnp.array(np.random.randn(D, D, D, d) + 1j * np.random.randn(D, D, D, d))
        ta = ta / jnp.linalg.norm(ta)
        tb = tb / jnp.linalg.norm(tb)

        lamda = [np.random.randn(D) for _ in range(3)]
        for i in range(3):
            lamda[i] = lamda[i] / np.linalg.norm(lamda[i])
        lamda = [jnp.array(i) for i in lamda]
        
        return ta, tb, lamda
    
    def evolution_op(self, tau, J):
        sx = (1 / 2) * jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        sy = (1 / 2) * jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
        sz = (1 / 2) * jnp.array([[1, 0], [0, -1]],dtype=jnp.complex128)
        
        op = (jnp.tensordot(sx, sx, axes=0) + jnp.tensordot(sy, sy, axes=0) + 
             jnp.tensordot(sz, sz, axes=0)).transpose([0, 2, 1, 3]).reshape(4, 4)       
        
        op = jnp.array(scipy.linalg.expm(- tau * J * op)).reshape([2, 2, 2, 2])
        
        return op
    
    def single_update(self, tau, J):
        op = self.evolution_op(tau, J)
        
        # 0 - 0
        ta = self.ta_lamda
        tb = jnp.einsum('ljkp, mj, nk-> lmnp', self.tb, jnp.diag(self.lamda[1]), jnp.diag(self.lamda[2]))
        S = jnp.einsum('ijko, imnp, abop -> ajkbmn', ta, tb, op).reshape(2 * (self.D**2), 2 * (self.D**2))
        U, s, Vh = jnp.linalg.svd(S)
        self.lamda[0] = s[:self.D] / jnp.linalg.norm(s[:self.D])
        self.ta = U[:, :self.D].reshape(2, self.D, self.D, self.D).transpose([3, 1, 2, 0])
        self.ta = jnp.einsum('ilma, lj, mk -> ijka', self.ta, jnp.diag(1 / self.lamda[1]), jnp.diag(1 / self.lamda[2]))

        self.tb = Vh[:self.D, :].reshape(self.D, 2, self.D, self.D).transpose([0, 2, 3, 1])
        self.tb = jnp.einsum('ilmb, jl, km -> ijkb', self.tb, jnp.diag(1 / self.lamda[1]), jnp.diag(1 / self.lamda[2]))
        
        # 2 - 2
        ta = self.ta_lamda
        tb = jnp.einsum('ijnp, li, mj-> lmnp', self.tb, jnp.diag(self.lamda[0]), jnp.diag(self.lamda[1]))
        S = jnp.einsum('ijko, lmkp, abop -> aijblm', ta, tb, op).reshape(2 * (self.D**2), 2 * (self.D**2))
        U, s, Vh = jnp.linalg.svd(S)
        self.lamda[2] = s[:self.D] / jnp.linalg.norm(s[:self.D])
        self.ta = U[:, :self.D].reshape(2, self.D, self.D, self.D).transpose([1, 2, 3, 0])
        self.ta = jnp.einsum('lmka, li, mj -> ijka', self.ta, jnp.diag(1 / self.lamda[0]), jnp.diag(1 / self.lamda[1]))

        self.tb = Vh[:self.D, :].reshape(self.D, 2, self.D, self.D).transpose([2, 3, 0, 1])
        self.tb = jnp.einsum('lmkb, il, jm -> ijkb', self.tb, jnp.diag(1 / self.lamda[0]), jnp.diag(1 / self.lamda[1]))
        
        # 1- 1
        ta = self.ta_lamda
        tb = jnp.einsum('imkp, li, nk-> lmnp', self.tb, jnp.diag(self.lamda[0]), jnp.diag(self.lamda[2]))
        S = jnp.einsum('ijko, ljnp, abop -> aikbln', ta, tb, op).reshape(2 * (self.D**2), 2 * (self.D**2))
        U, s, Vh = jnp.linalg.svd(S)
        self.lamda[1] = s[:self.D] / jnp.linalg.norm(s[:self.D])
        self.ta = U[:, :self.D].reshape(2, self.D, self.D, self.D).transpose([1, 3, 2, 0])
        self.ta = jnp.einsum('ljna, li, nk -> ijka', self.ta, jnp.diag(1 / self.lamda[0]), jnp.diag(1 / self.lamda[2]))

        self.tb = Vh[:self.D, :].reshape(self.D, 2, self.D, self.D).transpose([2, 0, 3, 1])
        self.tb = jnp.einsum('ljnb, il, kn -> ijkb', self.tb, jnp.diag(1 / self.lamda[0]), jnp.diag(1 / self.lamda[2]))
        
    def cal_energy(self, J, ite=10):
        e = []
        sx = (1/2) * jnp.array([[0, 1], [1, 0]], dtype=jnp.complex128)
        sy = (1/2) * jnp.array([[0, -1j], [1j, 0]], dtype=jnp.complex128)
        sz = (1/2) * jnp.array([[1, 0], [0, -1]], dtype=jnp.complex128)
        opl = [sx, sy, sz]

        lamdax = jnp.diag(jnp.sqrt(self.lamda[0]))
        lamday = jnp.diag(jnp.sqrt(self.lamda[1]))
        lamdaz = jnp.diag(jnp.sqrt(self.lamda[2]))
        
        ta = jnp.einsum('lmna, li, mj, nk -> ijka', self.ta, lamdax, lamday, lamdaz)
        tb = jnp.einsum('lmna, il, jm, kn -> ijka', self.tb, lamdax, lamday, lamdaz)
        dim = self.D**2
        Ta = jnp.tensordot(ta, ta.conj(), ([3], [3])).transpose([0, 3, 1, 4, 2, 5]).reshape(dim, dim, dim)
        Tb = jnp.tensordot(tb, tb.conj(), ([3], [3])).transpose([0, 3, 1, 4, 2, 5]).reshape(dim, dim, dim)

        for op in opl:
            ta1 = jnp.einsum('ijkb, lmna, ab -> iljmkn', ta, ta.conj(), op).reshape(dim, dim, dim)
            tb2 = jnp.einsum('ijkb, lmna, ab -> iljmkn', tb, tb.conj(), op).reshape(dim, dim, dim)
            res = PepsTrg(Ta, Tb, ta1, tb2).start(self.maxd, ite)
            e.append(res)
        
        return e


if __name__ == '__main__':
    peps = PEPS(3, 10)
    J = 1
    ite = 10
    energy = []
    tau = [0.1] * 100 + [0.01] * 100 + [0.001] * 100
    for i in range(300):
        peps.single_update(tau[i], J)
        e = peps.cal_energy(J, ite)
        E = np.sum(e)
        energy.append(1.5 * E)
        print(1.5 * E.real)