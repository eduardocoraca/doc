import numpy as np
import scipy.linalg
from tqdm import tqdm

class Sistema_mck():
    def __init__(self,M,C,K,T,dt,p):
        self.__M = M
        self.__C = C
        self.__K = K
        self.T = T
        self.dt = dt
        self.p = np.linspace(p[0], p[1], p[2])
        self.__p_samples = p[2]
        self.__dof = M.shape[0]
        self.time = np.linspace(0,T,int(T//dt))
        self.__x = np.zeros(M.shape[0]*2)
        self.wn = []
        self.states = np.empty(shape=(2*self.__dof, int(T//dt), self.__p_samples))

    def __matrizes(self,k):
        # k: posicao no vetor p
        iM = np.linalg.inv(self.__M)
        iMK = np.matmul(iM,self.__K*self.p[k])
        iMC = np.matmul(iM,self.__C)

        Ac = np.concatenate((np.concatenate((np.zeros(shape=(self.__dof,self.__dof)), np.eye(self.__dof)), axis=1),
                    np.concatenate((-iMK, -iMC), axis=1)), axis=0)
        Bc = np.concatenate((np.concatenate((np.zeros(shape=(self.__dof,self.__dof)), np.zeros(shape=(self.__dof,self.__dof))), axis=1),
                    np.concatenate((np.zeros(shape=(self.__dof,self.__dof)), iM), axis=1)), axis=0)

        A = scipy.linalg.expm(Ac*self.dt)
        B = (A-np.eye(A.shape[1]))*np.linalg.inv(Ac)*Bc

        l,_ = np.linalg.eig(Ac)
        wn = np.sqrt(np.abs(l))

        return A,B,wn

    def __u(self, n,k):
        return np.random.uniform(low=-1, high=1, size=2*self.__dof)*self.p[k]+self.p[k]

    def simular(self):
        for k in tqdm(range(0, len(self.p))):
            A,B,wn = self.__matrizes(k)
            self.wn.append(wn)
            for n in range(1,len(self.time)):
                # x[n+1] = Ad*x[n] + Bd*u[n]
                self.__x = np.squeeze(np.asarray( np.matmul(A,self.__x) + np.matmul(B,self.__u(n,k))))
                self.states[:,n,k] = self.__x
    
