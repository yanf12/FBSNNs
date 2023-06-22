import numpy as np
import time
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
import math
from plotting import newfig, savefig


class neural_net(nn.Module):
    def __init__(self, pathbatch=100, n_dim=100 + 1, n_output=1):
        super(neural_net, self).__init__()
        self.pathbatch = pathbatch
        self.fc_1 = nn.Linear(n_dim, 256)
        #self.fc_2 = nn.Linear(256, 256)
        #self.fc_3 = nn.Linear(256, 256)
        #self.fc_4 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_output)

        self.relu = nn.ReLU()
        self.prelu = nn.PReLU()
        self.tanh = nn.Tanh()

        with torch.no_grad():
            torch.nn.init.xavier_uniform(self.fc_1.weight)
            #torch.nn.init.xavier_uniform(self.fc_2.weight)
            #torch.nn.init.xavier_uniform(self.fc_3.weight)
            #torch.nn.init.xavier_uniform(self.fc_4.weight)

    def forward(self, state, train=False):
        state = torch.sin(self.fc_1(state))
        #state = torch.sin(self.fc_2(state))
        #state = torch.sin(self.fc_3(state))
        #state = torch.sin(self.fc_4(state))
        fn_u = self.out(state)
        return fn_u


class FBSNN(nn.Module):  # Forward-Backward Stochastic Neural Network
    def __init__(self, r, mu, sigma, rho, K,alpha,Xi, T, M, N, D, learning_rate,gbm_scheme=1):
        super().__init__()
        self.r = r  # interest rate
        self.mu = mu  # drift rate
        self.sigma = sigma  # vol of each underlying
        self.rho = rho  # correlation, assumed to be constant across the basket
        self.K = K # strike price
        self.alpha = alpha # weights of each underlying
        self.Xi = Xi  # initial point
        self.T = T  # terminal time

        self.M = M  # number of trajectories
        self.N = N  # number of time snapshots
        self.D = D  # number of dimensions
        self.fn_u = neural_net(pathbatch=M, n_dim=D + 1, n_output=1)

        self.optimizer = optim.Adam(self.fn_u.parameters(), lr=learning_rate)
        var_cov_mat = np.zeros((D, D))
        self.gbm_scheme = gbm_scheme  # 0:euler scheme for gbm #1: EXP scheme
        for i in range(D):
            for j in range(D):
                if i == j:
                    var_cov_mat[i, j] = sigma[i] * sigma[j]
                else:
                    var_cov_mat[i, j] = rho * sigma[i] * sigma[j]
        var_cov_mat = torch.tensor(var_cov_mat)
        self.var_cov_mat = var_cov_mat
    def phi_torch(self, t, X, DuDt,DuDx,D2uDx2):  # M x 1, M x D, M x 1, M x D


        print(DuDt.shape)

        return   # M x 1






    def g_torch(self, X):  # M x D
        # terminal condition
        # print(X.shape)
        X_T = X.transpose(-2,1)

        res = torch.tensor(self.alpha) @ X_T
        # print(res.shape)
        # print(res)
        return torch.clamp(res-self.K,min = 0).unsqueeze(-1) # M x 1

    def mu_torch(self, t, X, Y):  # M x 1, M x D, M x 1, M x D
        # return torch.zeros([self.M, self.D])  # M x D
        return torch.ones([self.M, self.D]) * torch.tensor(self.mu)

    def sigma_torch(self, t, X, Y):  # M x 1, M x D, M x 1
        # print(X.shape)
        d = self.D

        L = torch.linalg.cholesky(self.var_cov_mat)
        L = L.float()

        res = torch.zeros([self.M,self.D,self.D])
        for j in range(self.M):
            res[j,:,:]  = L
        return  res  # M x D x D

    def net_u_Du(self, t, X):  # M x 1, M x D


        inputs = torch.cat([t, X], dim=1) #M x (D+1）


        u = self.fn_u(inputs) # M x 1

        DuDt = torch.autograd.grad(torch.sum(u), t, retain_graph=True,create_graph=True)[0]  # M x 1
        DuDx = torch.autograd.grad(torch.sum(u), X, retain_graph=True,create_graph=True)[0]  # M x D


        D2uDx2_list = []

        for i in range(self.M):
            hessian = torch.autograd.functional.hessian(lambda x: self.fn_u(x)[0], inputs[i,:],create_graph=True)
            D2uDx2_list.append(hessian[1:,1:]) # note that this hessian include time, we exclude the time part here

        D2uDx2 = torch.stack(D2uDx2_list,dim =0)


        return u, DuDx,DuDt,D2uDx2 # M x 1, M x D, M x 1, M x D

    def Dg_torch(self, X):  # M x D
        return torch.autograd.grad(torch.sum(self.g_torch(X)), X, retain_graph=True)[0]  # M x D

    def fetch_minibatch(self):
        T = self.T
        M = self.M
        N = self.N
        D = self.D

        Dt = np.zeros((M, N + 1, 1))  # M x (N+1) x 1
        DW = np.zeros((M, N + 1, D))  # M x (N+1) x D

        dt = T / N
        Dt[:, 1:, :] = dt
        DW[:, 1:, :] = np.sqrt(dt) * np.random.normal(size=(M, N, D))

        t = np.cumsum(Dt, axis=1)  # M x (N+1) x 1
        W = np.cumsum(DW, axis=1)  # M x (N+1) x D

        return torch.from_numpy(t).float(), torch.from_numpy(W).float()

    def loss_function(self, t, W, Xi):  # M x (N+1) x 1, M x (N+1) x D, 1 x D
        loss = torch.zeros(1)
        X_buffer = []
        Y_buffer = []

        t0 = t[:, 0, :]  # M x 1
        W0 = W[:, 0, :]  # M x D
        X0 = torch.cat([Xi] * self.M)  # M x D

        X0.requires_grad = True
        t0.requires_grad = True
        Y0, DuDx0,DuDt0,D2uDx20  = self.net_u_Du(t0, X0)  # M x 1, M x D

        X_buffer.append(X0)
        Y_buffer.append(Y0)

        X0.requires_grad = True

        t0.requires_grad = True

        for n in range(0, self.N):
            t1 = t[:, n + 1, :]
            W1 = W[:, n + 1, :]

            if self.gbm_scheme==0:
                X1 = X0 + self.mu_torch(t0, X0, Y0) * (t1 - t0)*X0 + torch.matmul(self.sigma_torch(t0, X0, Y0),
                                                                                   (W1 - W0).unsqueeze(
                                                                                       -1)).squeeze(2)*X0 # M x D
            elif self.gbm_scheme ==1:
                X1 = X0*torch.exp(self.mu_torch(t0, X0, Y0) * (t1 - t0) - 0.5*torch.tensor(self.sigma)**2*(t1-t0)+torch.matmul(self.sigma_torch(t0, X0, Y0),
                                                                                   (W1 - W0).unsqueeze(
                                                                                       -1)).squeeze(2)) # not sure if this is right


            Y1_tilde = Y0 + self.phi_torch(t0, X0,DuDt0,DuDx0,D2uDx20) * (t1 - t0) + torch.sum(
                Y0 * torch.matmul(self.sigma_torch(t0, X0, Y0), (W1 - W0).unsqueeze(-1)).squeeze(2), dim=1).unsqueeze(1)

        #    Y1, DuDx1,DuDt1,D2uDx21  = self.net_u_Du(t1, X1)
        #     loss = loss + torch.sum((Y1 - Y1_tilde) ** 2)
        #
            t0 = t1
            W0 = W1
            X0 = X1
            Y0 = Y1
        #
        #
            X_buffer.append(X0)
            Y_buffer.append(Y0)
        #
        #
        # loss = loss + torch.sum((Y1 - self.g_torch(X1)) ** 2)
        # loss = loss + torch.sum((Z1 - self.Dg_torch(X1)) ** 2)

        X = torch.stack(X_buffer, dim=1)  # M x N x D
        Y = torch.stack(Y_buffer, dim=1)  # M x N x 1

        return loss, X, Y, Y[0, 0, 0]

    def train(self, N_Iter=1):

        start_time = time.time()
        loss_list = []
        for it in range(N_Iter):

            t_batch, W_batch = self.fetch_minibatch()  # M x (N+1) x 1, M x (N+1) x D
            loss, X_pred, Y_pred, Y0_pred = self.loss_function(t_batch, W_batch, self.Xi)

            self.optimizer.zero_grad()
            loss.backward()
            loss_list.append(loss.item())
            self.optimizer.step()

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                print('It: %d, Time: %.2f, Loss: %.3e, Y0: %.3f' %
                      (it, elapsed, loss, Y0_pred))
                start_time = time.time()

                plt.plot(np.log10(range(len(loss_list))), np.log10(loss_list))
                plt.show()

    def predict(self, Xi_star, t_star, W_star):
        _, X_star, Y_star, _ = self.loss_function(t_star, W_star, Xi_star)

        return X_star, Y_star


if __name__ == '__main__':
    D = 10  # no. of underlyings
    S0 = np.ones([10])  # Initial Value of Assets at first of february
    K = 1  # Strike Price
    r = 0.02
    mu = [0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.05, 0.06, 0.07, 0.06]
    sigma = [.10, .11, .12, .13, .14, .14, .13, .12, .11, .10]
    rho = 0.1  # Correlation between Brownian Motions
    T = 1  # Time to Maturity
    N_STEPS, N_PATHS = 1, 3
    var_cov_mat = np.zeros((D, D))
    M = N_PATHS  # Number of paths
    N = N_STEPS  # Number of time steps
    alpha = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]



    for i in range(D):
        for j in range(D):
            if i == j:
                var_cov_mat[i, j] = sigma[i] * sigma[j]
            else:
                var_cov_mat[i, j] = rho * sigma[i] * sigma[j]

    learning_rate = 2e-3

    Xi = torch.from_numpy(np.array([1.0, 1.0] * int(D / 2))[None, :]).float()
    T = 1.0
    # print(Xi.shape)

#%%


    # M=5
    model = FBSNN(r, mu, sigma, rho,K,alpha, Xi, T, M, N, D, learning_rate)

    # check GBM results
    t_test, W_test = model.fetch_minibatch()

    loss, X, Y, Z = model.loss_function(t_test, W_test,Xi)
    payoff = model.g_torch(X[:,-1,:]) * math.exp(-r*T)
    print(payoff.mean())




#%%


    # model.train(N_Iter=1)


    # t_test, W_test = model.fetch_minibatch()
    #
    # X_pred, Y_pred = model.predict(Xi, t_test, W_test)
    #
    #




#%%
    # def u_exact(t, X):  # (N+1) x 1, (N+1) x D
    #     r = 0.05
    #     sigma_max = 0.4
    #     return np.exp((r + sigma_max ** 2) * (T - t)) * np.sum(X ** 2, 1, keepdims=True)  # (N+1) x 1
    #
    #
    # t_test = t_test.detach().numpy()
    # X_pred = X_pred.detach().numpy()
    # Y_pred = Y_pred.detach().numpy()
    # Y_test = np.reshape(u_exact(np.reshape(t_test[0:M, :, :], [-1, 1]), np.reshape(X_pred[0:M, :, :], [-1, D])),
    #                     [M, -1, 1])
    # print(Y_test[0, 0, 0])
    #
    # samples = 5

    # %%

    # %%
    # plt.figure()
    # plt.plot(t_test[0:1, :, 0].T, Y_pred[0:1, :, 0].T, 'b', label=r'Learned $u(t,X_t)$')
    # plt.plot(t_test[0:1, :, 0].T, Y_test[0:1, :, 0].T, 'r--', label=r'Exact $u(t,X_t)$')
    # plt.plot(t_test[0:1, -1, 0], Y_test[0:1, -1, 0], 'ko', label=r'$Y_T = u(T,X_T)$')
    #
    # plt.plot(t_test[1:samples, :, 0].T, Y_pred[1:samples, :, 0].T, 'b')
    # plt.plot(t_test[1:samples, :, 0].T, Y_test[1:samples, :, 0].T, 'r--')
    # plt.plot(t_test[1:samples, -1, 0], Y_test[1:samples, -1, 0], 'ko')
    # plt.plot([0], Y_test[0, 0, 0], 'ks', label=r'$Y_0 = u(0,X_0)$')
    # plt.xlabel(r'$t$')
    # plt.ylabel(r'$Y_t = u(t,X_t)$')
    # plt.title('100-dimensional Black-Scholes-Barenblatt')
    # plt.legend()
    #
    # # savefig('BSB.png', crop=False)
    # plt.show()
    #
    # errors = np.sqrt((Y_test - Y_pred) ** 2 / Y_test ** 2)
    # mean_errors = np.mean(errors, 0)
    # std_errors = np.std(errors, 0)
    #
    # plt.figure()
    # plt.plot(t_test[0, :, 0], mean_errors, 'b', label='mean')
    # plt.plot(t_test[0, :, 0], mean_errors + 2 * std_errors, 'r--', label='mean + two standard deviations')
    # plt.xlabel(r'$t$')
    # plt.ylabel('relative error')
    # plt.title('100-dimensional Black-Scholes-Barenblatt')
    # plt.legend()

    # savefig('BSB_error.png', crop=False)
