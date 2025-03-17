import torch

##########################################################################################################
############### Initialization of Cores ##################################################################
##########################################################################################################
def data_generator(shape, tt_rank, orthogonal =False): #CHECKED
    d = len(shape)  # Number of dimensions
    cores = []
    
    for k in range(d):
        # Shape of the k-th core
        if k == 0:
            core_shape = (shape[k], tt_rank[0])
        elif k == d-1:
            core_shape = (tt_rank[-1], shape[k])
        else:
            core_shape = (tt_rank[k-1], shape[k], tt_rank[k])
        
        # Generate random core
        core = torch.randn(core_shape)
        
        # Enforce left-orthogonality via QR decomposition (except last core)
        if orthogonal and k < d-1:
            if k == 0:
                # First core: shape (n_1, r_1)
                Q, _ = torch.linalg.qr(core, mode='reduced')
                core = Q
            else:
                # Middle cores: reshape to (r_{k-1}*n_k, r_k), then QR
                core_reshaped = core.reshape(-1, core_shape[-1])
                Q, _ = torch.linalg.qr(core_reshaped, mode='reduced')
                core = Q.reshape(core_shape)
        
        cores.append(core)
    
    return cores



##########################################################################################################
############### Tensor Train Decomposition ###############################################################
##########################################################################################################
def TTSVD(T,tt_rank): #CHECKED
    # r= [r_1,r_2,...,r_{m-1}]
    T_left = torch.ones(1)
    d = T.shape
    N = len(d)
    T_core = []

    d_prod = d[0]

    for i in range(0,N-1):
        T_i = T.reshape(d_prod,-1)
        temp = torch.kron(T_left,torch.eye(d[i]))
        [U,S,V] = torch.linalg.svd(temp.T@T_i,full_matrices=False)
        L = U[:,:tt_rank[i]]
        if i == 0:
            T_core.append(L.reshape(d[i],tt_rank[i]))
        else:
            T_core.append(L.reshape(tt_rank[i-1],d[i],-1))
        T_left = temp@L
        d_prod = d_prod*d[i+1]
    
    T_m_1 = T.reshape(int(d_prod/d[N-1]),-1)
    T_core.append(T_left.T@T_m_1)

    return T_core

def TT_to_tensor(r,core): #CHECKED
    tt_rank = [1] +r + [1]
    
    ##1. Initialize the tensor
    N = len(core)
    I = []
    I.append( int(core[0].shape[0]))
    for i in range(1,N):
        I.append(int(core[i].shape[1]))

    T_hat = torch.eye(tt_rank[-1])

    ##2. Construct the tensor by reversing the N steps SVD process
    for i in range(N):
        T_hat =  core[-(i+1)].reshape(-1,tt_rank[-(i+1)])@T_hat
        T_hat = T_hat.reshape(tt_rank[-(i+2)],-1)
    
    ##3. Reshape the tensor to the original size
    T_hat = T_hat.reshape(I)
   
    return T_hat

def TT_add(core1,core2):
    d = len(core1)
    cores = []
    for i in range(d):
        if i == 0:
            cores.append(torch.cat((core1[i], core2[i]), dim=1))
        elif i == d-1:
            cores.append(torch.cat((core1[i], core2[i]), dim=0))
        else:
            core = torch.zeros((core1[i].shape[0] + core2[i].shape[0], core1[i].shape[1], core1[i].shape[-1] + core2[i].shape[-1]))
            for j in range(core1[i].shape[1]):
                core[:,j,:] = torch.block_diag(core1[i][:,j,:], core2[i][:,j,:]) 
            cores.append(core)
    return cores

def TT_delta_to_tensor(cores,deltas):

    d = len(cores)

    # left orthogonalization U_1,...,U_{d-1},S_d
    U_cores = mu_orthogonalization(cores, d)
    # right orthogonalization S_1,V_2,...,V_{d}
    V_cores = mu_orthogonalization(cores, 0)

    tangent_cores = []

    for i in range(d):
        if i == 0:
            tangent_cores.append(torch.cat((deltas[i],U_cores[i]), dim=1))
        elif i == d-1:
            tangent_cores.append(torch.cat((V_cores[i],deltas[i]), dim=0))
        else:
            core = torch.zeros((V_cores[i].shape[0] + deltas[i].shape[0], U_cores[i].shape[1], deltas[i].shape[-1] + U_cores[i].shape[-1]))
            for j in range(U_cores[i].shape[1]):
                up = torch.cat((V_cores[i][:,j,:],torch.zeros(V_cores[i].shape[0],U_cores[i].shape[-1])), dim=1)
                down = torch.cat((deltas[i][:,j,:],U_cores[i][:,j,:]), dim=1)
                core[:,j,:] = torch.cat((up,down), dim=0)
            tangent_cores.append(core)

    return tangent_cores

def TT_rounding(cores, rank_max):
    d = len(cores)
    round_cores = mu_orthogonalization(cores,0)

    for i in range(d-1):
        [U,S,V] = torch.linalg.svd(round_cores[i].reshape(-1,round_cores[i].shape[-1]),full_matrices=False)

        if i == 0:
            round_cores[i] = U[:,:rank_max].reshape(round_cores[i].shape[0],rank_max)
        else:
            round_cores[i] = U[:,:rank_max].reshape(round_cores[i].shape[0],round_cores[i].shape[1],rank_max)

        round_cores[i+1] = torch.tensordot(torch.matmul(torch.diag(S[:rank_max]),V[:rank_max,:]),round_cores[i+1],dims=([1],[0]))
    return round_cores

def TT_rounding_2(cores, rank_max):
    d = len(cores)
    round_cores = mu_orthogonalization(cores,d)

    for i in range(d-1,0,-1):
        [U,S,V] = torch.linalg.svd(round_cores[i].reshape(round_cores[i].shape[0],-1),full_matrices=False)
        
        if i == d-1:
            round_cores[i] = V[:rank_max,:].reshape(rank_max, round_cores[i].shape[-1])
        else:
            round_cores[i] = V[:rank_max,:].reshape(rank_max,round_cores[i].shape[1],round_cores[i].shape[-1])
    
        round_cores[i-1] = torch.tensordot(round_cores[i-1],torch.matmul(U[:,:rank_max],torch.diag(S[:rank_max])),dims=([-1],[0]))
    return round_cores

##########################################################################################################
############### mu-orthogonalizaiton #####################################################################
##########################################################################################################
def mu_orthogonalization(cores, mu):
    mu_cores = [c.clone() for c in cores]
    d = len(mu_cores)


    # left orthogonalization U_1,...,U_{mu-1}
    for k in range(mu-1):
        core = mu_cores[k]
        if k == 0:
            # First core: shape (n_1, r_1)
            Q, R = torch.linalg.qr(core, mode='reduced')
        else:
            # Middle cores: reshape to (r_{k-1}*n_k, r_k), then QR
            core_reshaped = core.reshape(-1, core.shape[-1])
            Q, R = torch.linalg.qr(core_reshaped, mode='reduced')

        mu_cores[k] = Q.reshape(core.shape)
        mu_cores[k+1] = torch.tensordot(R, mu_cores[k+1], dims=([1], [0]))


    
    # right orthogonalization V_d,...,V_{mu+1}
    for k in range(d-1, mu, -1):
        core = mu_cores[k]
        if k == d-1:
            # Last core: shape (r_{d-1}, n_d)
            Q, R = torch.linalg.qr(core.T,mode = 'reduced')
        else:
            # Middle cores: reshape to (r_{k-1}, n_k*r_k), then QR
            core_reshaped = core.reshape(core.shape[0], -1)
            Q, R = torch.linalg.qr(core_reshaped.T,mode = 'reduced')

        mu_cores[k] = (Q.T).reshape(core.shape)
        mu_cores[k-1] = torch.tensordot(mu_cores[k-1], R, dims=([-1], [1]))

    return mu_cores


