import torch
import numpy as np
import pdb

def norm_group_nuclear(x: torch.Tensor, nbr_h: int) -> float:
    '''
    Method
    ======
    Compute the group nuclear norm defined as
    ||M||_{G, 1, S(1)} = \sum_{g\in G}{||M[g]||_{S(1)}},
    where the group are a partition of the image into squares of size
    (nbr_h, nbr_h).
    Input
    =====
        x: the RGB image frmo which to compute the norm.
        nbr_h: size of the squares.
    '''
    (B, C, H, W) = x.shape
    x = x.clone().detach()  # to copy the tensor?
    assert H == W, pdb.set_trace()
    norm = 0
    for idb in range(B):
      for i in range(int(H/nbr_h)):
        for j in range(int(H/nbr_h)):
            if C > 1:
               # compute the largest singular value for a given square..
               x_inter = torch.cat((x[idb, 0, i*nbr_h:i*nbr_h + nbr_h,
                                   j*nbr_h:j*nbr_h + nbr_h],
                                 x[idb, 1, i*nbr_h:i*nbr_h + nbr_h,
                                   j*nbr_h:j*nbr_h + nbr_h],
                                 x[idb, 2, i*nbr_h:i*nbr_h + nbr_h,
                                   j*nbr_h:j*nbr_h + nbr_h]), 0)
            elif C == 1:
               x_inter = x[idb, 0, i*nbr_h:i*nbr_h + nbr_h,
                                   j*nbr_h:j*nbr_h + nbr_h]
       
            #assert x_inter.shape == (3*nbr_h, nbr_h), pdb.set_trace()
            U, S, V = torch.svd(x_inter)
            norm += torch.sum(S)
    norm = norm.item() * 1.0 / B
    assert type(norm) == float, pdb.set_trace()
    return(norm)

def _LP_group_nuclear(D: torch.Tensor,
                      radius: float,
                      nbr_h: int,
                      type_subsampling=None):
    '''
    INPUT
    =====
        nbr_h: nbr of horizontal and vertical division of images.
        radius: radius of the group-nuclear norm.
        D: tensor of dimension (C, H, W) or (1, C, H, W)
    Method
    ======
    Compute the LMO for the group-nuclear norm defines as
    \sum_{g\in G} ||M[g]||_{S(1)},
    where the G are square division of the image. The LMO(M) is given by
    D[g] = 0 is g\neq g^*
    D[g^*] = \rho U[:, 0].dot(V[:, 0]) where M[g^*] = U S V^T and
    g^* = \argmax_{g\in G} ||M[g]||_{S(infty)} = \argmax_{g\in G} largest_sing_val(M[g])
    '''
    assert type(D) == torch.Tensor
    
    (B, C, H, W) = D.shape
    assert H == W, pdb.set_trace()
    # Make sure that nbr_h is a divider of nbr_H
    assert int(H/nbr_h)*nbr_h == H, pdb.set_trace()
    # Initialize v_FW at zero.
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')
    

    def _largest_singular_value(D_sub: torch.Tensor) -> float:
        assert D_sub.ndim == 2, pdb.set_trace()
        U, S, V = torch.svd(D_sub)
        max_val = S[0].item()
        assert type(max_val) == float, pdb.set_trace()
        return(max_val)

    # TODO: explain how we deal with the multi-channel
    list_group = []
    list_coo = []
    for idb in range(B):
      max_val = -1e-10
      for i in range(int(H/nbr_h)):
         for j in range(int(H/nbr_h)):
            # compute the largest singular value for a given square..
            if C > 1:
               D_inter = torch.cat((D[idb, 0, i*nbr_h:i*nbr_h + nbr_h,
                                j*nbr_h:j*nbr_h + nbr_h],
                              D[idb, 1, i*nbr_h:i*nbr_h + nbr_h,
                                j*nbr_h:j*nbr_h + nbr_h],
                              D[idb, 2, i*nbr_h:i*nbr_h + nbr_h,
                                j*nbr_h:j*nbr_h + nbr_h]),
                             0)
            else:
               D_inter = D[idb, 0, i*nbr_h:i*nbr_h + nbr_h,
                                j*nbr_h:j*nbr_h + nbr_h]
            #assert D_inter.shape == (3*nbr_h, nbr_h), pdb.set_trace()
            U, S, V = torch.svd(D_inter.cuda())
            if max_val < S[0].item():
               max_val = S[0].item()  
               i_star = i
               j_star = j
               U_star, V_star =  U[:, 0], V[:, 0]            
      v_inter = radius * (torch.mm(U_star.view(len(U_star), 1),
                                 V_star.view(1, len(V_star))))
      #U_after, S_after, V_after = torch.svd(v_inter)
      v_inter = v_inter.view((-1, nbr_h, nbr_h))
               # compute v_FW
      v_FW[idb, :, i_star*nbr_h:i_star*nbr_h + nbr_h,
                    j_star*nbr_h:j_star*nbr_h + nbr_h] = v_inter
      
    # check the norm ?!
    assert np.abs(norm_group_nuclear(v_FW, nbr_h) - radius) < 1e-3, pdb.set_trace()
    return(v_FW.cuda())
    #return (i_star, j_star), v_inter, v_FW

def norm_group_nuclear_2(x: torch.Tensor, nbr_h: int) -> float:
    '''
    Method
    ======
    Compute the group nuclear norm defined as
    ||M||_{G, 1, S(1)} = \sum_{g\in G}{||M[g]||_{S(1)}},
    where the group are a partition of the image into squares of size
    (nbr_h, nbr_h).
    Input
    =====
        x: the RGB image frmo which to compute the norm.
        nbr_h: size of the squares.
    '''
    (C, H, W) = x.shape
    x = x.clone().detach()  # to copy the tensor?
    assert H == W, pdb.set_trace()
    norm = 0
    for i in range(int(H/nbr_h)):
        for j in range(int(H/nbr_h)):
            # compute the largest singular value for a given square..
            x_inter = x[0, i*nbr_h:i*nbr_h + nbr_h,
                                   j*nbr_h:j*nbr_h + nbr_h]
            assert x_inter.shape == (1*nbr_h, nbr_h), pdb.set_trace()
            U, S, V = torch.svd(x_inter)
            norm += torch.sum(S)
    norm = norm.item()
    assert type(norm) == float, pdb.set_trace()
    return(norm)

def _LP_group_nuclear_2(D: torch.Tensor,
                      radius: float,
                      nbr_h: int,
                      type_subsampling=None):
    '''
    INPUT
    =====
        nbr_h: nbr of horizontal and vertical division of images.
        radius: radius of the group-nuclear norm.
        D: tensor of dimension (C, H, W) or (1, C, H, W)
    Method
    ======
    Compute the LMO for the group-nuclear norm defines as
    \sum_{g\in G} ||M[g]||_{S(1)},
    where the G are square division of the image. The LMO(M) is given by
    D[g] = 0 is g\neq g^*
    D[g^*] = \rho U[:, 0].dot(V[:, 0]) where M[g^*] = U S V^T and
    g^* = \argmax_{g\in G} ||M[g]||_{S(infty)} = \argmax_{g\in G} largest_sing_val(M[g])
    '''
    assert type(D) == torch.Tensor
    if D.ndim == 4:
        assert D.shape[0] == 1, pdb.set_trace()
        D = D.squeeze(0)
    (C, H, W) = D.shape
    assert H == W, pdb.set_trace()
    # Make sure that nbr_h is a divider of nbr_H
    assert int(H/nbr_h)*nbr_h == H, pdb.set_trace()
    # Initialize v_FW at zero.
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')

    def _largest_singular_value(D_sub: torch.Tensor) -> float:
        assert D_sub.ndim == 2, pdb.set_trace()
        U, S, V = torch.svd(D_sub)
        max_val = S[0].item()
        assert type(max_val) == float, pdb.set_trace()
        return(max_val)

    # TODO: explain how we deal with the multi-channel
    list_group = []
    list_coo = []
    for i in range(int(H/nbr_h)):
        for j in range(int(H/nbr_h)):
            # compute the largest singular value for a given square..
            D_inter = D[0, i*nbr_h:i*nbr_h + nbr_h,
                                   j*nbr_h:j*nbr_h + nbr_h]
            assert D_inter.shape == (1*nbr_h, nbr_h), pdb.set_trace()
            list_group.append(_largest_singular_value(D_inter))
            list_coo.append((i, j))

    # find the largest value.
    indecies_1,  v_inter_1, v_FW_1 = _LP_group_nuclear_1(D.unsqueeze(0), radius, nbr_h)
    idx = np.argmax(list_group)
    #print("max: ", max(list_group), "max_1: ", max_val_1)
    (i_star, j_star) = list_coo[idx]
    D_inter_star = D[0, i_star*nbr_h:i_star*nbr_h + nbr_h,
                                j_star*nbr_h:j_star*nbr_h + nbr_h]
    U_star, _, V_star = torch.svd(D_inter_star)
    v_inter = radius * (torch.mm(U_star[:, 0].view(len(U_star[:, 0]), 1),
                                 V_star[:, 0].view(1, len(V_star[0, :]))))
    U_after, S_after, V_after = torch.svd(v_inter)
    print("equal: ", torch.all(v_inter_1.eq(v_inter)))
    print("index: ", indecies_1, "idx: ", i_star, "jdx: ", j_star)
    v_inter = v_inter.view((1, nbr_h, nbr_h))
    # compute v_FW
    v_FW[:, i_star*nbr_h:i_star*nbr_h + nbr_h,
         j_star*nbr_h:j_star*nbr_h + nbr_h] = v_inter
    print("FW_1", v_FW_1.shape, "FW: ", v_FW.shape, "equal FW: ", torch.all(v_FW_1.squeeze(0).eq(v_FW)))
    # check the norm ?!
    assert np.abs(norm_group_nuclear(v_FW, nbr_h) - radius) < 1e-3, pdb.set_trace()
    return(v_FW)
