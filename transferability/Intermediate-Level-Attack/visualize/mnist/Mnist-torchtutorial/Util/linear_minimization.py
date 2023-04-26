import numpy as np
import torch
import pdb
from scipy.sparse.linalg import eigs


def firstSVD(M: np.ndarray) -> (np.ndarray, np.ndarray, float):
    '''
    Homemade function to compute the first column of U and V in the SVD
    decomposition of a matrix M = U D V^T. because np.linalg.svd() computes the
    whole decomposition and sklearn.decomposition.TruncatedSVD does not gives the
    right and left eigenvectors
    for M = U S V^T,
    MM^T = U D^2 U^T, hence +-eigenvector(MM^T) is the first column of U.
    M^TM = V D^2 V^T, hence +-eigenvector(M^TM) is the first column of V.
    Hence we first find the first eigenvector of M^TM (we choose arbritarily
    between +-) and then use the fact that
    MVs^{-1} = U. In particular U[:, 0] = M.(V[:,0]/s_1) where s_1 is the
    first eigenvalue
    '''
    # val1, u = eigs(M.dot(np.transpose(M)), k=1)
    # TODO: gerer le cas où la deformation n'est pas carrée..
    val, v = eigs(np.transpose(M).dot(M), k=1)
    v = np.real(v).squeeze()
    val = np.sqrt(np.real(val[0]))  # first singular value of M
    u = M.dot(v/val)
    return(u, v, val)


def _LP_lp(D: np.ndarray,
           radius: float,
           p: float) -> np.ndarray:
    '''
    D is typically a (C, H, W) matrix where C is the number of channel.
    '''
    # computes v^{FW} = argmax_{||H|| \leq self.radius} Tr(H^T D)=<H, D>
    v_FW = np.zeros(D.shape, dtype=np.float32)
    if np.abs(p - 1) < 1e-6:
        (c_max, i_max, j_max) = np.unravel_index(np.abs(D).argmax(), D.shape)
        v_FW[c_max, i_max, j_max] = np.sign(D[c_max, i_max, j_max]) * radius
        return(v_FW)
    elif p > 1:
        q = p/(p - 1)
        # TODO: might be a problem if some value of D are zero...
        idx_non_zero = np.where(D != 0)
        norm_q = np.sum(np.abs(D[idx_non_zero])**q)**(1/q)
        assert norm_q > 0, pdb.set_trace()
        v_FW[idx_non_zero] = radius/norm_q**(q-1) * np.sign(D[idx_non_zero]) * np.abs(D[idx_non_zero])**(q-1)
        return(v_FW)
    else:
        raise ValueError('p must be between 1 and + infty')
        # TODO: implementing l_infty also


def _LP_group_lasso(D: np.ndarray,
                    radius: float,
                    mask=False,
                    channel_subsampling=False,
                    size_groups=4,
                    group_subsampling=True) -> np.ndarray:
    '''
    mask (bool): if true then the group lasso is with respect to (4, 4) matrices.
    '''
    # group lasso with $D_{1,2} = \sum_{j=1}^{d}{||D[:,j]||_2}$
    # should be done with respect to others partitions...
    v_FW = np.zeros(D.shape, dtype=np.float32)
    (C, H, W) = D.shape
    assert C == 1 or C == 3, print('an image has either 1 or 3 channels')
    if not mask:
        # Here each group is a row of the image.
        # TODO: make the code faster
        if not channel_subsampling:
            # find the best row among all the RGB channel
            l_c = []
            # find the largest row per channel.
            for c in range(C):
                norm_2_groups = [np.sqrt(np.sum(D[c, i, :]**2)) for i in range(D.shape[1])]
                i_max_col = np.argmax(norm_2_groups)
                assert norm_2_groups[i_max_col] > 0, pdb.set_trace()
                l_c.append((i_max_col, norm_2_groups[i_max_col]))
            c_max = np.argmax([norm_2 for (i, norm_2) in l_c])
            i_max = l_c[c_max][0]
            norm = l_c[c_max][1]
        else:
            # find the best row in a randomly chosen channel
            c_max = np.random.randint(0, C)
            norm_2_groups = [np.sqrt(np.sum(D[c_max, i, :]**2)) for i in range(D.shape[1])]
            i_max = np.argmax(norm_2_groups)
            assert norm_2_groups[i_max] > 0, pdb.set_trace()
            norm = norm_2_groups[i_max]
        v_FW[c_max, i_max, :] = radius/norm * D[c_max, i_max, :]
        assert np.abs(np.sqrt(np.sum(v_FW**2)) - radius) < 1e-3, pdb.set_trace()
    else:
        # split the image in squares of size (size_groups, size_groups)
        nbr_H = int(H/size_groups) - 1
        sg = size_groups
        if channel_subsampling:
            # TODO: make general function on how to define the group,
            # it is only at a given point that we use the square norm
            c = np.random.randint(0, C)
            if not group_subsampling:
                # TODO: do more sophisticated subsampling approach.
                norm_2_groups = np.zeros((nbr_H, nbr_H), dtype=np.float32)
                for i in range(0, nbr_H):
                    for j in range(0, nbr_H):
                        norm_2_groups[i, j] = np.sqrt(np.sum(D[c, sg*i:sg*i+sg,
                                                               sg*j:sg*j+sg]**2))
                (i_max, j_max) = np.unravel_index(np.abs(norm_2_groups).argmax(),
                                                  norm_2_groups.shape)
                v_FW[c, sg*i_max:sg*i_max+sg,
                     sg*j_max:sg*j_max+sg] = radius/norm_2_groups[i_max, j_max]\
                    * D[c, sg*i_max:sg*i_max+sg, sg*j_max:sg*j_max+sg]
            else:
                # TODO: set 0.2 as a parameter.
                nbr_group = int(0.2*nbr_H)
                (i_start, j_start) = np.random.randint(0, nbr_H-1, 2)
                i_final = min(i_start + nbr_group, nbr_H)
                j_final = min(j_start + nbr_group, nbr_H)
                nbr_i = i_final - i_start
                nbr_j = j_final - j_start
                norm_2_groups = np.zeros((nbr_i, nbr_j), dtype=np.float32)
                for i in range(i_start, i_start + nbr_i):
                    for j in range(j_start, j_start + nbr_j):
                        norm_2_groups[i-i_start,
                                      j-j_start] = np.sqrt(np.sum(D[c, sg*i:sg*i+sg,
                                                           sg*j:sg*j+sg]**2))
                (i_max, j_max) = np.unravel_index(np.abs(norm_2_groups).argmax(),
                                                  norm_2_groups.shape)
                i_max = i_start + i_max
                j_max = j_start + j_max
                v_FW[c, sg*i_max:sg*i_max+sg,
                     sg*j_max:sg*j_max+sg] = radius/norm_2_groups[i_max - i_start, j_max - j_start]\
                    * D[c, sg*i_max:sg*i_max+sg, sg*j_max:sg*j_max+sg]
        assert np.abs(np.sqrt(np.sum(v_FW**2)) - radius) < 1e-3, pdb.set_trace()
    return(v_FW)


def _LP_group_1_infty(D: np.ndarray,
                      radius: float,
                      mask=False,
                      channel_subsampling=False) -> np.ndarray:
    '''
    Implementing l_1_infty..
    '''
    v_FW = np.zeros(D.shape, dtype=np.float32)
    return(v_FW)


def _LP_nuclear(D: np.ndarray,
                radius: float,
                channel_subsampling=True) -> np.ndarray:
    v_FW = np.zeros(D.shape, dtype=np.float32)
    (C, H, W) = D.shape
    if channel_subsampling:
        c = np.random.randint(0, C)
        (u, v, val) = firstSVD(D[c, :, :])
        n = len(u)
        v_FW[c, :, :] = radius * np.reshape(u, (n, 1)).dot(np.reshape(v, (1, n)))
    else:
        raise ValueError('I have not treated this case yet...')
    return(v_FW)


def LP(D: torch.tensor,
       radius: float,
       type_ball: str,
       p=2,
       mask=True,
       channel_subsampling=False) -> torch.tensor:
    # first transform D is a numpy array
    (B, C, H, W) = D.shape
    assert B == 1, print('treat example only one by one.')
    D = D.squeeze(axis=0).numpy()
    # solves v^{FW} = argmax_{||H|| \leq self.radius} Tr(H^T D)
    assert type_ball in ['lp', 'nuclear', 'group_lasso']
    if type_ball == 'lp':
        V_FW = _LP_lp(D, radius, p)
        # TODO: add channel subsampling..
    elif type_ball == 'nuclear':
        V_FW = _LP_nuclear(D, radius,
                           channel_subsampling=channel_subsampling)
    elif type_ball == 'group_lasso':
        V_FW = _LP_group_lasso(D, radius,
                               mask=mask,
                               channel_subsampling=channel_subsampling)
    else:
        raise ValueError('ball type not understood')
    V_FW = torch.tensor(V_FW).view((1, C, H, W))
    return(V_FW)
