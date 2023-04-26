import numpy as np
import pdb
import torch


def _LP_lp(D: np.ndarray,
           radius: float,
           p: float,
           type_subsampling=None,
           proba_subsampling=0.5) -> np.ndarray:
    '''
    INPUT:
    D is typically a (C, H, W) matrix where C is the number of channel.
    p : float . If p < 1, encode p=+\infty.
    METHOD:
    computes v^{FW} = argmax_{||H|| \leq self.radius} Tr(H^T D)=<H, D>
    '''
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')

    def unravel_index(index, shape):
        # torch version of np.unravel_index()
        out = []
        for dim in reversed(shape):
            out.append((index % dim).item())
            index = index // dim
        return tuple(reversed(out))

    if np.abs(p - 1) < 1e-6:
        if type_subsampling is None:
            (c_max, i_max, j_max) = unravel_index(torch.abs(D).argmax(), D.shape)
            # (c_max, i_max, j_max) = np.unravel_index(np.abs(D).argmax(), D.shape)
            v_FW[c_max, i_max, j_max] = np.sign(D[c_max, i_max, j_max]) * radius
        elif type_subsampling == 'channel':
            c = np.random.randint(0, 3)
            (i_max, j_max) = unravel_index(torch.abs(D[c, :, :]).argmax(), D.shape[1:3])
            v_FW[c, i_max, j_max] = np.sign(D[c, i_max, j_max]) * radius
        else:
            raise ValueError('not taking into account this type of subsampling.')
        return(v_FW)
    elif p > 1:
        if type_subsampling is not None:
            raise ValueError('very little need for subsampling when p>1.')
        q = p/(p - 1)
        idx_non_zero = torch.where(D != 0)
        norm_q = torch.sum(torch.abs(D[idx_non_zero])**q)**(1/q)
        assert norm_q > 0, pdb.set_trace()
        v_FW[idx_non_zero] = radius/norm_q**(q-1) * torch.sign(D[idx_non_zero]) * torch.abs(D[idx_non_zero])**(q-1)
        return(v_FW)
    else:
        # Then it is over the infty norm.
        if type_subsampling is None:
            # v_FW = sign(D) * radius
            v_FW = torch.sign(D) * radius
        elif type_subsampling == 'channel':
            c = np.random.randint(0, 3)
            v_FW[c, :, :] = torch.sign(D[c, :, :]) * radius
        return(v_FW)


def _LP_lp_gpu(D: torch.Tensor,
               radius: float,
               p: float,
               type_subsampling=None,
               proba_subsampling=0.5) -> torch.Tensor:
    '''
    INPUT:
    D is typically a (B, C, H, W) matrix where C is the number of channel.
    METHOD:
    computes v^{FW} = argmax_{||H|| \leq self.radius} Tr(H^T D)=<H, D>
    '''
    assert type(D) == torch.Tensor and D.ndim == 4
    (B, C, H, W) = D.shape
    # initialize v_FW..
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')

    def unravel_index(index: float, shape: tuple) -> tuple:
        # torch version of np.unravel_index()
        out = []
        for dim in reversed(shape):
            out.append((index % dim))
            index = index // dim
        return tuple(reversed(out))

    if np.abs(p - 1) < 1e-6:
        if type_subsampling is None:
            l_idx_max = torch.abs(D).view(B, -1).argmax(1)
            # coo = [unravel_index(idx, (C, H, W)) for idx in l_idx_max]
            # [(i, *(unravel_index(l_idx_max[i], (C, H, W)))) for i in range(len(l_idx_max))]
            # (c_max, i_max, j_max) = unravel_index(torch.abs(D).argmax(dim=0), D.shape)
            # (c_max, i_max, j_max) = np.unravel_index(np.abs(D).argmax(), D.shape)
            D_flatten = torch.abs(D).view(B, -1)
            v_FW.view(B, -1)[torch.arange(B),
                             l_idx_max] = radius * torch.sign(D_flatten[torch.arange(B),
                                                                        l_idx_max])
        elif type_subsampling == 'channel':
            c = np.random.randint(0, 3)
            l_idx_max = torch.abs(D[:, c, :, :]).view(B, -1).argmax(1)
            D_flatten = torch.abs(D[:, c, :, :]).view(B, -1)
            v_FW[:, c, :, :].view(B, -1)[torch.arange(B),
                                         l_idx_max] = radius * torch.sign(D_flatten[torch.arange(B),
                                                                                    l_idx_max])
        else:
            raise ValueError('not taking into account this type of subsampling.')
        return(v_FW)
    elif p > 1:
        # TODO: make it batch?
        if type_subsampling is not None:
            raise ValueError('very little need for subsampling when p>1.')
        q = p/(p - 1)
        idx_non_zero = torch.where(D != 0)
        norm_q = torch.sum((torch.abs(D[idx_non_zero])**q).view(B, -1), axis=1)**(1/q)
        assert torch.all(norm_q > 0), pdb.set_trace()
        v_FW[idx_non_zero] = radius * torch.sign(D[idx_non_zero]) * torch.abs(D[idx_non_zero])**(q-1)
        v_FW = v_FW / (norm_q**(q-1)).view((B, 1, 1, 1))
        return(v_FW)
    else:
        if type_subsampling is None:
            # v_FW = sign(D) * radius
            v_FW = torch.sign(D) * radius
        elif type_subsampling == 'channel':
            c = np.random.randint(0, 3)
            v_FW[:, c, :, :] = torch.sign(D[:, c, :, :]) * radius
        return(v_FW)


def _LP_group_lasso(D: np.ndarray,
                    radius: float,
                    mask=False,
                    size_groups=4,
                    type_subsampling=None,
                    proba_subsampling=0.5) -> np.ndarray:
    '''
    mask (bool): if true then the group lasso is with respect to (4, 4) matrices.
    '''
    # TODO: make it in torch...
    # assert inputs
    if type_subsampling is not None:
        assert type_subsampling in ['channel', 'group', 'channel_group']
    assert proba_subsampling > 0 and proba_subsampling <= 1
    # group lasso with $D_{1,2} = \sum_{j=1}^{d}{||D[:,j]||_2}$
    # should be done with respect to others partitions...
    v_FW = np.zeros(D.shape, dtype=np.float32)
    (C, H, W) = D.shape
    assert C == 1 or C == 3, print('an image has either 1 or 3 channels')
    if not mask:
        # Here each group is a row of the image.
        # TODO: make the code faster
        if type_subsampling is None:
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
        elif type_subsampling == 'channel':
            # find the best row in a randomly chosen channel
            c_max = np.random.randint(0, C)
            norm_2_groups = [np.sqrt(np.sum(D[c_max, i, :]**2)) for i in range(D.shape[1])]
            i_max = np.argmax(norm_2_groups)
            assert norm_2_groups[i_max] > 0, pdb.set_trace()
            norm = norm_2_groups[i_max]
        else:
            raise ValueError('there is an issue in the choice of the inputs.')
        v_FW[c_max, i_max, :] = radius/norm * D[c_max, i_max, :]
        assert np.abs(np.sqrt(np.sum(v_FW**2)) - radius) < 1e-3, pdb.set_trace()
    else:
        # split the image in squares of size (size_groups, size_groups)
        nbr_H = int(H/size_groups) - 1
        sg = size_groups
        # TODO: rename channel subsampling
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
                      type_subsampling=None,
                      proba_subsampling=0.5) -> np.ndarray:
    '''
    Implementing l_1_infty..
    '''
    # TODO: do it in torch..
    v_FW = np.zeros(D.shape, dtype=np.float32)
    return(v_FW)


def _LP_group_norm_saliency(D: torch.Tensor,
                            radius: float):
    '''
    experimenting with group saliency..
    '''
    assert type(D) == torch.Tensor and D.ndim == 3
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')
    
    return(v_FW)


def _LP_nuclear_gpu(D: torch.Tensor,
                    radius: float,
                    type_subsampling=None,
                    batch=1) -> torch.Tensor:
    '''
    Doing LMO for nuclear norms with a function that can be implemented in
    GPU.
    '''
    # TODO: allow for non-channel subsampling.
    assert type(D) == torch.Tensor
    if D.ndim == 3:
        (C, H, W) = D.shape
        assert batch == 1
    else:
        (B, C, H, W) = D.shape
    # initialize v_FW..
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')  # to have float32
    if type_subsampling == 'channel':
        if D.ndim == 3:
            # TODO: remove that!! Should only be D.ndim == 4
            # use torch.symeig and take the first one only.
            c = np.random.randint(0, C)  # choose one channel
            U, _, V = torch.svd(D[c, :, :])
            v_FW[c, :, :] = radius * torch.mm(U[:, 0].view(len(U[:, 0]), 1),
                                              V[:, 0].view(1, len(V[0, :])))
        else:
            c = np.random.randint(0, C, B)  # choose one channel per image.
            # TODO: make that without list completion
            D_inter = torch.cat([D[i, c[i], :, :].unsqueeze(0) for i in range(B)], axis=0)
            U, _, V = torch.svd(D_inter)
            # TODO: remove the for loop.
            for i in range(B):
                v_FW[i, c[i], :, :] = radius * torch.mm(U[i, :, 0].view(len(U[i, :, 0]), 1),
                                                        V[i, :, 0].view(1, len(V[i, 0, :])))
    else:
        print('for nuclear type subsampling should be channel')
        raise ValueError('I have not treated non-channel subsampling for nuclear...')
    return(v_FW)


def _LP_Schatten(D: torch.Tensor,
                 radius: float,
                 p: float,
                 type_subsampling=None) -> torch.Tensor:
    '''
    Computing LMO for p-Schatten norm with a function that can be implemented in
    GPU.
    ||D||_{*,p} = ||sing_val(D)||_p
    The solution to the LMO(D) with p-Schatten norm and rho-radius of distortion is
    M = \rho /||D||_q^{q-1} (U diag^{q-1} V^T),
    where D = U diag V^T and 1/q + 1/p = 1 and p>1.
    '''
    assert p > 1 or p == -1
    assert type(D) == torch.Tensor and D.ndim == 3
    (C, H, W) = D.shape
    # initialize v_FW..
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')  # to have float32
    if p > 1:
        if type_subsampling == 'channel':
            # use torch.symeig and take the first one only.
            c = np.random.randint(0, C)  # choose one channel
            U, diag, V = torch.svd(D[c, :, :])
            q = p/(p-1)
            schatten_q_q_1 = torch.sum(torch.abs(diag)**q)**((q-1)/q)
            v_FW[c, :, :] = radius/schatten_q_q_1 * torch.mm(torch.mm(U, torch.diag(diag**(q-1))),
                                                             V.T)
        else:
            D_concat = torch.cat((D[0, :, :], D[1, :, :], D[2, :, :]), 0)
            U, diag, V = torch.svd(D_concat)
            q = p/(p-1)
            schatten_q_q_1 = torch.sum(torch.abs(diag)**q)**((q-1)/q)
            v_FW = (radius/schatten_q_q_1 * torch.mm(torch.mm(U, torch.diag(diag**(q-1))),
                                                     V.T)).view((3, H, W))
    elif p == -1:
        # corresponds to the infty-Schatten ball.
        if type_subsampling == 'channel':
            # use torch.symeig and take the first one only.
            c = np.random.randint(0, C)  # choose one channel
            U, diag, V = torch.svd(D[c, :, :])
            v_FW[c, :, :] = radius * torch.mm(U, V.T)
        else:
            # let's concatenate the RGB matrix and compute the SVD.
            D_concat = torch.cat((D[0, :, :], D[1, :, :], D[2, :, :]), 0)
            U, diag, V = torch.svd(D_concat)
            v_FW = (radius * torch.mm(U, V.T)).view((3, H, W))
    elif p == 1:
        # treat the case of the 1-Schatten norm -> nuclear norm
        print('This is the nuclear ball case, it should not be treated with this function')
    return(v_FW)


def _LP_Schatten_gpu(D: torch.Tensor,
                     radius: float,
                     p: float,
                     type_subsampling=None) -> torch.Tensor:
    '''
    batch version of LP Schatten gpu..
    '''
    assert p > 1 or p == -1
    assert type(D) == torch.Tensor and D.ndim == 4
    (B, C, H, W) = D.shape
    # initialize v_FW..
    v_FW = torch.zeros(D.shape).type('torch.FloatTensor')  # to have float32
    if p > 1:
        if type_subsampling == 'channel':
            c = np.random.randint(0, C, B)  # choose one channel per image.
            D_inter = torch.cat([D[i, c[i], :, :].unsqueeze(0) for i in range(B)], axis=0)
            U, diag, V = torch.svd(D_inter)
            q = p/(p-1)
            schatten_q_q_1 = torch.sum(torch.abs(diag)**q, axis=1)**((q-1)/q)
            assert len(schatten_q_q_1) == B, pdb.set_trace()
            # TODO: try to do without the for loop...
            for i in range(B):
                v_FW[i, c[i], :, :] = radius/schatten_q_q_1[i] * torch.mm(torch.mm(U[i, :, :], torch.diag(diag[i, :]**(q-1))),
                                                                          V[i, :, :].T)
        else:
            # should concatenate the (3, H, W) into (3*H, W)
            D_inter = D.view((B, 3*H, W))
            U, diag, V = torch.svd(D_inter)
            q = p/(p-1)
            schatten_q_q_1 = torch.sum(torch.abs(diag)**q, axis=1)**((q-1)/q)

            def _make_diag(diag, B, q):
                # because in list comprehension the scope is modified...
                diag_inter = torch.cat([torch.diag((diag**(q-1))[i, :]).unsqueeze(0) for i in range(B)], axis=0)
                return(diag_inter)

            diag_inter = _make_diag(diag, B, q)
            v_FW = (radius/schatten_q_q_1.view((B, 1, 1)) * torch.bmm(torch.bmm(U, diag_inter),
                                                                      V.permute(0, 2, 1))).view((B, 3, H, W))
    elif p == -1:
        if type_subsampling == 'channel':
            c = np.random.randint(0, C, B)  # choose one channel per image.
            D_inter = torch.cat([D[i, c[i], :, :].unsqueeze(0) for i in range(B)], axis=0)
            U, diag, V = torch.svd(D_inter)
            inter = torch.bmm(U, V.permute(0, 2, 1))
            # TODO: remove this for loop..
            for i in range(B):
                v_FW[i, c[i], :, :] = inter[i, :, :]
        else:
            D_inter = D.view((B, 3*H, W))
            U, diag, V = torch.svd(D_inter)
            v_FW = torch.bmm(U, V.permute(0, 2, 1)).view((B, 3, H, W))
    return(v_FW)


def LP(D: torch.Tensor,
       radius: float,
       type_ball: str,
       p=2,
       mask=True,
       type_subsampling=None,
       proba_subsampling=0.5) -> torch.Tensor:
    '''
    type_subsample: str. Either channel, or channel_group or group. For channel
    it performs the LMO over a randomly chosen channel. For group, it performs
    the LMO on a random subset of the vertices of the group norm. For
    channel_group it select one channel and some groups on the channel.
    Non-batch method here.
    '''
    # TODO: remove the mask possibility...
    # check inputs are coherent
    assert proba_subsampling <= 1 and proba_subsampling > 0
    if type_subsampling is not None:
        assert type_subsampling in ['channel', 'group', 'channel_group']
    # first transform D is a numpy array
    (B, C, H, W) = D.shape
    assert B == 1, print('treat example only one by one.')
    D = D.squeeze(axis=0)
    # solves v^{FW} = argmax_{||H|| \leq self.radius} Tr(H^T D)
    assert type_ball in ['lp', 'nuclear', 'group_lasso', 'schatten_p']
    if type_ball == 'lp':
        V_FW = _LP_lp(D, radius, p,
                      type_subsampling=type_subsampling,
                      proba_subsampling=proba_subsampling)
    elif type_ball == 'nuclear':
        V_FW = _LP_nuclear_gpu(D, radius,
                               type_subsampling=type_subsampling)
        # TODO: why do we need to unsqueeze here?
    elif type_ball == 'group_lasso':
        # TODO: remove the mask
        V_FW = _LP_group_lasso(D, radius,
                               mask=mask,
                               type_subsampling=type_subsampling,
                               proba_subsampling=proba_subsampling)
    elif type_ball == 'schatten_p':
        V_FW = _LP_Schatten(D, radius, p,
                            type_subsampling=type_subsampling)
    else:
        raise ValueError('ball type not understood')
    # V_FW = torch.tensor(V_FW).view((1, C, H, W))
    V_FW = V_FW.unsqueeze(0)
    # TODO: probably asser something about the style of the image?
    return(V_FW)


def LP_batch(D: torch.Tensor,
             radius: float,
             type_ball: str,
             p=2,
             mask=True,
             type_subsampling=None,
             proba_subsampling=0.5) -> torch.Tensor:
    if type_subsampling is not None:
        assert type_subsampling in ['channel', 'group', 'channel_group']
    if type_ball == 'nuclear':
        V_FW = _LP_nuclear_gpu(D, radius,
                               type_subsampling=type_subsampling)
        # TODO: do we need to add proba_subsampling?
    elif type_ball == 'schatten_p':
        V_FW = _LP_Schatten_gpu(D, radius, p,
                                type_subsampling=type_subsampling)
    elif type_ball == 'lp':
        V_FW = _LP_lp_gpu(D, radius,
                          p,
                          type_subsampling=type_subsampling,
                          proba_subsampling=proba_subsampling)
        # TODO do something else for LP_batch..
    else:
        raise ValueError('others that nuclear is not yet taken into account')
    return(V_FW)
