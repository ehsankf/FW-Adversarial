import torch

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
            D_inter = D.view((B, C*H, W))
            U, diag, V = torch.svd(D_inter)
            q = p/(p-1)
            schatten_q_q_1 = torch.sum(torch.abs(diag)**q, axis=1)**((q-1)/q)

            def _make_diag(diag, B, q):
                # because in list comprehension the scope is modified...
                diag_inter = torch.cat([torch.diag((diag**(q-1))[i, :]).unsqueeze(0) for i in range(B)], axis=0)
                return(diag_inter)

            diag_inter = _make_diag(diag, B, q)
            v_FW = (radius/schatten_q_q_1.view((B, 1, 1)) * torch.bmm(torch.bmm(U, diag_inter),
                                                                      V.permute(0, 2, 1))).view((B, C, H, W))
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
            D_inter = D.view((B, C*H, W))
            U, diag, V = torch.svd(D_inter)
            v_FW = torch.bmm(U, V.permute(0, 2, 1)).view((B, C, H, W))
    else:
        print('nuclear norm is treated with a specific function.')
    return(v_FW)
