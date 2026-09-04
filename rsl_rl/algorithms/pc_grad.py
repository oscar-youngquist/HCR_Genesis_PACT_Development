import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import copy
import random


class PCGrad():
    def __init__(self, optimizer, reduction='mean'):
        self._optim, self._reduction = optimizer, reduction
        return

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads, has_any_grad = self._pack_grad(objectives)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad, has_any_grad)
        return
    
    def pc_backward_pinn(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads, has_any_grad = self._pack_grad(objectives)
        pc_grad = self._project_conflicting_pinn(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad, has_any_grad)
        return
    
    def pc_backward_ppgrad(self, objectives):
        '''
        calculate the gradient of the parameters

        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads, has_any_grad = self._pack_grad(objectives)
        pc_grad = self._project_conflicting_pinn_balanced(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad, has_any_grad)
        return

    def _project_conflicting(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        # First, de-conflict all of the 
        for g_i in pc_grad:
            random.shuffle(grads)
            for g_j in grads:
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                           for g in pc_grad]).sum(dim=0)
        else: exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad


    # # Prioritized modification assuming the first gradient is from the "prime" task and the second is "auxilliary"
    # def _project_conflicting(self, grads, has_grads, shapes=None):
    #     shared = torch.stack(has_grads).prod(0).bool()
    #     pc_grad, num_task = copy.deepcopy(grads), len(grads)
    #     if len(pc_grad) > 1:
    #         g_prime = pc_grad[0]
    #         g_sub   = pc_grad[1]
    #         # We want to check if the sub-objective conflicts with the primairy, and needs to be projected
    #         g_s_g_p = torch.dot(g_sub, g_prime)

    #         if g_s_g_p < 0:
    #             g_sub -= (g_s_g_p) * g_prime / (g_prime.norm()**2)
    #             g_sub *= 0.5

    #     merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

    #     if self._reduction:
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                        for g in pc_grad]).mean(dim=0)
    #     elif self._reduction == 'sum':
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                        for g in pc_grad]).sum(dim=0)
    #     else: exit('invalid reduction method')

    #     merged_grad[~shared] = torch.stack([g[~shared]
    #                                         for g in pc_grad]).sum(dim=0)
    #     return merged_grad
    

    # # Prioritized modification assuming the first gradient is from the "prime" task and the second is "auxilliary"
    # def _project_conflicting_pinn(self, grads, has_grads, shapes=None):
    #     shared = torch.stack(has_grads).prod(0).bool()
    #     pc_grad, num_task = copy.deepcopy(grads), len(grads)

    #     # First, de-conflict all of the PINN-specific gradients
    #     pinn_grads = grads[1:]
    #     for g_i in pc_grad[1:]:
    #         random.shuffle(pinn_grads)
    #         for g_j in pinn_grads:
    #             g_i_g_j = torch.dot(g_i, g_j)
    #             if g_i_g_j < 0:
    #                 g_i -= (g_i_g_j) * g_j / (g_j.norm()**2)

    #     # Now project each pinn gradient onto the prime-objective, the PPO loss
    #     if len(pc_grad) > 1:
    #         g_prime = pc_grad[0]
            
    #         for g_sub in pinn_grads:
    #             g_s_g_p = torch.dot(g_sub, g_prime)
    #             if g_s_g_p < 0:
    #                 g_sub -= (g_s_g_p) * g_prime / (g_prime.norm()**2)
            
    #     merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

    #     if self._reduction:
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                        for g in pc_grad]).mean(dim=0)
    #     elif self._reduction == 'sum':
    #         merged_grad[shared] = torch.stack([g[shared]
    #                                        for g in pc_grad]).sum(dim=0)
    #     else: exit('invalid reduction method')

    #     merged_grad[~shared] = torch.stack([g[~shared]
    #                                         for g in pc_grad]).sum(dim=0)
    #     return merged_grad
    

    def _project_conflicting_pinn(self, grads, has_grads, shapes=None):
        assert len(grads) == 2
        shared = torch.stack(has_grads).prod(0).bool()
        grads_ = copy.deepcopy(grads)

        g_R = grads_[0]   # task gradient
        g_P = grads_[1]   # PINN gradient

        proj_coeff = torch.dot(g_R, g_P) / (g_R.norm() ** 2)

        g_P_orth = (g_P - proj_coeff * g_R)   # orthogonal component of the PINN loss gradient

        pp_grad = [grads[0], g_P_orth]   # Modified PINN loss gradient

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] for g in pp_grad]).mean(dim=0)
        elif self._reduction == "sum":
            merged_grad[shared] = torch.stack([g[shared] for g in pp_grad]).sum(dim=0)
        else:
            exit("invalid reduction method")

        merged_grad[~shared] = torch.stack([g[~shared] for g in pp_grad]).sum(dim=0)
        return merged_grad
    

    def _project_conflicting_pinn_balanced(self, grads, has_grads, shapes=None):
        assert len(grads) == 2
        # print("PCGrad with balanced scaling of PINN gradient")
        shared = torch.stack(has_grads).prod(0).bool()
        grads_ = copy.deepcopy(grads)

        g_R = grads_[0]   # task gradient
        g_P = grads_[1]   # PINN gradient

        proj_coeff = torch.dot(g_R, g_P) / (g_R.norm() ** 2)

        g_P_orth = (g_P - proj_coeff * g_R)   # orthogonal component of the PINN loss gradient

        # Adaptive \beta scaling to ensure the norm of projected PINN gradient
        #     is not greater than the norm of the task reward gradient
        beta = g_R.norm() / g_P_orth.norm() if g_P_orth.norm() > g_R.norm() else 1.0
        g_P_scaled = beta * g_P_orth

        pp_grad = [grads[0], g_P_scaled]   # Modified and scaled PINN loss gradient

        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] for g in pp_grad]).mean(dim=0)
        elif self._reduction == "sum":
            merged_grad[shared] = torch.stack([g[shared] for g in pp_grad]).sum(dim=0)
        else:
            exit("invalid reduction method")

        merged_grad[~shared] = torch.stack([g[~shared] for g in pp_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads, has_any_grad):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                # Leave parameters unused by every objective at ``None`` so
                # optimizer momentum and weight decay cannot move them.
                p.grad = grads[idx] if has_any_grad[idx] else None
                idx += 1
        return

    def _pack_grad(self, objectives):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads, param_masks = [], [], [], []
        for obj in objectives:
            self._optim.zero_grad(set_to_none=True)
            obj.backward(retain_graph=True)
            grad, shape, has_grad, param_has_grad = self._retrieve_grad()
            param_masks.append(param_has_grad)
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        has_any_grad = [any(flags) for flags in zip(*param_masks)]
        return grads, shapes, has_grads, has_any_grad

    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        get the gradient of the parameters of the network with specific 
        objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad, param_has_grad = [], [], [], []
        for group in self._optim.param_groups:
            for p in group["params"]:
                active = p.grad is not None
                param_has_grad.append(active)
                shape.append(p.shape)
                if active:
                    grad.append(p.grad.detach().clone())
                    has_grad.append(torch.ones_like(p))
                else:
                    grad.append(torch.zeros_like(p))
                    has_grad.append(torch.zeros_like(p))
        return grad, shape, has_grad, param_has_grad
