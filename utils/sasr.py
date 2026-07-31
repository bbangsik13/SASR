import torch
from collections import defaultdict
import math

class SASR: # Structure-Aware Sharpness Regularization
    def __init__(self, gaussians, rho_xyz=1.,rho_f_dc=0.,rho_f_rest=0.,rho_opacity=0.,rho_scaling=0.,rho_rotation=0., gamma=0.):
        self.gaussians=gaussians
        self.rho = {
            'xyz': rho_xyz, 
            'f_dc': rho_f_dc,
            'f_rest': rho_f_rest,
            'opacity': rho_opacity,
            'scaling': rho_scaling,
            'rotation': rho_rotation,
        }
        self.max_reg_weight = gamma
        self.state = defaultdict(dict)

    def clear_state(self):
        self.state.clear()
        torch.cuda.empty_cache()

    @torch.no_grad()
    def ascent_step(self,viewpoint_camera, depth):
        self.query_geometric_tolerance(viewpoint_camera,depth)
        N = self.tolerance_2d.shape[0]
        for n, p in self.gaussians.named_parameters.items(): 
            if p.grad is None:
                continue
            grad_norm = torch.norm(p.grad.reshape(N, -1), p=2, dim=1).view(N, *[1] * (p.dim() - 1))
            
            eps = self.state[p].get("eps")
            if eps is None:
                eps = torch.clone(p).detach()
                self.state[p]["eps"] = eps
            eps[...] = p.grad[...]
            self.state[p]["grad"] = p.grad.detach().clone() # save the gradient of the empirical loss for descent_step

            # Algorithm 1.5 : scale the perturbation magnitude considering the geometric tolerance
            if n in ['xyz','scaling']:
                eps.mul_(self.rho[n] * self.tolerance_3d.view_as(grad_norm) / (grad_norm + 1.e-16))
            elif n in ['rotation']:
                eps.mul_(self.rho[n] * self.tolerance_2d.view_as(grad_norm) / (grad_norm + 1.e-16))
            else:
                eps.mul_(self.rho[n] / (grad_norm + 1.e-16))
            
            p.add_(eps) # local maxima
        self.gaussians.optimizer.zero_grad()
    
    @torch.no_grad()
    def descent_step(self,viewpoint_camera):
        max_gamma = ((viewpoint_camera.max_kernel-1)*2 - 1) / 6.
        gamma = self.tolerance_2d / max_gamma * self.max_reg_weight
        for n, p in self.gaussians.named_parameters.items():
            if p.grad is None:
                continue
            p.sub_(self.state[p]["eps"]) # perturbed parameter → current parameter
            
            # Algorithm 1.9: scale the regularization weight
            if n in ['xyz','rotation','scaling']:
                gamma = gamma.view(p.shape[0], *[1] * (p.dim() - 1))
                p.grad = (1 - 2 * gamma) / (1 - gamma) * self.state[p]["grad"] + gamma / (1 - gamma) * p.grad 

    
    def query_geometric_tolerance(self,viewpoint_camera,depth):
        means3D = self.gaussians.get_xyz
        p_orig = torch.cat([means3D.detach(),torch.ones((means3D.shape[0],1)).cuda()],dim=-1)
        p_hom=p_orig@viewpoint_camera.full_proj_transform

        p_proj=(p_hom[:,:2]/(p_hom[:,3:4]+1e-7))
        H, W = viewpoint_camera.freq_map.shape
        coords = p_proj  
        x = coords[:, 0]
        y = coords[:, 1]

        ix = ((x + 1) * 0.5 * (W - 1)).round().long() 
        iy = ((y + 1) * 0.5 * (H - 1)).round().long() 

        ix = torch.clamp(ix, 0, W - 1)
        iy = torch.clamp(iy, 0, H - 1)
        self.tolerance_2d = viewpoint_camera.freq_map[iy, ix]  
        
        fx = viewpoint_camera.image_width / (2 * math.tan(viewpoint_camera.FoVx / 2))
        self.tolerance_3d =  self.tolerance_2d * depth.squeeze(0)[iy, ix]  / fx 