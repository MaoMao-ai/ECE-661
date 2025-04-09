import torch
import torch.nn as nn
import torch.nn.functional as F

def random_noise_attack(model, device, dat, eps):
    # Add uniform random noise in [-eps,+eps]
    x_adv = dat.clone().detach() + torch.FloatTensor(dat.shape).uniform_(-eps, eps).to(device)
    # Clip the perturbed datapoints to ensure we are in bounds [0,1]
    x_adv = torch.clamp(x_adv.clone().detach(), 0., 1.)
    # Return perturbed samples
    return x_adv

# Compute the gradient of the loss w.r.t. the input data
def gradient_wrt_data(model,device,data,lbl):
    dat = data.clone().detach()
    dat.requires_grad = True
    out = model(dat)
    loss = F.cross_entropy(out,lbl)
    model.zero_grad()
    loss.backward()
    data_grad = dat.grad.data
    return data_grad.data.detach()


def PGD_attack(model, device, dat, lbl, eps, alpha, iters, rand_start):
    # TODO: Implement the PGD attack
    # - dat and lbl are tensors
    # - eps and alpha are floats
    # - iters is an integer
    # - rand_start is a bool

    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach().to(device)

    # If rand_start is True, add uniform noise to the sample within [-eps,+eps],
    # else just copy x_nat
    if rand_start:
        noise = torch.empty_like(x_nat).uniform_(-eps, eps)
        x_adv = torch.clamp(x_nat + noise, 0.0, 1.0)
    else:
        x_adv = x_nat.clone().detach()

    # Make sure the sample is projected into original distribution bounds [0,1]

    # Iterate over iters
    for i in range(iters):
        # Compute gradient w.r.t. data (we give you this function, but understand it)
        data_grad = gradient_wrt_data(model, device, x_adv, lbl)
        # Perturb the image using the gradient
        x_adv = x_adv + alpha * data_grad.sign()
        # Clip the perturbed datapoints to ensure we still satisfy L_infinity constraint
        x_adv = torch.clamp(x_adv, x_nat - eps, x_nat + eps)
        # Clip the perturbed datapoints to ensure we are in bounds [0,1]
        x_adv = torch.clamp(x_adv, 0.0, 1.0)
    # Return the final perturbed samples
    return x_adv


def FGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    x_nat = dat.clone().detach().to(device)
    x_nat.requires_grad = True
    out = model(x_nat)
    loss = F.cross_entropy(out, lbl)
    model.zero_grad()
    loss.backward()
    data_grad = x_nat.grad.data
    x_adv = x_nat + eps * data_grad.sign()
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()


def rFGSM_attack(model, device, dat, lbl, eps):
    # TODO: Implement the FGSM attack
    # - Dat and lbl are tensors
    # - eps is a float
    # Clone and detach the original data
    x_nat = dat.clone().detach().to(device)
    alpha = eps / 2 
    rand_perturb = torch.FloatTensor(x_nat.shape).uniform_(-alpha, alpha).to(device)
    x_adv = x_nat + rand_perturb
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    x_adv.requires_grad = True
    out = model(x_adv)
    loss = F.cross_entropy(out, lbl)
    model.zero_grad()
    loss.backward()
    data_grad = x_adv.grad.data
    x_adv = x_adv + (eps - alpha) * data_grad.sign()
    x_adv = torch.max(torch.min(x_adv, x_nat + eps), x_nat - eps)
    x_adv = torch.clamp(x_adv, 0.0, 1.0)
    return x_adv.detach()


def FGM_L2_attack(model, device, dat, lbl, eps):
    # x_nat is the natural (clean) data batch, we .clone().detach()
    # to copy it and detach it from our computational graph
    x_nat = dat.clone().detach().to(device)
    x_nat.requires_grad = True

    # Forward pass
    out = model(x_nat)
    loss = F.cross_entropy(out, lbl)

    # Zero all existing gradients
    model.zero_grad()

    # Backward pass to compute gradients w.r.t. input data
    loss.backward()
    data_grad = x_nat.grad.data 

    # Flatten gradient per sample and compute L2 norm
    batch_size = data_grad.shape[0]
    data_grad_flat = data_grad.view(batch_size, -1) 

    # Compute L2 norm for each sample
    l2_norm = torch.norm(data_grad_flat, p=2, dim=1)
    l2_norm = l2_norm.view(batch_size, 1, 1, 1) 

    # Prevent division by zero
    l2_norm = torch.clamp(l2_norm, min=1e-12)

    # Normalize the gradient
    grad_normalized = data_grad / l2_norm

    # Perturb the data
    x_adv = x_nat + eps * grad_normalized

    # Clip to maintain [0,1] range
    x_adv = torch.clamp(x_adv, 0.0, 1.0)

    return x_adv.detach()