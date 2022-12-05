import torch
import torch.nn.functional as F
from MyUtils.helpers import get_device


def relu_evidence(y):
    return F.relu(y)

def kl_divergence(beta_alpha, num_classes, device=None):
    # Calculate the KL divergence term according to Eq (8) in the paper
    if not device:
        device = get_device()
    ones = torch.ones([1, num_classes], dtype=torch.float32, device=device)
    sum_beta_alpha = torch.sum(beta_alpha, dim=1, keepdim=True)
    first_term = (
        torch.lgamma(sum_beta_alpha)
        - torch.lgamma(beta_alpha).sum(dim=1, keepdim=True)
        - torch.lgamma(ones.sum(dim=1, keepdim=True))
    )
    last_two_term = (
        (beta_alpha - ones)
        .mul(torch.digamma(beta_alpha) - torch.digamma(sum_beta_alpha))
        .sum(dim=1, keepdim=True)
    )
    kl = first_term + last_two_term
    return kl

def beta_loss(
    output, target, epoch_num, num_classes, annealing_step, device=None
):
    if not device:
        device = get_device()
    evidence = relu_evidence(output)
    #  Calculate the parameters of the Beta distribution determined by the evidence according to Eq (5) in the paper
    beta_alpha = evidence + 1
    beta_alpha = beta_alpha.to(device)
    S = torch.sum(beta_alpha, dim=1, keepdim=True)

    # Beta expectation loss
    A = torch.sum(target * (torch.digamma(S) - torch.digamma(beta_alpha)), dim=1, keepdim=True)
    # KL-divergence regularized term
    annealing_coef = torch.min(
        torch.tensor(1.0, dtype=torch.float32),
        torch.tensor(epoch_num / annealing_step, dtype=torch.float32),
    )

    # Calculate Beta parameters after removal of non-misleading evidence
    beta_alpha_hat = (beta_alpha - 1) * (1 - target) + 1
    kl_div = annealing_coef * kl_divergence(beta_alpha_hat, num_classes, device=device)

    loss=torch.mean(A+kl_div)
    return loss

if __name__ == '__main__':
    pass