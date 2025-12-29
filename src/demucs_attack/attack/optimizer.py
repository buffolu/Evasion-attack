import torch
from tqdm import tqdm

from demucs_attack.attack.loss import combined_source_loss
from demucs_attack.attack.apply_model_grad import apply_model_grad


class AdversarialAttack:
    def __init__(self, model, epsilon, lr, device):
        self.model = model
        self.epsilon = epsilon
        self.lr = lr
        self.device = device

    def run(self, mixture, sources, iterations):
        """
        sources: dict with keys {mixture, drums, bass, other, vocals}
        """
        perturbation = torch.zeros_like(mixture, requires_grad=True)
        optimizer = torch.optim.Adam([perturbation], lr=self.lr)

        best_loss = float("inf")
        best_perturbation = None
        stem_order = ["drums", "bass", "other", "vocals"]

        for _ in tqdm(range(iterations)):
            optimizer.zero_grad()

            with torch.no_grad():
                perturbation.clamp_(-self.epsilon, self.epsilon)
            perturbed = mixture + perturbation

            estimated = apply_model_grad(
                self.model,
                perturbed.unsqueeze(0),
                device=self.device,
            ).squeeze(0)

            loss = 0.0

            for i, name in enumerate(stem_order):
                est = estimated[i]
                tgt = sources[name]
                loss += combined_source_loss(est, tgt, self.device)

            loss.backward()
            optimizer.step()
            perturbation.data.clamp_(-self.epsilon, self.epsilon)



            if loss.item() < best_loss:
                best_loss = loss.item()
                best_perturbation = perturbation.detach().clone()
                print(f'new loss is {best_loss}')
        return best_perturbation
