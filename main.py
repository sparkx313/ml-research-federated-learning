import numpy as np
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import random

@dataclass
class FLConfig:
    n_clients: int = 10
    n_rounds: int = 20
    dp_epsilon: float = 1.0
    dp_delta: float = 1e-5
    dp_max_grad_norm: float = 1.0
    byzantine_fraction: float = 0.2

class FederatedModel:
    def __init__(self, input_dim=784, n_classes=10):
        np.random.seed(42)
        self.W1 = np.random.randn(input_dim, 128) * 0.01
        self.b1 = np.zeros(128)

    def get_params(self) -> Dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy()}

    def set_params(self, params: Dict):
        self.W1 = params["W1"].copy()
        self.b1 = params["b1"].copy()

    def simulate_local_update(self) -> Dict:
        return {k: v + np.random.randn(*v.shape)*0.001 for k,v in self.get_params().items()}

def clip_gradients(update, global_params, max_norm):
    diff = {k: update[k] - global_params[k] for k in update}
    norm = np.sqrt(sum(np.sum(d**2) for d in diff.values()))
    scale = min(1.0, max_norm / max(norm, 1e-10))
    return {k: global_params[k] + diff[k]*scale for k in update}

def gaussian_mechanism(gradients, sensitivity, epsilon, delta):
    sigma = sensitivity * np.sqrt(2*np.log(1.25/delta)) / epsilon
    return {k: v + np.random.randn(*v.shape)*sigma for k,v in gradients.items()}

def krum_aggregation(updates: List[Dict], f: int) -> Dict:
    n = len(updates)
    flat = [np.concatenate([v.flatten() for v in u.values()]) for u in updates]
    scores = [sum(sorted([np.linalg.norm(flat[i]-flat[j])**2
                          for j in range(n) if j!=i])[:n-f-2])
              for i in range(n)]
    return updates[int(np.argmin(scores))]

def run_federated(config: FLConfig):
    model = FederatedModel()
    print(f"FL: {config.n_clients} clients, {config.n_rounds} rounds, "
          f"epsilon={config.dp_epsilon}, byzantine={config.byzantine_fraction:.0%}")
    for rnd in range(1, config.n_rounds+1):
        updates = []
        for c in range(config.n_clients):
            local = model.simulate_local_update()
            clipped = clip_gradients(local, model.get_params(), config.dp_max_grad_norm)
            noisy = gaussian_mechanism(clipped, config.dp_max_grad_norm,
                                       config.dp_epsilon, config.dp_delta)
            if c < int(config.n_clients * config.byzantine_fraction):
                noisy = {k: np.random.randn(*v.shape) for k,v in noisy.items()}
            updates.append(noisy)
        n_byz = int(config.n_clients * config.byzantine_fraction)
        agg = krum_aggregation(updates, f=n_byz)
        model.set_params(agg)
        if rnd % 5 == 0:
            norm = np.sqrt(sum(np.sum(v**2) for v in model.get_params().values()))
            print(f"  Round {rnd:3d}: param_norm={norm:.4f}")

if __name__ == "__main__":
    run_federated(FLConfig(n_clients=20, n_rounds=20, dp_epsilon=1.0, byzantine_fraction=0.15))
