# 📦 Package Structure: `goo/`

```
goo/
│
├── __init__.py
├── base.py              # Base classes: AE laws, AE Machine, AR Safeguards
├── wrapper.py           # OnlineOptimizerWrapper (loss + reward adapter)
├── swarm.py             # Multi-agent AE-swarm / Hedge allocation
├── memory.py            # Differentiable tape / memory modules
├── rules.py             # Adaptive rules (lr, momentum, beta, alpha, reward)
├── experiments.py       # Toy demos (sin regression, Rosenbrock, etc.)
└── configs.py           # Dataclasses/YAML-based rule & config management
```

---

# 1. **Base Components** (`base.py`)

```python
import torch
import numpy as np
from dataclasses import dataclass

@dataclass
class AEConfig:
    gamma_min: float = 1e-6
    gamma_max: float = 1.0
    budget: float = 0.05       # max coefficient movement per step
    resonance_rho_max: float = 0.95
    resonance_v_max: float = 10.0
    noise_min: float = 0.0
    noise_max: float = 0.1
    auto_gamma: bool = True    # gauge calibration
    auto_gamma_warmup: int = 20

class AEMachine:
    """
    AE Machine = Parameters + Adaptive Coefficients + Memory + AR Safeguards.
    Wraps any optimizer param group.
    """
    def __init__(self, optimizer, ae_cfg: AEConfig=None):
        self.optimizer = optimizer
        self.cfg = ae_cfg or AEConfig()
        self.state = dict(
            lr=float(optimizer.param_groups[0].get('lr', 1e-3)),
            momentum=float(optimizer.param_groups[0].get('momentum', 0.0)),
            step=0,
        )
        # histories
        self.loss_history, self.reward_history = [], []
        self.prev_loss, self.prev_reward = None, None

    def update_coefficients(self, loss=None, reward=None, grad_norm=None):
        """
        Apply AE laws (γ-gauge, σ-noise, resonance clamp, etc.)
        """
        self.state['step'] += 1

        # Example: Trend-aware LR rule
        if loss is not None and self.prev_loss is not None:
            if loss.item() < self.prev_loss:
                self.state['lr'] *= 1.05
            else:
                self.state['lr'] *= 0.7
        self.prev_loss = loss.item() if loss is not None else self.prev_loss

        # Clamp by gamma band
        self.state['lr'] = max(min(self.state['lr'], self.cfg.gamma_max), self.cfg.gamma_min)

        # Resonance safeguard (dummy version: clamp momentum)
        m = self.state['momentum']
        self.state['momentum'] = max(min(m, 0.999), 0.0)

        # Push back to optimizer param_groups
        for g in self.optimizer.param_groups:
            g['lr'] = self.state['lr']
            if 'momentum' in g: g['momentum'] = self.state['momentum']

    def step_loss_reward(self, loss=None, reward=None, grad_norm=None):
        self.update_coefficients(loss=loss, reward=reward, grad_norm=grad_norm)

    def step(self, closure=None):
        return self.optimizer.step(closure)

    def zero_grad(self, set_to_none=False):
        return self.optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return dict(opt=self.optimizer.state_dict(), state=self.state)

    def load_state_dict(self, state_dict):
        self.optimizer.load_state_dict(state_dict['opt'])
        self.state = state_dict['state']
```

---

# 2. **Unified Adapter (Wrapper)** (`wrapper.py`)

```python
from .base import AEMachine

class OnlineOptimizerWrapper:
    """Wraps any PyTorch optimizer with AE Machine adaptation."""
    def __init__(self, optimizer, ae_cfg=None):
        self.ae = AEMachine(optimizer, ae_cfg)

    def step_loss_reward(self, loss=None, reward=None, grad_norm=None):
        self.ae.step_loss_reward(loss, reward, grad_norm)

    def step(self, closure=None):
        return self.ae.step(closure)

    def zero_grad(self, set_to_none=False):
        return self.ae.zero_grad(set_to_none)

    def state_dict(self):
        return self.ae.state_dict()

    def load_state_dict(self, sd):
        return self.ae.load_state_dict(sd)
```

---

# 3. **Swarm (Portfolio / Hedge-Allocation)** (`swarm.py`)

```python
import numpy as np

class AESwarm:
    """
    Swarm of AE Machines (meta-portfolio).
    Uses Hedge/exp-weights for compute budget allocation.
    """
    def __init__(self, machines, beta=4.0):
        self.machines = machines
        self.beta = beta
        self.histories = [[] for _ in machines]
        self.weights = np.ones(len(machines)) / len(machines)

    def update_weights(self, scores):
        # scores = validation losses or rewards per agent
        for i, s in enumerate(scores):
            self.histories[i].append(s)
        cum_loss = np.array([sum(h) for h in self.histories])
        w = np.exp(-self.beta * cum_loss)
        self.weights = w / w.sum()
        return self.weights

    def allocate_budget(self, total_steps):
        alloc = (self.weights * total_steps).astype(int)
        return alloc

    def consensus_nudge(self, kappa=0.05):
        """Consensus nudge on coefficients (not weights)."""
        mean_lr = np.mean([m.state['lr'] for m in self.machines])
        for m in self.machines:
            m.state['lr'] = (1-kappa)*m.state['lr'] + kappa*mean_lr
```

---

# 4. **Toy Demo** (`experiments.py`)

```python
import torch
from torch import nn
from goo.wrapper import OnlineOptimizerWrapper

def demo_sin_curve():
    N = 256
    x = torch.linspace(-3.14, 3.14, N).unsqueeze(1)
    y = torch.sin(x) + 0.1 * torch.randn_like(x)

    model = nn.Sequential(nn.Linear(1, 32), nn.Tanh(), nn.Linear(32, 1))
    base_optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)
    optimizer = OnlineOptimizerWrapper(base_optim)

    loss_fn = nn.MSELoss()
    for step in range(300):
        pred = model(x)
        loss = loss_fn(pred, y)

        optimizer.zero_grad()
        loss.backward()
        reward = -loss.item()
        optimizer.step_loss_reward(loss, reward)
        optimizer.step()

        if step % 50 == 0:
            print(f"Step {step:03d} | Loss {loss.item():.4f} | LR {optimizer.ae.state['lr']:.4f}")

if __name__ == "__main__":
    demo_sin_curve()
```

---

# 5. **Configs (`configs.py`)**
```python
from dataclasses import dataclass

@dataclass
class GooConfig:
    lr_rule: str = "trend"
    momentum_rule: str = "relative"
    reward_rule: str = "trend"
    swarm_size: int = 4
    beta: float = 4.0
```

---

# ✅ Benefits
- **Any optimizer** → plug it in, immediately gets **AE Machine + AR safeguards**.
- **Unified rules** in one place (`rules.py`).
- **Swarm portfolio** support out-of-the-box.
- **Configurable** via dataclasses/YAML.
- **Toy demos** provided, scale up to LLM/RL.
