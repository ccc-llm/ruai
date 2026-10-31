"""torch: nn.Linear initialization, Sequential, losses, SGD / Adam training, models as plain classes."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(0)
lin = nn.Linear(3, 2)
print(lin, lin.weight, lin.bias)
print(lin(torch.tensor([[1.0, 2.0, 3.0]])), lin(torch.tensor([1.0, 0.0, -1.0])))
model = nn.Sequential(nn.Linear(2, 8), nn.Tanh(), nn.Linear(8, 1))
print(model, len(list(model.parameters())), model[0].weight.shape, model[2].bias)
X = torch.tensor([[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]])
y = torch.tensor([[0.0], [1.0], [1.0], [0.0]])
loss_fn = nn.MSELoss()
opt = optim.SGD(model.parameters(), lr=0.5, momentum=0.9)
for step in range(301):
    opt.zero_grad()
    loss = loss_fn(model(X), y)
    loss.backward()
    opt.step()
    if step % 100 == 0:
        print(step, round(loss.item(), 4))
print(loss_fn, model(X).detach().round(), (model(X) > 0.5).long().view(-1))


class Classifier:
    def __init__(self) -> None:
        self.hidden = nn.Linear(4, 16)
        self.out = nn.Linear(16, 3)

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        return self.out(torch.relu(self.hidden(x)))

    def parameters(self) -> list[torch.Tensor]:
        return list(self.hidden.parameters()) + list(self.out.parameters())


torch.manual_seed(42)
pts = torch.randn(60, 4)
labels = (pts[:, 0] > 0).long() + (pts[:, 1] > 0.5).long()
clf = Classifier()
adam = torch.optim.Adam(clf.parameters(), lr=0.01)
ce = nn.CrossEntropyLoss()
for epoch in range(200):
    adam.zero_grad()
    logits = clf(pts)
    l = ce(logits, labels)
    l.backward()
    adam.step()
    if epoch % 50 == 0:
        print(epoch, round(l.item(), 3))
with torch.no_grad():
    pred = clf(pts).argmax(dim=1)
print(ce, round((pred == labels).float().mean().item(), 3), pred[:10], labels[:10])
print(F.mse_loss(torch.tensor([1.0, 2.0]), torch.tensor([1.5, 2.5])), F.cross_entropy(torch.tensor([[2.0, 0.5]]), torch.tensor([0])))
print(nn.ReLU(), nn.Sigmoid(), nn.Softmax(dim=1), nn.Flatten(), nn.L1Loss()(torch.tensor([1.0]), torch.tensor([3.0])))
p = list(lin.parameters())[0]
print(p.grad, p.requires_grad, p.is_leaf)
