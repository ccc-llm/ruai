"""torch: tensors, dtypes, printing, autograd, RNG (manual_seed / rand / randn)."""
import torch

x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], requires_grad=True)
v = torch.tensor([1.0, -1.0], requires_grad=True)
print(x, v, x.shape, x.size(0), x.dtype, x.requires_grad, x.is_leaf, x.grad)
for t in [x + 1, 1 + x, x - 1, 1 - x, x * 2, x / 2, 1 / x, -x, x ** 2, x * x, x @ x, x @ v, v @ v, v @ x]:
    print(t)
for t in [x.sum(), x.sum(dim=0), x.mean(), x.mean(dim=1), x.relu(), torch.sigmoid(x), x.tanh(), (x / 4).exp(), x.log(), x.sqrt(), x.abs(), x.softmax(dim=1), torch.log_softmax(x, dim=1)]:
    print(t)
for t in [x.view(4), x.reshape(-1), x.t(), x.T, x.unsqueeze(0), x[0], x[0:1], x[0, 1], x.max(), x.flatten()]:
    print(t)
print(torch.tensor([1.0, 1e-5]), torch.tensor([1.0, 2.0, 3.0]), torch.tensor([1e9, 1.0]), torch.tensor([0.5, 2000.0]), torch.tensor([True, False]), torch.tensor([[1, -20], [3, 4]]))
print(torch.arange(5), torch.arange(0.0, 1.0, 0.25), torch.zeros(2), torch.zeros(2, 3), torch.ones(3, dtype=torch.long), torch.full((2,), 3.0), torch.tensor(2.5), torch.tensor(7))
print(x.sum().item(), (x > 2), (x == 1).sum(), x.argmax(), x.argmax(dim=1))
vals, idx = x.max(dim=1)
print(vals, idx)
y = (x * x).sum()
y.backward()
print(x.grad, x.requires_grad, y.requires_grad, y)
with torch.no_grad():
    z = x * 2
print(z, z.requires_grad, x.detach() * 2)
torch.manual_seed(0)
print(torch.rand(5), torch.randn(3), torch.randn(4))
torch.manual_seed(0)
print(torch.randn(2, 20))
torch.manual_seed(1)
print(torch.rand(2, 3), torch.randn(40)[:4])
print(torch.tensor([0.1, 0.2]).numpy(), torch.from_numpy(torch.tensor([1.5]).numpy()), torch.cat([v.detach(), v.detach()]), torch.stack([v.detach(), v.detach()]))
for bad in [0, 1, 2, 3]:
    try:
        if bad == 0:
            print(x @ torch.ones(3, 2))
        elif bad == 1:
            print(x + torch.ones(3))
        elif bad == 2:
            x.sum(dim=0).backward()
        else:
            print(x.numpy())
    except RuntimeError as e:
        print("RuntimeError:", e)
