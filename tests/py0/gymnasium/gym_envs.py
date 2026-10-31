"""gymnasium: make / reset / step / spaces on CartPole, FrozenLake, Pendulum, MountainCar, Acrobot."""
import gymnasium as gym
import numpy as np

env = gym.make("CartPole-v1")
print(env)
obs, info = env.reset(seed=42)
print(obs, repr(obs), info, obs.dtype, obs.shape)
print(env.action_space, env.observation_space, env.observation_space.shape, env.action_space.n, env.action_space.seed(3))
a = env.action_space.sample()
print(a, int(a))
o, r, term, trunc, inf = env.step(a)
print(repr(o), r, term, trunc, inf)
total = 0.0
steps = 0
obs, info = env.reset(seed=7)
while True:
    action = 1 if float(obs[2]) > 0 else 0
    obs, r, term, trunc, info = env.step(action)
    total += r
    steps += 1
    if term or trunc:
        break
print(steps, total, obs)
fl = gym.make("FrozenLake-v1", render_mode="ansi")
print(fl, fl.observation_space, fl.observation_space.shape, fl.action_space)
o2, i2 = fl.reset(seed=1)
print(o2, int(o2))
for act in [2, 2, 1, 1, 1, 2]:
    o2, r2, t2, tr2, i2 = fl.step(act)
    print(o2, float(r2), t2, tr2, round(i2["prob"], 6))
print(fl.render())
p = gym.make("Pendulum-v1")
po, pi = p.reset(seed=0)
pr = 0.0
print(po, p.action_space, p.action_space.low, p.observation_space.high)
for k in range(3):
    po, pr, pt, ptr, pinfo = p.step(np.array([1.0], dtype="float32"))
    print(po, round(pr, 10))
p.action_space.seed(5)
print(repr(p.action_space.sample()), p.action_space.contains(np.array([0.5], dtype="float32")))
m = gym.make("MountainCar-v0")
mo, mi = m.reset(seed=3)
mr = 0.0
for k in range(5):
    mo, mr, mt, mtr, minfo = m.step(2)
print(m, m.observation_space, mo, mr)
ac = gym.make("Acrobot-v1")
ao, ai = ac.reset(seed=11)
ar = 0.0
for t in range(4):
    ao, ar, at, atr, ainfo = ac.step(t % 3)
print(ac.observation_space.shape, np.round(ao, 5), ar)
try:
    gym.make("Nope-v0")
except Exception as e:
    print("error")
