import torch

import torch
x=torch.tensor([1,2,3],dtype=torch.float32,requires_grad=True)
y = torch.tensor([1,2,3],dtype=torch.float32,requires_grad=True)
z = torch.tensor([1,2,3],dtype=torch.float32,requires_grad=True)
a = torch.tensor(1,dtype=torch.float32,requires_grad=True)
b = torch.tensor(2,dtype=torch.float32,requires_grad=True)
c = torch.tensor(3,dtype=torch.float32,requires_grad=True)

r = (3*x+2*y).dot(z)
s = (2*x+2*y).dot(z)
t = (x + y).dot(z)

r2=r*a
s2=s*b
t2=t*c

loss=r2+2*s2+3*t2
loss.backward()
r.retain_grad()
s.retain_grad()
t.retain_grad()
r2.retain_grad()
s2.retain_grad()
t2.retain_grad()
print(a.grad)
print(b.grad)
print(c.grad)
print(r.grad)
print(s.grad)
print(t.grad)
print(x.grad)
print(y.grad)
print(z.grad)