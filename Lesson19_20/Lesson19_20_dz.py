import torch
# x = torch.linspace(-6, 6, 13)
# s = torch.sigmoid(x)
# print("x sigma sigma'")
# for a, b, c in zip(x, s, s * (1 - s)):
#     print(f"{float(a):6.1f} {float(b):8.4f} {float(c):8.4f}")


x = torch.linspace(-3, 3, 10)
s = torch.sigmoid(x)
t = torch.tanh(x)
print(f"{'x':>6} {'sigmoid':>10} {'sigmoid_grad':>14} {'tanh':>10} {'tanh_grad':>12}")
for a, b, c, d, e in zip(x, s, s * (1 - s), t, 1 - t**2):
    print(f"{float(a):6.1f} {float(b):10.4f} {float(c):14.4f} {float(d):10.4f} {float(e):12.4f}")