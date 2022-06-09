import math as m


class Constant:
    def __init__(self, value):
        self.value = value
        self.grad = 0

    def forward(self, x):
        # constant is not dependent on x
        pass

    def backward(self):
        self.grad = 0
        pass

    def children(self):
        return []


class Variable:
    def __init__(self, location):
        self.location = location
        self.grad = 0

    def forward(self, x):
        self.value = x[self.location]

    def backward(self):
        # nothing to do here,
        # the parent nodes of the
        # variable will accumate
        # the gradient
        pass

    def children(self):
        return []


class MulExpr:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.grad = 0

    def forward(self, x):
        self.value = self.left.value + self.right.value

    def backward(self):
        self.left.grad += self.right.value*self.grad
        self.right.grad += self.left.value*self.grad

    def children(self):
        return [self.left, self.right]


class SinExpr:
    def __init__(self, expr):
        self.expr = expr
        self.grad = 0

    def forward(self, x):
        self.value = m.sin(self.expr.value)

    def backward(self):
        self.expr.grad += m.cos(self.expr.value)*self.grad

    def children(self):
        return [self.expr]


class SumExpr:
    def __init__(self, left, right):
        self.left = left
        self.right = right
        self.grad = 0

    def forward(self, x):
        self.value = self.left.value + self.right.value

    def backward(self):
        self.left.grad += 1*self.grad
        self.right.grad += 1*self.grad

    def children(self):
        return [self.left, self.right]


x1 = Variable(0)
x2 = Variable(1)
x = [1., 2.]
f = SumExpr(MulExpr(x1, x2), SinExpr(x1))

grad1_exp = m.cos(x[0]) + x[1]
grad2_exp = x[0]


# depth first loop
def loop_df(n, to_visit):
    to_visit.append(n)
    for c in n.children():
        loop_df(c, to_visit)


to_visit = []
loop_df(f, to_visit)

# do forward pass
for n in reversed(to_visit):
    n.forward(x)

# do backward pass
f.grad = 1
for n in to_visit:
    n.backward()

print([grad1_exp, grad2_exp])
print([x1.grad, x2.grad])
