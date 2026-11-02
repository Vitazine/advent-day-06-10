import mlx.core as mx
from loss import completion_loss as default_loss

class UniformModel:
    def __call__(self, inputs):
        return mx.zeros((*inputs.shape, 16))

batch=mx.array([[1,2,3,4,5,6,7,8,9,10,0,0], [1,2,3,4,5,6,0,0,0,0,0,0]])
lengths=mx.array([[6,10],[3,6]])
loss,n=default_loss(UniformModel(),batch,lengths)
assert n.item()==7, f'Expected 4+3 real target tokens, got {n.item()}'
assert abs(loss.item()-2.7725887)<1e-5
print('Actual masked loss padding boundary: PASS')
