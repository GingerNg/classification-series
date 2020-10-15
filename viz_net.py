import torch
from torch import nn
from torchviz import make_dot
import cfg
from torchvision.models import AlexNet
# from cnn_flower import CNNNet


model = cfg.MODEL_NAMES[cfg.model_name](num_classes=cfg.NUM_CLASSES)

# model = CNNNet()
x = torch.randn(1, 3, 32, 32).requires_grad_(True)
y = model(x)
vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
vis_graph.view()



# model = AlexNet()

# x = torch.randn(1, 3, 227, 227).requires_grad_(True)
# y = model(x)
# vis_graph = make_dot(y, params=dict(list(model.named_parameters()) + [('x', x)]))
# vis_graph.view()
