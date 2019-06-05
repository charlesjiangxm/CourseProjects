from nn_forward import MLP
import torch

torch.set_printoptions(profile="short")  # or 'default'
import numpy as np

# get the original mlp model
Mlp = MLP(model_loc="./model/mlp_model_1layer_256")
model = Mlp.init_model
Mlp.test(model)
print(model)

# apply numpy svd
RANK = 20
fc2_weight = np.array(model.state_dict()['fc2.weight'].cpu())
print("Original Weight Shape:", fc2_weight.shape)
u, s, vh = np.linalg.svd(fc2_weight)
u_truc = u[:, :RANK]
s_truc = np.diag(s[:RANK])
print("Preserved Rank: ", s[:RANK])
vh_truc = vh[:RANK, :]
fc2_weight_restruct = np.dot(u_truc, s_truc.dot(vh_truc))
print("Truncated Weight Shape: ", u_truc.shape, s_truc.shape)

# modify and forward the model again
model.fc2.weight.data = torch.Tensor(fc2_weight_restruct)
Mlp.test(model)
