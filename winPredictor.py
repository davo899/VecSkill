import torch
from winPredictorModel import MODEL
from torchsummary import summary

input_features = 330
model = MODEL.to(torch.device("cuda:0"))
model.load_state_dict(torch.load(input("Model file: ")))
summary(model, (1, input_features))
print(model)
