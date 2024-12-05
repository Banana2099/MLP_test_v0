import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

class superAi(nn.Module):
    def __init__(self, input_n:int, dropout_rate:float):
        super().__init__()

        self.input = nn.Linear(input_n, 24).to(torch.float64)
        self.hidden = nn.Linear(24, 12).to(torch.float64)
        self.output = nn.Linear(12, 3).to(torch.float64)


        self.dr_rate = dropout_rate
    
    def forward(self, x):
        y = torch.tensor(x, dtype=torch.float64) if not torch.is_tensor(x) else x
        y = F.relu(self.input(y))
        y = F.dropout(y, p=self.dr_rate, training=self.training)

        # hidden
        y = F.relu(self.hidden(y))
        y = F.dropout(y, p=self.dr_rate, training=self.training)

        return self.output(y)


class aiTrainer:
    def __init__(self, model:superAi, learning_rate:float, discount_rate:float):
        self.model = model
        self.lr = learning_rate
        self.gamma = discount_rate
        self.optimizer = torch.optim.Adam(model.parameters(),lr=learning_rate)
        self.loss = nn.MSELoss()

    def train_step(self, inputs, actions_tensor, rewards, snake_status):
        inputs_tensor = torch.tensor(inputs, dtype=float)
        rewards_tensor = torch.tensor(rewards, dtype=float)
        snake_status = tuple(snake_status)

        pred =  self.model(inputs_tensor)
        target = pred.clone()
        for i, input in enumerate(inputs_tensor):
            act_value = rewards_tensor[i]
            # if snake_status[i] == "GameOver":
            #     pass
            target[i][torch.argmax(actions_tensor[i]).item()] = act_value

        self.optimizer.zero_grad()
        loss = self.loss(target, pred)
        loss.backward()
        self.optimizer.step()
        return self.model


def module_weight_average(modules_list:list):
    num_m = len(modules_list)
    base_module = copy.deepcopy(modules_list[0])
    base_weights = base_module.state_dict()

    for param in base_weights:
        base_weights[param] = base_weights[param]*0  

    for module in modules_list:
        temp_state = module.state_dict()
        for param in base_weights:
            base_weights[param] += temp_state[param]

    for param in base_weights:
        base_weights[param] = base_weights[param]/num_m

    base_module.load_state_dict(base_weights)      
    return base_module