import torch
class LSTMNetwork(torch.nn.Module):

  def __init__(self, input_size, hidden_size, num_classes):
    super(LSTMNetwork, self).__init__()
    ## LSTM Layer 
    self.lstm = torch.nn.LSTM(input_size = input_size,
                              hidden_size = hidden_size,
                              batch_first = True)
    ## linear layer
    self.linear = torch.nn.Linear(hidden_size, num_classes)

  def forward(self, input_data):
    _, (hidden, _) = self.lstm(input_data)
    output = self.linear(hidden[-1])
    return output