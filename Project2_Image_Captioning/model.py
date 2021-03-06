import torch
import torch.nn as nn
import torchvision.models as models


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        for param in resnet.parameters():
            param.requires_grad_(False)
        
        modules = list(resnet.children())[:-1]
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)

    def forward(self, images):
        features = self.resnet(images)
        features = features.view(features.size(0), -1)
        features = self.embed(features)
        return features
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super().__init__()
        
        self.hidden_size = hidden_size
        
        self.hidden2out = nn.Linear(hidden_size, vocab_size)
        
        # Embedding layer that turns words to a vector
        self.word_embeddings = nn.Embedding(vocab_size, embed_size)
        
        # The LSTM takes embedded word vector as input and outputs hidden states of size hidden_dim
        self.lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, batch_first=True)
        
        self.softmax = nn.LogSoftmax(dim=1)

    
    def init_hidden(self, batch_size):

        return (torch.zeros((1, batch_size, self.hidden_size), device=device), 
                torch.zeros((1, batch_size, self.hidden_size), device=device))
    
    def forward(self, features, captions):
        
        # Remove the <end> word from the captions 
        captions = captions[:, :-1]     
        
        # Create embedded word vectors for each word in the captions
        captions = self.word_embeddings(captions)
        embeddings = torch.cat((features.unsqueeze(1), captions), dim=1)
        
        # the lstm takes in our embeddings and hidden state
        lstm_out, _ = self.lstm(embeddings)
        # Fully connected layer
        outputs = self.hidden2out(lstm_out) 

        return outputs

    def sample(self, inputs, states=None, max_len=20):
        " accepts pre-processed image tensor (inputs) and returns predicted sentence (list of tensor ids of length max_len) "
        sen = []
        for i in range(max_len):
            lstm_out, states = self.lstm(inputs, states)         
            outputs = self.hidden2out(lstm_out.squeeze(1))       
            _, predicted = outputs.max(dim=1)                    
            sen.append(predicted.item())
            
            inputs = self.word_embeddings(predicted)             
            inputs = inputs.unsqueeze(1)                         
        return sen
