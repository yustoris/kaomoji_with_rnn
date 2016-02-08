import numpy as np
from data_loader import DataLoader
import random

class ReccurentNetwork:
    def __init__(self, data, size):
        self.data = data

        self.input_size = size
        self.output_size = size
        self.hidden_size = 100

        # Initialize weights and biases
        self.W_input = np.random.uniform(
            -np.sqrt(1./self.input_size), np.sqrt(1./self.input_size),
            (self.hidden_size, self.input_size))
        self.W_output = np.random.uniform(
            -np.sqrt(1./self.hidden_size), np.sqrt(1./self.hidden_size),
            (self.output_size, self.hidden_size))
        self.W_hidden = np.random.uniform(
            -np.sqrt(1./self.hidden_size), np.sqrt(1./self.hidden_size),
            (self.hidden_size, self.hidden_size))
        self.b_input = np.zeros((self.hidden_size, 1))
        self.b_output = np.zeros((self.output_size, 1))        

    def update_batch(self, loader):
        random.shuffle( self.data )
        mini_batch_size = 100
        data_size = len(self.data)
        batches = [
            self.data[k:k+mini_batch_size] for k
            in range(0, data_size, mini_batch_size)
        ]

        mem_dW_output = np.zeros_like(self.W_output)
        mem_dW_hidden = np.zeros_like(self.W_hidden)
        mem_dW_input = np.zeros_like(self.W_input)

        mem_db_output = np.zeros_like(self.b_output)
        mem_db_input = np.zeros_like(self.b_input)

        loss_acc = 0.0
        for batch in batches:
            loss = 0.0
            eta = 0.1
            for x, d in batch:
                z_hidden, u_hidden, y = self.forward(x)
                loss += self.loss(y, d)
                dW_input, dW_hidden, dW_output, db_input, db_output = self.backdrop(
                    x, u_hidden, z_hidden, y, d
                )
                for param, dparam, mem in zip(
                        [self.W_input, self.W_hidden, self.W_output, self.b_input, self.b_output],
                        [dW_input, dW_hidden, dW_output, db_input, db_output],
                        [mem_dW_input, mem_dW_hidden, mem_dW_output, mem_db_input, mem_db_output],
                ):
                    mem += dparam * dparam
                    param += -eta * dparam / np.sqrt(mem + 1e-8)

            print('Batch', loss/mini_batch_size, mini_batch_size)
            self.test(loader)
            loss_acc += loss/mini_batch_size
        print('Batch Avg', loss_acc/len(batches))

    def loss(self, y, d):
        loss = 0.0
        for t in range(len(y)):
            target_idx = np.argmax(d[t])
            loss += -np.log(y[t][target_idx,0])
        return loss
            
    def forward(self, x):
        t_max = len(x)
        z_hidden = [np.zeros((self.hidden_size,1)) for t in range(t_max)]
        u_hidden = [np.zeros((self.hidden_size,1)) for t in range(t_max)]
        y = [np.zeros((self.output_size,1)) for t in range(t_max)]
        for t in range(t_max):
            # Hidden layer
            u_hidden[t] = self.W_input.dot(x[t]) + self.b_input
            if t >= 1:
                u_hidden[t] += self.W_hidden.dot(z_hidden[t-1])
            z_hidden[t] = np.tanh(u_hidden[t])

            # Output layer
            y[t] = self.softmax(self.W_output.dot(z_hidden[t]) + self.b_output)
            
        return (z_hidden, u_hidden, y)
    
    def backdrop(self, x, u_hidden, z_hidden, y, d):
        t_max = len(y)
        delta_hidden = [np.zeros((self.output_size, 1)) for t in range(t_max)]
        dW_output = np.zeros_like(self.W_output)
        dW_hidden = np.zeros_like(self.W_hidden)
        dW_input = np.zeros_like(self.W_input)

        db_output = np.zeros_like(self.b_output)
        db_input = np.zeros_like(self.b_input)
        
        for t in reversed(range(t_max)):
            delta_output = y[t].copy()
            delta_output[np.argmax(d[t])] -= 1.0
            
            delta_hidden[t] = self.W_output.T.dot(delta_output)
            if t <= t_max - 2:
                delta_hidden[t] += self.W_hidden.T.dot(delta_hidden[t+1])
            delta_hidden[t] *= 1 - z_hidden[t]**2 # self.tanh_d(u_hidden[t])

            dW_input += delta_hidden[t].dot(x[t].T)
            db_output += delta_output
            db_input += delta_hidden[t]
            dW_hidden += delta_hidden[t].dot(z_hidden[t-1].T)
            dW_output += delta_output.dot(z_hidden[t].T)

        for dparam in [dW_input, dW_hidden, dW_output, db_input, db_output]:
            np.clip(dparam, -5, 5, out=dparam)
        return (dW_input, dW_hidden, dW_output, db_input, db_output)
            

    def test(self, loader):
        # for init_c in np.random.choice(list(loader.chars), 10):
        for attempt in range(10):
            init_c = '('
            print(init_c, end='')
        
            c_idx = loader.char_to_idx[init_c]
            for t in range(100):
                z_hidden, u_hidden, y = self.forward(
                    [loader._one_hot_vec(len(loader.chars)+1, c_idx)]
                )
                c_idx = np.random.choice(range(len(loader.chars)+1), p=y[-1].ravel())
                if c_idx >= len(loader.chars):
                    break
                print(loader.idx_to_char[c_idx], end='')
            print()
            
    ### Misc functions

    def tanh_d(self, x):
        return 1.0 - np.tanh(x)**2

    def softmax(self, x):
        de = np.exp(x - np.max(x))
        return de/np.sum(de)

    def softmax_d(self, x):
        soft_max_v = self.softmax(x)
        return soft_max_v*(1 - soft_max_v)

if __name__ == '__main__':
    loader = DataLoader()
    rnn = ReccurentNetwork(loader.char_vecs, len(loader.chars)+1)
    epoch = 1

    while True:
        print('Epoch', epoch)
        rnn.update_batch(loader)
        epoch+=1
    
