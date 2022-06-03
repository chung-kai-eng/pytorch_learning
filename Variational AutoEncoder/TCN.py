  
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm


class Chomp1d(nn.Module): # padding
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        print(padding)
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


# num_channels: the number of series 
# num_inputs: number of input period data
class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size-1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

    def compute_output_shape(self, input_shape):
        pass



# num_input -> num_channel[0] -> num_channel[1]
# num_input -> num_channel[0] -> num_channel[1]
# num_input -> num_channel[0] -> num_channel[1]
class TCNEncoder(nn.Module):
    def __init__(self, input_shape=None, num_tcn_layers=3, tcn_num_channels=[64, 32, 16], kernel_size=3, latent_dim=10):
        """
        num_inputs {}
        kernel_size {int}: receptive field (the period of time)
        """
        super(TCNEncoder, self).__init__()
        layers = []
        self.input_shape = input_shape # (num_channel, seq_len)
        self.dynamic_input_channel = input_shape[0] # input_channel -> tcn_num_channel (final channel of tcn)
        self.dynamic_input_shape = input_shape[1] # sequence length

        for i in range(num_tcn_layers):
            if i != 0:
                self.compute_input_shape(pooling_kernel_size=2)
                self.dynamic_input_channel = tcn_num_channels[-1]
                
            tcn_layer = TemporalConvNet(num_inputs=self.dynamic_input_channel, num_channels=tcn_num_channels, kernel_size=kernel_size)
            layers.append(tcn_layer)
            # BatchNormalization
            # print([x for x in tcn_layer.children()])
            layers.append(nn.BatchNorm1d(tcn_num_channels[-1]))
            # Max Pooling
            layers.append(nn.MaxPool1d(kernel_size=2))

        self.encode_tcn_layer = nn.Sequential(*layers)

        self.linear1 = nn.Linear(16*2, latent_dim) # compute mean
        self.linear2 = nn.Linear(16*2, latent_dim) # compute log variance

    def compute_input_shape(self, pooling_kernel_size=2):
        # causal padding = (kernel_size - 1) * dilation
        self.dynamic_input_shape = self.dynamic_input_shape // pooling_kernel_size

    def forward(self, x):
        x = self.encode_tcn_layer(x)
        print(x.shape)
        x = torch.flatten(x, start_dim=1)
        print('flatten shape: {}'.format(x.shape)) 

        mu = self.linear1(x) # compute mean
        print(mu.shape)
        log_var = self.linear2(x) # compute log variance
        # print(log_var.shape)
        return mu, log_var

# thincking more about upsampling (corresponding to max pooling)
# ConvTranspose also can be utilized or not?
class TCNDecoder(nn.Module):
    def __init__(self, latent_dim=10, num_tcn_layers=3, tcn_num_channels=[64, 32, 16], kernel_size=3):
        super(TCNDecoder, self).__init__()

        # with opposite order of num_channels in encoder
        num_tcn_layers = len(tcn_num_channels)
        self.linear1 = nn.Linear(latent_dim, 16*2)
        self.unflatten = nn.Unflatten(dim=1, unflattened_size=torch.Size([16, 2])) # 
        self.dynamic_input_shape = latent_dim
        self.dynamic_input_channel = 16 # input_channel -> tcn_num_channel (final channel of tcn)

        layers = []

        for i in range(num_tcn_layers):
            if i != 0:
                self.compute_input_shape(pooling_kernel_size=2)
                self.dynamic_input_channel = tcn_num_channels[-1]
            
            layers.append(nn.Upsample(scale_factor=2))
            layers.append(nn.BatchNorm1d(tcn_num_channels[-1]))
            
            tcn_layer = TemporalConvNet(num_inputs=self.dynamic_input_channel, num_channels=tcn_num_channels, kernel_size=kernel_size, dropout=0.0)
            layers.append(tcn_layer)

        self.decoder_tcn_layer = nn.Sequential(*layers)
        # for the last layer use same padding
        # self.output = nn.Conv1d( kernel_size=3, padding='same', activation='sigmoid' ) # softmax (multi-class)

    def compute_input_shape(self, pooling_kernel_size=2):
        # causal padding = (kernel_size - 1) * dilation
        self.dynamic_input_shape *= pooling_kernel_size

    def forward(self, x):
        x = self.linear1(x)
        x = self.unflatten(x) # reshape 
        print('after unflatten: ', x.shape)
        x = self.decoder_tcn_layer(x)
        print('after decoder: ', x.shape)
        x = self.output(x)

        return x


class TCN_VAE(nn.Module):
    def __init__(self):
        super(TCN_VAE, self).__init__()
        self.encoder = TCNEncoder(input_shape=(10, 20), num_tcn_layers=3, latent_dim=10)
        self.decoder = TCNDecoder(latent_dim=10, num_tcn_layers=3)
        self.Normal_dist = torch.distributions.Normal(0, 1) # for reparameterization
        self.loss = 0

    def reparameterize(self, mu, log_var):
        std = torch.exp(log_var/2)
        z = mu + std * self.Normal_dist.sample(mu.shape)
        return z

    def forward(self, x):
        mu, log_var = self.encoder(x)
        z = self.reparameterize(mu, log_var)
        output = self.decoder(z)
        return output, mu, log_var


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])