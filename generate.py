import torch
import sys
data_path = sys.argv[1]

def main():
    model = NetG()
    model.load_state_dict(data_path)
    noise = torch.cuda.FloatTensor(batch_size, args.hidden_dim).normal_(0, 1)
    noise = torch.cuda.FloatTensor(noise)  # total freeze netG
    generated_sample = torch.cuda.FloatTensor(model(noise).data)
    print(generated_sample)

    class NetG(nn.Module):
        def __init__(self, decoder):
            super(NetG, self).__init__()
            self.decoder = decoder  # nn.sequential operator

        def forward(self, input):
            return self.decoder(input)
if __name__ == '__main__':
    main()



