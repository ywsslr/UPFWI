## this is a code for training
import torch
from torch import nn
from dataset import load_data
from InversionNet import InversionNet
from BatchSimulation import simulation
from tqdm import tqdm
import warnings
warnings.filterwarnings("ignore", message="At least six grid cells per wavelength is recommended")
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = 'cuda' if torch.cuda.is_available() else 'cpu'


# some hyper parameters to set
batch_size = 16
num_epochs = 20

# prepare the dataset
train_loader = load_data('/data1/lr/FWIDATA/train/', batch_size, shuffle=True)
test_loader = load_data('/data1/lr/FWIDATA/test/', batch_size)

# model
net = InversionNet().to(device)

# optimizer and loss
loss = nn.MSELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=1e-4)

# train
def train_one_epoch(net, loss, optimizer, dataloader, device, epoch):
    net.train()
    print(f"epoch:{epoch}, start training....")
    for i, (seis_data, _) in enumerate(dataloader):
        optimizer.zero_grad()
        seis_data = seis_data.to(device)
        v_hat = net(seis_data)
        seis_data_hat = simulation(v_hat)
        l = loss(seis_data_hat, seis_data)
        l.backward()
        optimizer.step()
        if (i+1) % 10 == 0:
            print('Epoch:[{}/{}], Step:[{}/{}], Loss:{}'
                  .format(epoch+1, num_epochs, i+1, len(dataloader), l.item()))
    
# evaluate
def evaluate(net, loss, dataloader, device):
    net.eval()
    print(f"start evaluate....")
    with torch.no_grad():
        for i, (seis_data, _) in enumerate(dataloader):
            seis_data = seis_data.to(device)
            v_hat = net(seis_data)
            seis_data_hat = simulation(v_hat)
            l = loss(seis_data_hat, seis_data)
            if (i+1) % 10 == 0:
                print('Test batch: [{}/{}], Loss: {}'
                    .format(i+1, len(dataloader), l.item()))

# main
def main():
    for epoch in tqdm(range(num_epochs)):
        train_one_epoch(net, loss, optimizer, train_loader, device, epoch)
        evaluate(net, loss, test_loader, device)

main()