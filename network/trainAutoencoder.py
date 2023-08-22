from DNNs import Autoenc
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from torch.utils.data.sampler import SubsetRandomSampler


def plot_procedure(train_losses, val_losses):
    plt.plot(train_losses, color='red', label='Training loss')
    plt.plot(val_losses, color='blue', label='Validation loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend(loc='upper right')

    plt.show()


def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)

    autoencoder_model = Autoenc(input_size=1024, hidden_size=200)
    autoencoder_model.to(device)

    data_path = "../data_youcook2/features.pt"
    data = torch.load(data_path)

    no_of_data = data.shape[0]
    train_ratio = .8
    perm = torch.randperm(no_of_data)
    train_indices = perm[:int(no_of_data * train_ratio)]
    val_indices = perm[int(no_of_data * train_ratio):]

    num_epochs = 20
    lr = 0.01
    batch_size = 64

    train_loader = DataLoader(data[train_indices], batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(data[val_indices], batch_size=batch_size)

    optimizer = torch.optim.Adam(autoencoder_model.parameters(), lr=lr, weight_decay=1e-8)

    loss_function = torch.nn.MSELoss()

    train_losses = []
    val_losses = []

    for epoch in range(num_epochs):
        total_loss = 0
        num_batches_train = 0
        num_batches_val = 0
        for i, d in enumerate(train_loader, 0):
            d = d.to(device)
            optimizer.zero_grad()
            output_d, representation = autoencoder_model(d)
            loss = loss_function(output_d, d)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            num_batches_train += 1

        epoch_loss_train = total_loss / num_batches_train
        print(epoch_loss_train)
        train_losses.append(epoch_loss_train)

        val_epoch_loss = 0.0
        with torch.no_grad():
            for j, d in enumerate(val_loader):
                num_batches_val += 1
                d = d.to(device)
                output, _ = autoencoder_model(d)
                loss = loss_function(output, d)
                val_epoch_loss += loss.item()
            epoch_loss_val = val_epoch_loss / num_batches_val
            val_losses.append(epoch_loss_val)

    plot_procedure(train_losses, val_losses)
    return autoencoder_model


def save_representations(trained_model):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_path = "../data_youcook2/features.pt"
    data = torch.load(data_path)
    data = data.to(device)

    _, representation = trained_model(data[:500])
    torch.save(representation, "../data_youcook2/representations.pt")


    print("SAVED")
    pass


if __name__ == '__main__':
    trained_model = train()
    save_representations(trained_model)

