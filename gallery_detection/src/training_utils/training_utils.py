import imp, os
from matplotlib import pyplot as plt
import torch
from torch.utils.tensorboard import SummaryWriter

def basic_train(network, train_loader, criterion, optimizer, n_epochs, device, lr):
    writer = SummaryWriter(log_dir="/home/lorenzo/tensor_board")
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        loss_history = []
        print("", end="\r")
        print("Epoch {} out of {}".format(
            epoch + 1, n_epochs), end="")
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            torch.cuda.empty_cache()
            inputs, labels = data
            inputs = inputs.to(torch.device("cuda"))
            labels = labels.to(torch.device("cuda"))


            # zero the parameter gradients

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            j = i + epoch * train_loader.__len__()
            writer.add_scalar(f"Loss/train/lr_{lr}",loss,j)
    fig = plt.figure(1)
    plt.subplot(211)
    plt.imshow(inputs[0][0].detach().cpu().numpy())
    plt.subplot(212)
    plt.gca().clear()
    plt.plot(outputs[0].detach().cpu().numpy())
    plt.plot(labels[0].detach().cpu().numpy())
    plt.gca().set_xlim(0,360)
    fig.canvas.draw()
    writer.add_figure(f"result/lr{lr}",fig,epoch)

    return loss_history

def load_class(full_class_string):
    """
    Dynamically load a class from a string
    
    >>> klass = load_class("module.submodule.ClassName")
    >>> klass2 = load_class("myfile.Class2")
    """
    class_data = full_class_string.split(".")
    
    module_str = class_data[0]
    class_str = class_data[-1]
    submodules_list = []

    if len(class_data) > 2:
        submodules_list = class_data[1:-1]

    f, filename, description = imp.find_module(module_str)
    module = imp.load_module(module_str, f, filename, description)
    
    # Find each submodule
    for smod in submodules_list:
        path = os.path.dirname(filename) if os.path.isfile(filename) else filename

        f, filename, description = imp.find_module(smod, [path])
        
        # Now we can load the module
        try:
            module = imp.load_module(" ".join(class_data[:-1]), f, filename, description)
        finally:
            if f:
                f.close()

    # Finally, we retrieve the Class
    return getattr(module, class_str)