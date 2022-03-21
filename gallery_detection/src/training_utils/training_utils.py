import imp, os


def basic_train(network, train_loader, criterion, optimizer, n_epochs, device):
    loss = 1
    for epoch in range(n_epochs):  # loop over the dataset multiple times
        loss_history = []
        print("", end="\r")
        print("Epoch {} out of {}: loss {}".format(
            epoch + 1, n_epochs, loss), end="")
        for i, data in enumerate(train_loader):
            # get the inputs; data is a list of [inputs, labels]
            print("", end="\r")
            print(f"{i}", end="")
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = network(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

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