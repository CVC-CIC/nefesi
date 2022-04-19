import torchvision.models as models
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import functools
import numpy as np

activation = {}
def get_activation(name):
    def hook(model, input, output):
        mean_act=torch.mean(output,[0,2,3])
        if not name in activation:
            activation[name] = []
        activation[name].append(mean_act)
    return hook

def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))



def class_selectivity_ML(dictionary):
    all_layers_SI=[]
    for layer in dictionary:
        layer_data= dictionary[layer]

        all_neuron_SI=[]
        for neuron in range(list(layer_data[0].size())[0]):
            mean_act=[x[neuron].item() for x in layer_data]
            maxim=max(mean_act)
            no_max=mean_act
            no_max.pop(np.argmax(mean_act))
            neuron_SI= (maxim - np.mean(no_max))/(maxim + np.mean(no_max)+0.00000001)
            all_neuron_SI.append(neuron_SI)
        all_layers_SI.append(np.mean(all_neuron_SI))
    class_sel=np.mean(all_layers_SI)




    #         SI calculat per a una sola neurona, ara falta ferho per cada neurona de cada capa (2a formula Moroco)




    return class_sel


def data_parallel(module, input, device_ids, output_device=None):
    if not device_ids:
        return module(input)

    if output_device is None:
        output_device = device_ids[0]

    replicas = nn.parallel.replicate(module, device_ids)
    inputs = nn.parallel.scatter(input, device_ids)
    replicas = replicas[:len(inputs)]
    outputs = nn.parallel.parallel_apply(replicas, inputs)
    return nn.parallel.gather(outputs, output_device)


def main():
    print('Positive in each itteration')

    global activation
    # folder_dir ="C:/Users/arias/OneDrive/Escritorio/Nefesi2022/"
    folder_dir = "/home/guillem/Nefesi2022/"
    device = torch.device("cuda" if torch.cuda.is_available()
                          else "cpu")


    model = models.vgg16(pretrained=False)
    # for param in model.parameters():
    #     param.requires_grad = False

    model.classifier[6] = nn.Linear(4096, 200)

    model.to(device)






    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])

    train_folder=folder_dir+"Dataset/tiny-imagenet-200/train"
    trainset = datasets.ImageFolder(root=train_folder, transform=transform)

    trainloader = torch.utils.data.DataLoader(trainset, batch_size=100,
                                              shuffle=True, num_workers=2)


    val_folder=folder_dir+"Dataset/tiny-imagenet-200/val/images"
    testset = datasets.ImageFolder(root=val_folder,  transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=50,
                                             shuffle=False, num_workers=2)


    # Define hyperparameters and settings
    lr = 0.00001  # Learning rate
    num_epochs = 100  # Number of epochs
    log_interval = 100  # Number of iterations before logging
    classreg_interval = 10
    # Set loss function (categorical Cross Entropy Loss)
    loss_func = nn.CrossEntropyLoss()
    factor=1
    # Set optimizer (using Adam as default)
    optimizer = optim.Adam(model.parameters(), lr=lr)



    # register hooks
    hooked_layers=['features.1' ,'features.3',  'features.6','features.8','features.11','features.13','features.15','features.18','features.20','features.22','features.25','features.27','features.29']



    class_sel=0

    for epoch in range(num_epochs):  # loop over the dataset multiple times


        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # get the inputs; data is a list of [inputs, labels]



            # if i % classreg_interval != classreg_interval-1:
            activation = {}
            handles = []
            for layer in hooked_layers:
                output = rgetattr(model, layer)
                handles.append(output.register_forward_hook(get_activation(layer)))

            with torch.set_grad_enabled(False):

                for local_batch, local_labels in testloader:
                    # Transfer to GPU
                    data_parallel(model,local_batch,[0,1])



                    local_batch, local_labels = local_batch.to(device), local_labels.to(device)
                    test_outputs = model(local_batch)

            class_sel = class_selectivity_ML(activation)
            #     clear hooks
            for handle in handles:
                handle.remove()

            inputs, labels = data
            inputs, labels = inputs.cuda(), labels.cuda()

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)


            loss1 = loss_func(outputs, labels)


            loss = loss1 + factor*class_sel
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss1.item()
            if i % log_interval == log_interval-1:    # print every 2000 mini-batches
                print(f'[{epoch + 1}, {i + 1:5d}] loss: {running_loss / 2000:.3f} ')
                running_loss = 0.0

    #     torch.save(model,'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_partial' + str(  epoch))
    # torch.save(model, 'C:/Users/arias/Desktop/Github/nefesi/Model_generation/Savedmodel/vgg16_normal')
        if epoch % 10 == 9:
            torch.save(model, folder_dir+'nefesi/Model_generation/Savedmodel/vgg16_partial_pos'+str(epoch))

    torch.save(model, folder_dir+'nefesi/Model_generation/Savedmodel/VGG16_POSITIVE_each_iteration')


    print('Finished Training')


if __name__ == '__main__':
    main()