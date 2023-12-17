import os
import sys
import torch
from torch import nn
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision import models
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor, Lambda, Compose
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import math
from re import split
import time
import shutil

"""# Utilities"""

class Logger():
    def __init__(self, log_file_path):
        self.terminal = sys.stdout
        self.log = open(log_file_path, 'a')
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)  

    def flush(self):
        pass
        
class OutputMultiplexer():
    output_mode = 'terminal'
    output_folder = ''
    output_figure_folder = ''
    output_checkpoint_folder = ''
    figure_counter = 1
    checkpoint_counter = 0

    @staticmethod
    def setup_for_file_output():
        OutputMultiplexer.output_mode = 'file'
        base_path = os.path.dirname(os.path.realpath(__file__))
        OutputMultiplexer.output_folder = os.path.join(base_path, time.strftime("%Y-%m-%d_%H%M"))
        os.mkdir(OutputMultiplexer.output_folder)

        # copy python script into output folder
        shutil.copy(__file__, OutputMultiplexer.output_folder)

        # also print to file
        sys.stdout = Logger(os.path.join(OutputMultiplexer.output_folder, 'result.txt'))

        OutputMultiplexer.output_figure_folder = os.path.join(OutputMultiplexer.output_folder, 'figures')
        os.mkdir(OutputMultiplexer.output_figure_folder)

        OutputMultiplexer.output_checkpoint_folder = os.path.join(OutputMultiplexer.output_folder, 'checkpoints')
        os.mkdir(OutputMultiplexer.output_checkpoint_folder)

    @staticmethod
    def show_plt():
        if OutputMultiplexer.output_mode == 'file':
            plt.savefig(os.path.join(OutputMultiplexer.output_figure_folder, str(OutputMultiplexer.figure_counter) + '.svg'))
            plt.clf()
            OutputMultiplexer.figure_counter += 1
        else:
            plt.show()

    @staticmethod
    def get_checkpoint_path():
        OutputMultiplexer.checkpoint_counter += 1
        return os.path.join(OutputMultiplexer.output_checkpoint_folder, str(OutputMultiplexer.checkpoint_counter) + '.tar')

class ModelVisualizer():
    def __init__(self, model, default_module_names = None):
        self.model = model
        self.default_module_names = default_module_names

    def __histogram(self, data, title):
        data_flatten = data.cpu().numpy().flatten()
        non_zero_data = data_flatten[data_flatten != 0]
        if non_zero_data.size > 0:
            plt.hist(non_zero_data, density=False, bins=100)
            plt.ylabel("Count")
            plt.xlabel("Weight")
            plt.title(title)
            OutputMultiplexer.show_plt()

    @staticmethod
    def imshow(image, title):
        plt.imshow(np.transpose(image.cpu().numpy(), (1, 2, 0)))
        plt.title(title)
        OutputMultiplexer.show_plt()

    def histogram_modules(self, module_names, histogram_all_modules = False):
        # uncomment to stop figure plot:
        #return
        for module_name, module in self.model.named_parameters():
            if(module_name in module_names or histogram_all_modules):
                self.__histogram(module.detach(), module_name)
    
    def histogram_default_modules(self):
        if(self.default_module_names == None):
            print("WARNING: default_module_names is None, nothing is plotted.")
        self.histogram_modules(self.default_module_names)

    def get_modules_sparsity(self, module_names, include_all_modules = False):
        sparsity_list = []
        for module_name, module in self.model.named_parameters():
            if(module_name in module_names or include_all_modules):
                sparsity_list.append(torch.count_nonzero(module.detach()) / torch.numel(module.detach()))
        return sparsity_list

    def get_default_modules_sparsity(self):
        if(self.default_module_names == None):
            print("WARNING: default_module_names is None, empty list returned.")
        return self.get_modules_sparsity(self.default_module_names)

    def get_global_sparsity(self, include_all_modules = False):
        if(self.default_module_names == None and not include_all_modules):
            print("WARNING: default_module_names is None")
            return None
        non_zeros = 0
        total = 0
        for module_name, module in self.model.named_parameters():
            if(module_name in self.default_module_names or include_all_modules):
                non_zeros += torch.count_nonzero(module.detach())
                total += torch.numel(module.detach())
        return non_zeros / total

# Not the quickest way, for test and evaluation

class ModuleStructuralIterator():
    def __init__(self, module_shape, structure):
        self.div_filter = min(structure['filter'], module_shape[0])
        self.div_channel = min(structure['channel'], module_shape[1])
        self.div_row = min(structure['row'], module_shape[2])
        self.div_column = min(structure['column'], module_shape[3])

        # Assumption: no reminder remains in any of these divisions.
        self.filter_group_size = module_shape[0] // self.div_filter
        self.channel_group_size = module_shape[1] // self.div_channel
        self.row_group_size = module_shape[2] // self.div_row
        self.column_group_size = module_shape[3] // self.div_column

        self.column_group = -1
        self.row_group = self.channel_group = self.filter_group = 0

        if self.div_filter <= 0 or self.div_channel <= 0 or self.div_row <= 0 or self.div_column <= 0:
            self.div_filter = self.div_channel = self.div_row = self.div_column = 0

    @staticmethod
    def get_output_shape(module_shape, structure):
        output_shape = []
        output_shape.append(min(structure['filter'], module_shape[0]))
        output_shape.append(min(structure['channel'], module_shape[1]))
        output_shape.append(min(structure['row'], module_shape[2]))
        output_shape.append(min(structure['column'], module_shape[3]))
        return output_shape

    def __iter__(self):
        return self

    def __index(self):
        return [
                slice(self.filter_group * self.filter_group_size, (self.filter_group + 1) * self.filter_group_size),
                slice(self.channel_group * self.channel_group_size, (self.channel_group + 1) * self.channel_group_size),
                slice(self.row_group * self.row_group_size, (self.row_group + 1) * self.row_group_size),
                slice(self.column_group * self.column_group_size, (self.column_group + 1) * self.column_group_size)
        ], (self.filter_group, self.channel_group, self.row_group, self.column_group)

    def __next__(self):
        self.column_group += 1
        if(self.column_group < self.div_column):
            return self.__index()
        self.column_group = 0

        self.row_group += 1
        if(self.row_group < self.div_row):
            return self.__index()
        self.row_group = 0

        self.channel_group += 1
        if(self.channel_group < self.div_channel):
            return self.__index()
        self.channel_group = 0

        self.filter_group += 1
        if(self.filter_group < self.div_filter):
            return self.__index()
        raise StopIteration


"""# Model Buffers"""

class ModelBuffers():
    def __init__(self):
        self.weight_mask = {}
        self.pre_weight_mask = {}
        self.structured_weight_mask = {}
        self.pre_structured_weight_mask = {}
        self.quantization_table = {}

    def state_dict(self):
        return {
            'weight_mask' : self.weight_mask,
            'pre_weight_mask' : self.pre_weight_mask,
            'structured_weight_mask' : self.structured_weight_mask,
            'pre_structured_weight_mask' : self.pre_structured_weight_mask,
            'quantization_table' : self.quantization_table
        }

    def load_state_dict(self, state_dict):
        self.weight_mask = state_dict['weight_mask']
        self.pre_weight_mask = state_dict['pre_weight_mask']
        self.structured_weight_mask = state_dict['structured_weight_mask']
        self.pre_structured_weight_mask = state_dict['pre_structured_weight_mask']
        self.quantization_table = state_dict['quantization_table']

"""# Structure"""

class Structure():
    EACH = 1000000000
    ANY = 1

    def __init__(self, structure):
        if type(structure) is list:
            self.structure_list = structure
        elif type(structure) is dict:
            order = ['filter', 'channel', 'row', 'column']
            self.structure_list = []
            for dimension in order:
                if dimension in structure:
                    self.structure_list.append(structure[dimension])
            if len(self.structure_list) != len(structure):
                raise Exception("Only 'filter', 'channel', 'row', 'column' are allowed as dict keys")
            self.structure_dict = structure
        else:
            raise TypeError("Only list and dict are allowed")

    def __getitem__(self, index):
        if type(index) is str:
            return self.structure_dict[index]
        return self.structure_list[index]

    def __len__(self):
        return len(self.structure_list)

    def __iter__(self):
        return iter(self.structure_list)


"""# Adversarial Attack"""

class AdversarialAttack():
    def __init__(self, model, method, epsilon = 8/255, network_input_range = (0, 1), 
                 pgd_iterations = 7, pgd_alpha = 0.01, pgd_random_start = True):
        self.model = model
        self.method = method
        self.epsilon = epsilon
        self.network_input_range = network_input_range

        self.pgd_iterations = pgd_iterations
        self.pgd_alpha = pgd_alpha
        self.pgd_random_start = pgd_random_start

    def perturb(self, network_input, correct_output, method = None):
        if method == None:
            method = self.method

        with torch.enable_grad():
            if method == 'PGD':
                return self.__pgd(network_input, correct_output)
            elif method == 'FGSM':
                return self.__fgsm(network_input, correct_output)
            else:
                raise Exception("Only 'PGD' and 'FGSM' methods are avilable")

    def __fgsm(self, network_input, correct_output):
        initial_training_mode = self.model.training
        self.model.eval()

        perturbed_input = network_input.clone()
        perturbed_input.requires_grad_(True)

        prediction = self.model(perturbed_input)
        loss = F.cross_entropy(prediction, correct_output)

        perturbed_input.grad = None
        self.__zero_model_grads()
        loss.backward()

        with torch.no_grad():
            perturbed_input.add_(perturbed_input.grad.sign(), alpha = self.epsilon)
            perturbed_input.clamp_(self.network_input_range[0], self.network_input_range[1])

        perturbed_input.grad = None
        perturbed_input.requires_grad_(False)

        if initial_training_mode:
            self.model.train()

        return perturbed_input

    def __pgd(self, network_input, correct_output):
        initial_training_mode = self.model.training
        self.model.eval()

        perturbed_input = network_input.clone()
        perturbed_input.requires_grad_(True)

        perturbed_min = torch.clamp(perturbed_input - self.epsilon, self.network_input_range[0], self.network_input_range[1])
        perturbed_max = torch.clamp(perturbed_input + self.epsilon, self.network_input_range[0], self.network_input_range[1])

        if self.pgd_random_start:
            with torch.no_grad():
                perturbed_input.add_(torch.empty_like(perturbed_input).uniform_(-self.epsilon, self.epsilon))
                perturbed_input.clamp_(perturbed_min, perturbed_max)

        for _ in range(self.pgd_iterations):
            prediction = self.model(perturbed_input)
            loss = F.cross_entropy(prediction, correct_output)
            perturbed_input.grad = None
            self.__zero_model_grads()
            loss.backward()

            with torch.no_grad():
                perturbed_input.add_(perturbed_input.grad.sign(), alpha = self.pgd_alpha)
                perturbed_input.clamp_(perturbed_min, perturbed_max)

        perturbed_input.grad = None
        perturbed_input.requires_grad_(False)

        if initial_training_mode:
            self.model.train()

        return perturbed_input

    def __zero_model_grads(self):
        for module_name, module in self.model.named_parameters():
            module.grad = None

"""# ElasticTrain Test Loops"""

class ElasticTestLoops():
    def __init__(
        self, train_dataloader, test_dataloader, model, loss_fn, visualizer, attacker, device, 
        accuracy_top_k=(1,), adversarial_beta = 0.5):
        self.train_dataloader = train_dataloader
        self.test_dataloader = test_dataloader
        self.model = model
        self.loss_fn = loss_fn
        self.visualizer = visualizer
        self.attacker = attacker
        self.device = device
        self.top_k, _ = torch.sort(torch.tensor(list(accuracy_top_k), device = self.device))
        self.adversarial_beta = adversarial_beta
        self.model_buffers = ModelBuffers()

    def load_checkpoint(self, path):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model_buffers.load_state_dict(checkpoint['elastic_trainer_state_dict']['model_buffers_state_dict'])
        return checkpoint['local_dict']

    def __top_k_corrects(self, prediction, answer):
        _, top_predictions = prediction.topk(self.top_k[-1], dim = 1)
        answer_expanded = answer.unsqueeze(dim = -1).expand(top_predictions.shape)

        correct = (top_predictions == answer_expanded).type(torch.float)
        k_correct_count = correct.sum(dim = 0)
        top_k_correct_count = k_correct_count.cumsum(dim = 0)

        return top_k_correct_count[self.top_k - 1]

    def test(self, adversarial = False, method = None, eval_train_data = False):
        dataloader = self.train_dataloader if eval_train_data else self.test_dataloader
        size = len(dataloader.dataset)
        num_batches = len(dataloader)
        self.model.eval()
        test_loss = 0
        top_k_corrects = torch.zeros(self.top_k.shape, device = self.device)
        with torch.no_grad():
            for X, y in dataloader:
                X, y = X.to(self.device), y.to(self.device)

                if adversarial:
                    input = attacker.perturb(X,y, method)
                else:
                    input = X
                
                pred = self.model(input)

                test_loss += self.loss_fn(pred, y).item()
                top_k_corrects.add_(self.__top_k_corrects(pred, y))

        test_loss /= num_batches
        top_k_accuracy = top_k_corrects / size
        print(f"{method} Adversarial Test Error:" if adversarial else "Test Error:")
        print(f"  Avg loss: {test_loss:>8f}")
        accuracies = {}
        for i in range(len(self.top_k)):
            k = self.top_k[i]
            accuracy = top_k_accuracy[i]
            accuracies[k.item()] = accuracy
            print(f"  Top-{k} accuracy: {(100 * accuracy):>0.2f}%")
        return test_loss, accuracies

    def tests(self, adversarial, eval_train_data = False):
        _, accuracies = self.test(eval_train_data = eval_train_data)
        if adversarial:
            self.test(adversarial = True, method = 'FGSM', eval_train_data = eval_train_data)
            self.test(adversarial = True, method = 'PGD', eval_train_data = eval_train_data)
            
        return accuracies


"""# Define Models

DNR VGGNet:
"""

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name, number_of_output_classes, init_weights=True):
        super(VGG, self).__init__()
        self.features, in_channels = self._make_layers(cfg[vgg_name])
        self.classifier = nn.Linear(in_channels, number_of_output_classes)

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers), in_channels

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain = 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(0.5)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

"""ShrinkBench ResNet for CIFAR10:"""

def _weights_init(m):
    classname = m.__class__.__name__
    # print(classname)
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        torch.nn.init.kaiming_normal_(m.weight)

class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)

class BasicBlockCIFAR10(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, option='A'):
        super(BasicBlockCIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            if option == 'A':
                """
                For CIFAR10 ResNet paper uses option A.
                """
                self.shortcut = LambdaLayer(lambda x:
                                            F.pad(x[:, :, ::2, ::2], (0, 0, 0, 0, planes//4, planes//4), "constant", 0))
            elif option == 'B':
                self.shortcut = nn.Sequential(
                     nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                     nn.BatchNorm2d(self.expansion * planes)
                )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNetCIFAR10(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10):
        super(ResNetCIFAR10, self).__init__()
        self.in_planes = 16

        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=2)
        self.linear = nn.Linear(64, num_classes)
        self.linear.is_classifier = True    # So layer is not pruned
        self.apply(_weights_init)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet20(num_classes = 10):
    return ResNetCIFAR10(BasicBlockCIFAR10, [3, 3, 3], num_classes)

def ResNet32(num_classes = 10):
    return ResNetCIFAR10(BasicBlockCIFAR10, [5, 5, 5], num_classes)

def ResNet56(num_classes = 10):
    return ResNetCIFAR10(BasicBlockCIFAR10, [9, 9, 9], num_classes)
    
    
"""DNR ResNet:

[Reference Code](https://github.com/ksouvik52/DNR_ASP_DAC2021/blob/0e3ec30cd548205a7a7186dca4e85e39f15d1b1d/ResNet.py)
"""

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion*planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*planes)

        self.shortcut = nn.Sequential()

        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*planes))

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class ResNet(nn.Module):

    def __init__(self, block, num_blocks, init_weight = True, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.linear = nn.Linear(512*block.expansion, num_classes)

        if init_weight:
            self._initialize_weights()

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []

        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight.data, gain = 0.5)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

def ResNet18(num_classes = 1000):
    return ResNet(BasicBlock, [2,2,2,2], num_classes)

def ResNet34(num_classes = 1000):
    return ResNet(BasicBlock, [3,4,6,3], num_classes)

def ResNet50(num_classes = 1000):
    return ResNet(Bottleneck, [3,4,6,3], num_classes)

def ResNet101(num_classes = 1000):
    return ResNet(Bottleneck, [3,4,23,3], num_classes)

def ResNet152(num_classes = 1000):
    return ResNet(Bottleneck, [3,8,36,3], num_classes)


"""CIFAR10 MobileNet V2:

"""

class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)

        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = out + self.shortcut(x) if self.stride==1 else out
        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(1280)
        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.relu(self.bn2(self.conv2(out)))
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


"""# Load Datasets"""

class DatasetLoaders():
    @staticmethod
    def __setup_dataset(train_augmentations, test_augmentations, dataset, batch_size, number_of_classes, path, image_net = False):
        train_augmentations = transforms.Compose(train_augmentations)
        test_augmentations = transforms.Compose(test_augmentations)
        
        if image_net:
            train_data = dataset(root = path, split = 'train', transform = train_augmentations)
            test_data = dataset(root = path, split = 'val', transform = test_augmentations)
        else:
            train_data = dataset(root = path, train = True, transform = train_augmentations, download = True)
            test_data = dataset(root = path, train = False, transform = test_augmentations, download = True)

        train_dataloader = DataLoader(train_data, batch_size = batch_size, shuffle = True, pin_memory = True, num_workers = 4)
        test_dataloader = DataLoader(test_data, batch_size = batch_size, shuffle = False, pin_memory = True, num_workers = 4)

        return train_dataloader, test_dataloader, number_of_classes

    @staticmethod
    def CIFAR10(batch_size, for_adversarial = False, path = './data'):
        number_of_classes = 10
        mean, std = [0.491, 0.482, 0.447], [0.247, 0.243, 0.262]
        
        train_augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()]

        test_augmentations = [
            transforms.ToTensor()]

        if not for_adversarial:
            train_augmentations.append(transforms.Normalize(mean, std))
            test_augmentations.append(transforms.Normalize(mean, std))

        return DatasetLoaders.__setup_dataset(
            train_augmentations, test_augmentations, datasets.CIFAR10, batch_size, number_of_classes, path)
        
    @staticmethod
    def CIFAR100(batch_size, for_adversarial = False, path = './data'):
        number_of_classes = 100
        mean, std = [0.507, 0.487, 0.441], [0.267, 0.256, 0.276]

        train_augmentations = [
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor()]

        test_augmentations = [
            transforms.ToTensor()]

        if not for_adversarial:
            train_augmentations.append(transforms.Normalize(mean, std))
            test_augmentations.append(transforms.Normalize(mean, std))

        return DatasetLoaders.__setup_dataset(
            train_augmentations, test_augmentations, datasets.CIFAR100, batch_size, number_of_classes, path)
        
    @staticmethod
    def ImageNet(batch_size, for_adversarial = False, path = './data'):
        number_of_classes = 1000
        mean, std = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]

        train_augmentations = [
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()]

        test_augmentations = [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()]

        if not for_adversarial:
            train_augmentations.append(transforms.Normalize(mean, std))
            test_augmentations.append(transforms.Normalize(mean, std))

        return DatasetLoaders.__setup_dataset(
            train_augmentations, test_augmentations, datasets.ImageNet, batch_size, number_of_classes, path, True)


"""# Main"""

torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

# Get CPU or GPU device for training.
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device\n")

# Determine if you plan to train/test for adversarial accuracy
adversarial = False

# Comment this line if you want the outputs/visuals on terminal
OutputMultiplexer.setup_for_file_output()

# Load the train and test data
train_dataloader, test_dataloader, number_of_classes = DatasetLoaders.CIFAR10(
    batch_size = 128, for_adversarial = adversarial, path = 'PATH_TO_TRAIN_TEST_DATASET')

# Load the model
model = VGG('VGG16', number_of_classes).to(device)

print('-------------------------------------------------------')
print(model)
print('-------------------------------------------------------')

set_of_fc_module_names = set()
set_of_conv_module_names = set()
for module_name, module in model.named_modules():
    if isinstance(module, nn.Linear):
        set_of_fc_module_names.add(module_name + '.weight')
    elif isinstance(module, nn.Conv2d):
        set_of_conv_module_names.add(module_name + '.weight')

set_of_module_names = set.union(set_of_fc_module_names, set_of_conv_module_names)

visualizer = ModelVisualizer(model, set_of_module_names)

attacker = AdversarialAttack(model, 'PGD')

tester = ElasticTestLoops(
    train_dataloader, test_dataloader, model, nn.CrossEntropyLoss(), 
    visualizer, attacker, device)


local_dict = tester.load_checkpoint('PATH_TO_TRAINED_MODEL')
print(f"Checkpoint loaded at epoch {local_dict['epoch']}")

print("Testing model accuracy...")
print("##########################################################################")
print("Results:")
# Test model accuracy
tester.tests(adversarial = adversarial)

visualizer.histogram_default_modules()
print("FINAL SPARSITY: ", visualizer.get_default_modules_sparsity())
print("GLOBAL SPARSITY: ", visualizer.get_global_sparsity())

# Trained model statistics

print("##########################################################################")
print("Trained model statistics:")

"""# Global sparsity statistics"""

non_zeros = 0
total = 0
for module_name, module in model.named_parameters():
    if(module_name in set_of_module_names):
        non_zeros = non_zeros + torch.count_nonzero(module)
        total = total + torch.numel(module)

print("Total number of network weights:", total)
print("Number of non-zero network weights:", non_zeros)
print("Compression:", total/non_zeros)


print("##########################################################################")

"""# Workload balancing statistics"""

# You can specify the workload balancing structure here:
conv_workload_balance_structure = {
    'filter' : 16,
    'channel' : 4,
    'row' : Structure.ANY,
    'column' : Structure.ANY
}

print("Workload balancing statistics are mostly meaningful only if the model is workload-balance trained, do you want workload balancing statistics printed(y/N)?")
ans = input()

if (ans.lower() in {'y', 'yes'}):
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print("Module Name:", module_name)
            count = 0
            for index, _ in ModuleStructuralIterator(module.weight.shape, conv_workload_balance_structure):
                workload = module.weight[index].flatten()
                print('Workload #', count, ':', torch.count_nonzero(workload), 'out of', torch.numel(workload), 'remained')
                count += 1
        


print("##########################################################################")

"""# Structured pruning statistics"""

# You can specify the the targeted pruning structure here:
structure = {
    'filter' : Structure.EACH,
    'channel' : Structure.EACH,
    'row' : Structure.ANY,
    'column' : Structure.ANY
    }
    
print("Structured pruning statistics are mostly meaningful only if the model involves structured pruning, do you want structured pruning statistics printed(y/N)?")
ans = input()

if (ans.lower() in {'y', 'yes'}):
    total_count = 0
    total_count_zero = 0
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print("Module Name:", module_name)

            count = 0
            count_zero = 0
            for index, _ in ModuleStructuralIterator(module.weight.shape, structure):
                section = module.weight[index].flatten()
                count += 1
                if torch.count_nonzero(section) == 0:
                    count_zero += 1

            total_count += count
            total_count_zero += count_zero

            print(count_zero, "out of", count, "structures pruned.", "Percentage:", count_zero/count)

    print("----------------------------------------------")
    print(total_count_zero, "out of", total_count, "structures pruned in all convolutional layers.", "Percentage:", total_count_zero / total_count)


print("##########################################################################")

"""# Quantization statistics"""

print("Quantization statistics are mostly meaningful only if the model involves quantization, do you want quantization statistics printed(y/N)?")
ans = input()

if (ans.lower() in {'y', 'yes'}):
    for module_name, module in model.named_parameters():
        if(module_name in set_of_module_names):
            print("Number of unique weight values in", module_name, ":", torch.unique(module[module != 0], sorted = False).numel())


print("##########################################################################")

# You can specify the multi codebook quantization structure here:
conv_workload_balance_structure = {
    'filter' : 16,
    'channel' : 4,
    'row' : Structure.ANY,
    'column' : Structure.ANY
}

print("Multi codebook quantization statistics are mostly meaningful only if the model involves multi codebook quantization, do you want multi codebook quantization statistics printed(y/N)?")
ans = input()

if (ans.lower() in {'y', 'yes'}):
    for module_name, module in model.named_modules():
        if isinstance(module, nn.Conv2d):
            print("Module Name:", module_name)

            count = 0
            for index, _ in ModuleStructuralIterator(module.weight.shape, conv_workload_balance_structure):
                workload = module.weight[index].flatten()
                print('Quantization section #', count, ' contains', torch.unique(workload[workload != 0], sorted = False).numel(), 'unique weight values')
                count += 1


