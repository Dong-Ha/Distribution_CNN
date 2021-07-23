import argparse
import torch

from model import CNN
from wrapper2 import train_model, predict

def main(configs):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if configs.MODE == 'train':

        model = CNN()
        train_model(model,device=device)

    elif configs.MODE == 'predict':
        model = CNN()
        predict(model,device=device)
    
    else:
        raise ValueError("'--mode' expected 'train', or 'predict', got '{}'".format(configs.MODE))


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode',dest='MODE', type=str, required=True, help="run mode : [train|test]")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    configs = parse_arguments()
    main(configs)