import utils
import argparse
import warnings
import torch
from test_utils import train_model, test_model
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='./data/AF')
parser.add_argument('--batch', type=int, default=15)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--nlayers', type=int, default=2)
parser.add_argument('--emb_size', type=int, default=64)
parser.add_argument('--nhead', type=int, default=8)
parser.add_argument('--task_rate', type=float, default=0.5)
parser.add_argument('--masking_ratio', type=float, default=0.15)
parser.add_argument('--lamb', type=float, default=0.8)
parser.add_argument('--epochs', type=int, default=300)
parser.add_argument('--ratio_highest_attention', type=float, default=0.5)
parser.add_argument('--avg', type=str, default='macro')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--nhid', type=int, default=128)
parser.add_argument('--nhid_task', type=int, default=128)
parser.add_argument('--nhid_tar', type=int, default=128)
parser.add_argument('--task_type', type=str,
                    default='classification', help='[classification, regression]')
args = parser.parse_args()


def main():
    prop = utils.get_prop(args)

    X_train, y_train, X_test, y_test = utils.data_loader(args.dataset)
    
    X_train_task, y_train_task, X_test, y_test = utils.preprocess(prop, X_train, y_train, X_test, y_test)
    print(X_train_task.shape, y_train_task.shape, X_test.shape, y_test.shape)
    
    prop['nclasses'] = torch.max(y_train_task).item(
    ) + 1 if prop['task_type'] == 'classification' else None
    prop['dataset'], prop['seq_len'], prop['input_size'] = prop['dataset'], X_train_task.shape[1], X_train_task.shape[2]
    prop['device'] = torch.device('cuda:0' if torch.cuda.is_available() else "cpu")
    
    model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer = utils.initialize_training(prop)
    
    print('Training start...')
    train_model(model, optimizer, criterion_tar, criterion_task, best_model, best_optimizer, X_train_task, y_train_task, prop)
    print('Training complete...')
    
    test_model(best_model, criterion_task, X_test, y_test, prop)


if __name__ == '__main__':
    main()
