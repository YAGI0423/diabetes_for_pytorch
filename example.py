from torch.utils.data import DataLoader
from diabetesForPytorch.datasets import DiabetesDataset


if __name__ == '__main__':
    dataLoader = DataLoader(
        DiabetesDataset(
            is_train=True,
            normalize=True,
        ),
        batch_size=4,
        shuffle=True,
    )

    batch_sample = next(iter(dataLoader))
    
    print('\n\n')
    print(f'< SAMPLES >'.center(100, '='))
    for x, y in zip(*batch_sample):
        tuple(print(f'{v:.3f}', end='\t') for v in x.tolist())
        print(f'|\t{y.item()}')
    print(f'=' * 100)
        