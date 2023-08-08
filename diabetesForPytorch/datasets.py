from torch.utils.data import Dataset
from sklearn import datasets

from torch import Tensor


class DiabetesDataset(Dataset):
    def __init__(self, is_train: bool=True, normalize: bool=False) -> None:
        self.x, self.y = self.__getDiabets(is_train=is_train)
        
        if normalize:
            self.x = self.__minMax_normalize(self.x)

    def __len__(self) -> int:
        return self.x.__len__()

    def __getitem__(self, index) -> tuple[Tensor, Tensor]:
        return self.x[index], self.y[index]
    
    def __getDiabets(self, is_train: bool) -> tuple[Tensor, Tensor]:
        diabetes = datasets.load_diabetes()
        x, y = Tensor(diabetes.data), Tensor(diabetes.target).view(-1, 1)

        half_idx = int(len(x) * 0.8)
        if is_train:
            return x[:half_idx], y[:half_idx]
        return x[half_idx:], y[half_idx:]

    def __minMax_normalize(self, x) -> Tensor:
        x = Tensor(x)
        _, col = x.shape
        
        for c in range(col):
            c_data = Tensor(x[:, c])
            max_, min_ = c_data.max(), c_data.min()
            x[:, c] = (c_data - min_) / (max_ - min_)
        return x