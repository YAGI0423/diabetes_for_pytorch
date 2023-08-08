### 이 저장소(Repository)는 「Pytorch를 위한 Diabetes 데이터셋 구현」에 대한 내용을 다루고 있습니다.

***
작성자: YAGI<br>

최종 수정일: 2023-08-09
***
<br>

***
+ 프로젝트 기간: 2023-08-07 ~ 2023-08-09
***
<br>

***
+ 해당 프로젝트는 Ronald Fisher의 「The use of multiple measurement in taxonomic problem as an example of linear discriminant analysis」(1936)를 바탕으로 하고 있습니다.

> Ronald Fisher. (1936). The use of multiple measurement in taxonomic problem as an example of linear discriminant analysis.
***
<br>

## 프로젝트 요약
&nbsp;&nbsp;
파이토치(Pytorch)의 'Dataset' 형식으로 된 Diabetes 데이터셋을 제공합니다. 기존 파이토치 Dataset과 마찬가지로 DataLoader를 이용하여 순회 가능한 객체(Iterable)를 구현할 수 있습니다.
<br><br>

## Getting Start

### Get Logic Dataset
```python
from torch.utils.data import DataLoader
from diabetesForPytorch.datasets import DiabetesDataset

#is_train: True -> 학습 데이터, False -> 검증 데이터
#normalize: True -> 0 ~ 1
dataset = DiabetesDataset(
    is_train=True,
    normalize=True,
)

#DataLoader
dataLoader = DataLoader(dataset, batch_size=4, shuffle=False)
```
***
<br><br>


## 개발 환경
**Language**

    + Python 3.9.12

    
**Library**

    + pytorch 1.12.0
    + sklearn 1.0.2

<br><br>

## License
This project is licensed under the terms of the [MIT license](https://github.com/YAGI0423/diabetes_for_pytorch/blob/main/LICENSE).