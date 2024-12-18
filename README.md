# classification-with-knowledge-distillation
A project demonstrating feature and response based knowledge distillation effectiveness from resnet50 to a 7M parameter network for sematic segmentation tasks.

## setup

1) Setup venv with python 3.12

2) Install ./requirements.txt

  ```sh
  pip install -r ./requirements.txt
  ```

3) Run commands in train.txt to train model

  ```sh
  python .\train.py
  python .\kd_train.py --distil_type response
  python .\kd_train.py --distil_type feature
  python .\kd_train.py --distil_type both
  ```

  Weight paths are saved in each folder.

4) Run commands in test.txt to evaluate model performance

```sh
python .\test.py --weights .\best_model.pth
```
