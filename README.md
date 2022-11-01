# Learning of Long-Horizon Sparse-Reward Robotic Manipulator Tasks with Base Controllers
-----------------------
This is the official implementation of our TNNLS paper:

"Learning of Long-Horizon Sparse-Reward Robotic Manipulator Tasks with Base Controllers"

Guangming Wang, Minjian Xin, Wenhua Wu, Zhe Liu, and Hesheng Wang

![DDPGwb](https://user-images.githubusercontent.com/62023129/199145782-19b0d1ab-448e-4598-a185-6187b115ef0d.png)


## Prerequisites
------------------------------
- Python 3.6.9
- PyTorch 1.10.1
- CUDA 10.2
- pybullet

## Usage
----------------------
### Train
Train the complete algorithm with state-input. `-t` is the training task. `-l` is the saving log.
```
python learn.py -q -b -c -t 1 -l 1
```
Train the complete algorithm with image-input.
```
python learn.py -q -b -c -i
```
Train with ensemble Base Controllers.
```
python learn.py -q -b -c -e
```

### test
Test the model in log `-l` of task `-t`.
```
python test.py -t 1 -l 1 
```

### plot
plot the learning curve for a set of training.
```
python plot.py
```
## Citation
--------------------------------
If you find our work useful in your research, please cite:

@article{wang2022learning,
  title={Learning of Long-Horizon Sparse-Reward Robotic Manipulator Tasks With Base Controllers},
  author={Wang, Guangming and Xin, Minjian and Wu, Wenhua and Liu, Zhe and Wang, Hesheng},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2022},
  publisher={IEEE}
}
