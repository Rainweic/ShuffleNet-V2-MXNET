<!--
 * @Author: your name
 * @Date: 2020-01-31 20:47:01
 * @LastEditTime : 2020-02-02 11:29:44
 * @LastEditors  : Please set LastEditors
 * @Description: In User Settings Edit
 * @FilePath: /ShuffleNet-V2-MXNET/README.md
 -->
# ShuffleNet-V2-MXNET

## install
### cpu
```shell
pip3 install mxnet gluoncv
```
### gpu
```shell
pip3 install mxnet-cu90 gluoncv
```
**cuda9.0->mxnetcu90** <br>
**cuda10.0->mxnetcu100** <br>
**cuda101->mxnetcu101** <br>

## train(gpu): 
```python
python3 train_cifar10.py
```