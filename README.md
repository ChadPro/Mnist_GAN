# Mnist_GAN
### 0.环境依赖
Python : 2.7.12  
Tensorflow : 1.12.0  
Cuda : 9.0.176  
Cudnn : 7.4.1  

### 1.测试生成器
运行:
```python
python gan_test.py
```
会显示如下图:  
![image](https://github.com/ChadPro/Mnist_GAN/raw/master/pictures/figure_1.png)

### 2.训练生成器、识别器
运行:
```python
python gan_sample.py
```
在tensorboard中的loss如下图:  
![image](https://github.com/ChadPro/Mnist_GAN/raw/master/pictures/loss.png)

### 3.损失函数及原理分析
**首先:**    
```python
# 生成器生成数据
G_logit, G_output = generator(Z)
# 识别器识别来自真实及虚拟的数据
D_real, D_logit_real = discriminator(X)
D_fake, D_logit_fake = discriminator(G_output)
```
其中G_output为回归到-1到1范围，784大小的生成器生成图片数据。  
**然后:**  
```python
# 损失函数
D_loss = - tf.reduce_mean(tf.log(D_real)+tf.log(1. - D_fake))
G_loss = - tf.reduce_mean(tf.log(D_fake))
```
分析，对于识别器，希望D\_real尽可能为1(发现真实数据)，而D_fake尽可能为0(发现生成数据)，即是，我们希望识别器将真实图片输入标记为1，将生成器给的图片标记为0；  
而对于生成器，希望D_fake尽可能为1，即是希望识别器没有发现这是生成的数据。  

**最后:**  
送入优化器中:
```python
D_solver = tf.train.AdamOptimizer().minimize(D_loss, var_list=theta_D)
G_solver = tf.train.AdamOptimizer().minimize(G_loss, var_list=theta_G)
```