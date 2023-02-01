

### HPC disk

![image-20221222124540666](/home/yl/.config/Typora/typora-user-images/image-20221222124540666.png)

```shell
/hpctmp/e1097784
/scratch2/e1097784
```

```
singularity exec -e /app1/common/singularity-img/3.0.0/tensorflow_2.9.1_cuda_11.8.0-cudnn8-ubuntu20.04-py38.sif bash
ssh -L 16524:localhost:6006 e1097784@atlas8.nus.edu.sg
```



### PBS basic command

```shell
# Check current task and task id
qstat
# Check history tasks and details
# Check server id
qstat -xfn
# submmit a task
qsub <pbs-file-name>
# delete a tast
qdel <task-id>

```



### Start Jupyter server

```shell
# log on HPC
ssh e1097784@atlas9.nus.edu.sg

# submint the jupyter job
qsub jupyter.pbs

# check server state
qstat -xfn

5140145.venus01 e1097784 volta_g* jupyter     61533   1  20   70gb 72:00 R 00:00
   volta08/0*20

# exit current terminal
exit
# create ssh turnal
ssh -L localhost:8888:volta<08>:8889 e1097784@atlas9.nus.edu.sg
[0.723428571428659, 0.732428571428664, 0.7318571428572351, 0.7320000000000924, 0.7317142857143779, 0.7287142857143762, 0.7327142857143785, 0.7370000000000951, 0.7358571428572374, 0.7357857142858087, 0.7396428571429537, 0.7392857142858107, 0.7450000000000996, 0.7441428571429562, 0.7400714285715254,
```



### Submit job

```shell
# log on HPC
ssh e1097784@atlas9.nus.edu.sg

cd batch_task

# submit the job
qsub lstm_d1.pbs
# Check status
qstat
qstat -xfn
```



### Download python packet

```shell
# log on HPC
ssh e1097784@atlas8.nus.edu.sg
# Enter the tensorflow container
singularity exec /app1/common/singularity-img/3.0.0/tensorflow_2.9.1_cuda_11.8.0-cudnn8-ubuntu20.04-py38.sif bash
# download python packet
pip install xxx

```



### Screen Command

```shell
# Create a new virtual terminal screen
screen -S <screen.name>

# Screen running background
ctrl+a, ctrl+d

# list current screen
screen -ls

# Go back to existing screen
screen -r <screen.name>
screen -r <job.id>

# exit the screen
exit
```



### Bash command

```shell
# current path
pwd

```

## 虚拟环境安装Tensorflow指令

```
environment location: /home/yl/anaconda3/envs/Tensorflow

conda create -n ${env_name} python=3.5


//激活环境
source activate 
conda activate Tensorflow
 
//退出环境
conda deactivate
 
//删除环境
conda remove -n ${env_name} --all
 
//查看conda环境
conda info --env
 
//查看conda的安装包
conda list

```

## 读写一个csv文件

```
 with open('/content/drive/My Drive/FedLearn/Results-LSTM-d1-SGD/lstm-2.csv','ab') as f:
      f.write(open('/content/drive/My Drive/FedLearn/Results-LSTM-d1-SGD/paper-lstm-2.csv','rb').read())
```

## Tensorboard

tf.summary.scalar():收集损失函数和准确率等单值变量

tf.summary.histogram()：收集高维度的变量参数

tf.summary.image()：收集输入的图片张量能显示图片

https://blog.csdn.net/weixin_44503976/article/details/108820547?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-108820547-blog-126455055.pc_relevant_3mothn_strategy_recovery&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7EBlogCommendFromBaidu%7ERate-1-108820547-blog-126455055.pc_relevant_3mothn_strategy_recovery&utm_relevant_index=2

`singularity exec -e /app1/common/singularity-img/3.0.0/tensorflow_2.9.1_cuda_11.8.0-cudnn8-ubuntu20.04-py38.sif bash`

tensorboard --logdir="./hpc/logs" --port=6006

`ssh -L 16524:localhost:6006 e1097784@atlas8.nus.edu.sg`

tensorboard --logdir_spec=run1:"./hpc/logs/log1",run2:"./hpc/logs/log2",run3:"./hpc/logs/log3",run4:"./hpc/logs/log4" --port=6006



当使用.fit_generator生成验证数据时, TensorBoard callback does not create histograms .


### 梯度爆炸问题

https://cloud.tencent.com/developer/article/1661001
https://stackoverflow.com/questions/69427103/gradient-exploding-problem-in-a-graph-neural-network
