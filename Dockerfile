# 使用PyTorch官方镜像作为基础镜像，指定版本以获得可复现性
FROM pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime

# 设置容器内的工作目录
WORKDIR /app

# 首先复制依赖列表，利用Docker的缓存机制，只有当依赖改变时才重新安装
COPY requirements.txt .
# 安装Python依赖（使用清华源加速下载）
RUN pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ -r requirements.txt

# 然后将当前目录所有代码复制到容器的/app目录下
COPY . .

# 如果您的代码在运行时需要暴露端口（例如用于TensorBoard），可以声明
# EXPOSE 6006

# 设置容器启动时默认执行的命令，即运行训练脚本
CMD ["python", "train.py"]