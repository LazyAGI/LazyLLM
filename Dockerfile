FROM lazyllm/cuda:12.1.0-cudnn8-devel-ubuntu22.04

WORKDIR /tmp

USER root

ENV LANG=C.UTF-8
ENV LC_ALL=C.UTF-8
ENV TZ=Asia/Shanghai

RUN set -ex \
    && ln -sf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone \
    && apt-get update && apt-get install -y openssh-server \
    git vim tzdata curl net-tools locales zip libtinfo5 cmake ffmpeg \
    exuberant-ctags libclang-dev tcl expect telnet rsync libibverbs1 libgl1 \
    && rm -rf /var/lib/apt/lists/* \
    && sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config \
    && service ssh start \
    && echo 'LANG="en_US.UTF-8"' > /etc/default/locale \
    && echo 'LC_ALL="en_US.UTF-8"' >> /etc/default/locale \
    && locale-gen en_US.UTF-8

RUN set -ex \
    && wget https://repo.anaconda.com/miniconda/Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && bash Miniconda3-py310_23.1.0-1-Linux-x86_64.sh -b -p /opt/miniconda3 \
    && rm Miniconda3-py310_23.1.0-1-Linux-x86_64.sh \
    && wget https://packages.redis.io/redis-stack/redis-stack-server-7.2.0-v10.rhel7.x86_64.tar.gz \
    && tar xf redis-stack-server-7.2.0-v10.rhel7.x86_64.tar.gz \
    && chown -R root:root /tmp && chmod 1777 /tmp \
    && mv redis-stack-server-7.2.0-v10 /usr/local/ \
    && rm -rf redis-stack-server-7.2.0-v10.rhel7.x86_64.tar.gz

ENV PATH="/opt/miniconda3/bin:/usr/local/redis-stack-server-7.2.0-v10/bin:${PATH}"

COPY image-build-requirements* /tmp/

RUN conda init bash \
    && conda create -n lazyllm --clone base \
    && echo "source activate lazyllm" > ~/.bashrc

RUN bash -c "source activate lazyllm && \
    conda install -y mpi4py && \
    pip install -r image-build-requirements0.txt --default-timeout=10000 --no-deps && \
    pip install -r image-build-requirements1.txt --default-timeout=10000 --no-deps && \
    pip install -r image-build-requirements2.txt --default-timeout=10000 --no-deps && \
    pip install -r image-build-requirements3.txt --default-timeout=10000 --no-deps && \
    pip install flash-attn==2.7.0.post2 && \
    pip cache purge && rm -rf /tmp/*"

# Fix vllm bug
RUN perl -pi -e 's/parser.add_argument\("--port", type=int, default=8000, ge=1024, le=65535\)/parser.add_argument("--port", type=int, default=8000)/g' /opt/miniconda3/envs/lazyllm/lib/python3.10/site-packages/vllm/entrypoints/api_server.py

ARG LAZYLLM_VERSION=""
ENV LAZYLLM_VERSION=$LAZYLLM_VERSION
RUN bash -c "source activate lazyllm && pip install lazyllm==${LAZYLLM_VERSION}"  \
    && rm -rf /tmp/*

ENTRYPOINT ["/bin/bash"]
WORKDIR /root
