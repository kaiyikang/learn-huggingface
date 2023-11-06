# learn-huggingface

The Repo used to learn about huggingface.

# Usage

## 1. Pull Huggingface docker

I use transformers-cpu version.

```bash
docker pull huggingface/transformers-pytorch-cpu
```

## 2. Run container

Run first:

```bash
docker run --name hf -it -v $(pwd):/app huggingface/transformers-pytorch-cpu
```
Restart the container:

```bash
docker start -i hf
```
