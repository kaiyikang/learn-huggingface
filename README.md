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
docker run --name huggingface -it -v ${pwd}:/workspace huggingface/transformers-pytorch-cpu
```

Run again:

```bash
docker start huggingface
```
