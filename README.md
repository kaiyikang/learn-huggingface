# learn-huggingface

The Repo used to learn about huggingface.

# Usage

## 1. Pull Huggingface docker

I use transformers-cpu version.

```bash
docker pull huggingface/transformers-pytorch-cpu
```

## 2. Run container

```bash
docker run -it -v ${pwd}:/workspace huggingface/transformers-pytorch-cpu
```
