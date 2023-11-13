# learn-huggingface

The Repo used to learn about Huggingface.
Source: https://huggingface.co/learn/nlp-course

# Usage

## 1. Pull Huggingface docker

I use transformers-cpu version.

```bash
docker pull huggingface/transformers-pytorch-cpu
```

## 2. Run Dockerfile

```bash
# Build Image
docker build --tag huggingface .

# Run the huggingface
docker run --name hf -it -v "$(pwd)":/app huggingface

# Re/start the container
docker start -i hf
```****