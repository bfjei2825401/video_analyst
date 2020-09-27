# Typical command to build a docker image from dockerfile

```bash
docker build -t video_analyst/siamfcpp:vot-eval -f ./Dockerfile ./
# docker run -itd --name vot-eval --network host --ipc host --gpus all -v /home/$USER:/home/$USER video_analyst/siamfcpp:vot-eval
# slz-2080ti conf: add dataset directory mapping
docker run -itd -u 1000 --name vot-eval --network host --ipc host --gpus all -v /home/$USER:/home/$USER -v /mnt/dataset/:/mnt/dataset video_analyst/siamfcpp:vot-eval
# slz708 conf
docker run -itd -u 1000 --name vot-eval --network host --ipc host --gpus all -v /home/$USER:/home/$USER -v /mnt/dataset/:/mnt/dataset -v /mnt/data:/mnt/data video_analyst/siamfcpp:vot-eval
docker exec -it vot-eval bash
```

P.S.
- tmux bind key in container: Ctrl-x (avoid conflict with host)
