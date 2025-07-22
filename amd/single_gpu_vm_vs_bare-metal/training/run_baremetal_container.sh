#!/bin/bash
docker run -it \
  --network=host \
  --group-add=video \
  --ipc=host \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device /dev/kfd \
  --device /dev/dri \
  --cpuset-cpus="0-12" \
  --memory="224g" \
  rocm/dev-ubuntu-22.04:6.4-complete /bin/bash
