Also, to get GPU going:

- Install nvidia container toolkit
- Change `/etc/nvidia-container-runtime/config.toml`:
  - Add this line: `no-cgroups = true`
  - I think the existing related line is already commented out
- And then I'm running this from the top of the repo:
```
podman run --hooks-dir tom/container/ --rm -it -v $PWD:/fundiff fundiff bash
```
