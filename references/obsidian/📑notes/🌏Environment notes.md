As we aim for reproducability and configuration. We give a detailed description about the environment in which the experiments were conducted.

> We recommend that authors state the name of the software and specific release version they used, and also cite the paper describing the tool. In cases when an updated version of a tool used has been published, it should be cited to ensure that the precise version used is clear. And when there is doubt, an e-mail to the developers could resolve the issue.
 (found in [[@GivingSoftwareIts2019]])

**Data:** Data set versioning using `SHA256` hash. Data is loaded by key using `wandb` library.

**Hardware:** Training and inference of our models is performed on the bwHPC cluster. Each node features ... cpus, ... gpus with cuda version ... and Ubuntu ... . If runpods (https://www.runpod.io/gpu-instance/pricing) or lambdalabs (https://lambdalabs.com/service/gpu-cloud) is used, cite it as well. If I use my docker image cite as well. Model training of gradient boosting approach and transformers is performed on accelerates. To guarantee deterministic behaviour (note gradient boosting may operate on samples, initializations of weights happens randomly, cuda runtime may perform non-deterministic optimizations) we fix the random 


**Log:**
```shell
[uloak@uc2n513 uloak]$ lsb_release -a
LSB Version:    :core-4.1-amd64:core-4.1-noarch:cxx-4.1-amd64:cxx-4.1-noarch:desktop-4.1-amd64:desktop-4.1-noarch:languages-4.1-amd64:languages-4.1-noarch:printing-4.1-amd64:printing-4.1-noarch
Distributor ID: RedHatEnterprise
Description:    Red Hat Enterprise Linux release 8.4 (Ootpa)
Release:        8.4
Codename:       Ootpa
[uloak@uc2n513 uloak]$ cat /proc/cpuinfo  | grep 'name'| uniq
model name      : Intel(R) Xeon(R) Gold 6230 CPU @ 2.10GHz
[uloak@uc2n513 uloak]$ cat /proc/cpuinfo | grep processor | wc -l
80
[uloak@uc2n513 uloak]$ free -m
              total        used        free      shared  buff/cache   available
Mem:         386612       26313      312004       22058       48294      336166
Swap:          7629         573        7056
[uloak@uc2n513 uloak]$ free -h
              total        used        free      shared  buff/cache   available
Mem:          377Gi        24Gi       313Gi        17Gi        39Gi       333Gi
Swap:         7.5Gi       573Mi       6.9Gi
[uloak@uc2n513 uloak]$ nvcii
bash: nvcii: command not found...
[uloak@uc2n513 uloak]$ lspci | grep VGA
01:00.1 VGA compatible controller: Matrox Electronics Systems Ltd. MGA G200eH3 (rev 02)
[uloak@uc2n513 uloak]$ lshw -C video
WARNING: you should run this programme as super-user.
  *-display                 
       description: VGA compatible controller
       product: MGA G200eH3
       vendor: Matrox Electronics Systems Ltd.
       physical id: 0.1
       bus info: pci@0000:01:00.1
       logical name: /dev/fb0
       version: 02
       width: 32 bits
       clock: 33MHz
       capabilities: vga_controller bus_master cap_list rom fb
       configuration: depth=32 driver=mgag200 latency=0 resolution=1024,768
       resources: irq:17 memory:d8000000-d8ffffff memory:d9b98000-d9b9bfff memory:d9000000-d97fffff memory:c0000-dffff
  *-display
       description: 3D controller
       product: GV100GL [Tesla V100 SXM2 32GB]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:3a:00.0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: bus_master cap_list
       configuration: driver=nvidia latency=0
       resources: iomemory:cb00-caff iomemory:cb80-cb7f irq:378 memory:df000000-dfffffff memory:cb000000000-cb7ffffffff memory:cb800000000-cb801ffffff
  *-display
       description: 3D controller
       product: GV100GL [Tesla V100 SXM2 32GB]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:3b:00.0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: bus_master cap_list
       configuration: driver=nvidia latency=0
       resources: iomemory:ca00-c9ff iomemory:ca80-ca7f irq:399 memory:de000000-deffffff memory:ca000000000-ca7ffffffff memory:ca800000000-ca801ffffff
  *-display
       description: 3D controller
       product: GV100GL [Tesla V100 SXM2 32GB]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:b2:00.0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: bus_master cap_list
       configuration: driver=nvidia latency=0
       resources: iomemory:db00-daff iomemory:db80-db7f irq:400 memory:ef000000-efffffff memory:db000000000-db7ffffffff memory:db800000000-db801ffffff
  *-display
       description: 3D controller
       product: GV100GL [Tesla V100 SXM2 32GB]
       vendor: NVIDIA Corporation
       physical id: 0
       bus info: pci@0000:b3:00.0
       version: a1
       width: 64 bits
       clock: 33MHz
       capabilities: bus_master cap_list
       configuration: driver=nvidia latency=0
       resources: iomemory:da00-d9ff iomemory:da80-da7f irq:401 memory:ee000000-eeffffff memory:da000000000-da7ffffffff memory:da800000000-da801ffffff
WARNING: output may be incomplete or inaccurate, you should run this programme as super-user.
```


**Jupyter Log:** 🪐
```
JUPYTERHUB_OAUTH_CLIENT_ALLOWED_SCOPES=[]
JUPYTERHUB_CLIENT_ID=jupyterhub-user-uloak
SLURM_NODEID=0
SLURM_TASK_PID=118378
__LMOD_REF_COUNT_PATH=/opt/bwhpc/common/jupyter/base/2022-03-30/bin:1;/sbin:1;/bin:1;/usr/sbin:1;/usr/bin:1
_ModuleTable002_=MDAwMDAzLip6ZmluYWwtLjAwMDAwMDAzMC4qemZpbmFsIiwKfSwKfSwKbXBhdGhBID0gewoiL29wdC9id2hwYy9raXQvbW9kdWxlZmlsZXMiLCAiL29wdC9id2hwYy9jb21tb24vbW9kdWxlZmlsZXMvQ29yZSIsCn0sCnN5c3RlbUJhc2VNUEFUSCA9ICIvb3B0L2J3aHBjL2tpdC9tb2R1bGVmaWxlczovb3B0L2J3aHBjL2NvbW1vbi9tb2R1bGVmaWxlcy9Db3JlIiwKfQo=
SLURM_PRIO_PROCESS=0
LMOD_FAMILY_JUPYTER_VERSION=2022-03-30
MODULES_RUN_QUARANTINE=LD_LIBRARY_PATH LD_PRELOAD
LANG=en_US.UTF-8
LMOD_SYSTEM_NAME=uc2
SLURM_SUBMIT_DIR=/opt/jupyterhub/3.0.0
HISTTIMEFORMAT=%F %T 
HOSTNAME=uc2n513
SLURM_CPUS_PER_TASK=10
ENVIRONMENT=BATCH
ROCR_VISIBLE_DEVICES=0
SLURM_PROCID=0
SLURM_JOB_GID=59900
JUPYTERHUB_OAUTH_ACCESS_SCOPES=["access:servers!server=uloak/", "access:servers!user=uloak"]
SLURMD_NODENAME=uc2n513
SLURM_TASKS_PER_NODE=1
__LMOD_REF_COUNT_PYTHONPATH=/home/kit/stud/uloak/.local/lib/python3.8/site-packages:1;/opt/bwhpc/common/jupyter/base/2022-03-30/lib/python3.8/site-packages:1
SLURM_JOB_RESERVATION=juypter_weekday_eeTh5
JUPYTERHUB_ACTIVITY_URL=http://10.0.3.226:8081/jhub/hub/api/users/uloak/activity
XDG_SESSION_ID=c3922
MODULES_CMD=/usr/share/Modules/libexec/modulecmd.tcl
JUPYTERHUB_SERVICE_URL=http://0.0.0.0:0/jhub/user/uloak/
SLURM_NNODES=1
USER=uloak
JUPYTERHUB_BASE_URL=/jhub/
SLURM_GET_USER_ENV=1
__LMOD_REF_COUNT_MODULEPATH=/opt/bwhpc/kit/modulefiles:1;/opt/bwhpc/common/modulefiles/Core:1
PWD=/pfs/data5/home/kit/stud/uloak
SLURM_JOB_NODELIST=uc2n513
HOME=/home/kit/stud/uloak
SLURM_CLUSTER_NAME=uc2
TMP=/scratch/slurm_tmpdir/job_21895975
SLURM_NODELIST=uc2n513
SLURM_GPUS_ON_NODE=1
LMOD_VERSION=8.6.16
LMOD_PAGER=less
JUPYTERHUB_USER=uloak
SLURM_JOB_CPUS_PER_NODE=10
XDG_DATA_DIRS=/home/kit/stud/uloak/.local/share/flatpak/exports/share:/var/lib/flatpak/exports/share:/usr/local/share:/usr/share
SLURM_TOPOLOGY_ADDR=VirtualInterconnnect.uc2ibbr.uc2ibbe14.uc2n513
SLURM_WORKING_CLUSTER=uc2:uc2n990:6817:9728:109
MKL_NUM_THREADS=1
SLURM_JOB_NAME=jupyterhub-spawner
TMPDIR=/scratch/slurm_tmpdir/job_21895975
SLURM_JOB_GPUS=0
LMOD_sys=Linux
SLURM_JOBID=21895975
_ModuleTable001_=X01vZHVsZVRhYmxlXyA9IHsKTVR2ZXJzaW9uID0gMywKY19yZWJ1aWxkVGltZSA9IGZhbHNlLApjX3Nob3J0VGltZSA9IGZhbHNlLApkZXB0aFQgPSB7fSwKZmFtaWx5ID0gewpqdXB5dGVyID0gImp1cHl0ZXIvYmFzZSIsCn0sCm1UID0gewpbImp1cHl0ZXIvYmFzZSJdID0gewpmbiA9ICIvb3B0L2J3aHBjL2NvbW1vbi9tb2R1bGVmaWxlcy9Db3JlL2p1cHl0ZXIvYmFzZS8yMDIyLTAzLTMwLmx1YSIsCmZ1bGxOYW1lID0gImp1cHl0ZXIvYmFzZS8yMDIyLTAzLTMwIiwKbG9hZE9yZGVyID0gMSwKcHJvcFQgPSB7fSwKc3RhY2tEZXB0aCA9IDAsCnN0YXR1cyA9ICJhY3RpdmUiLAp1c2VyTmFtZSA9ICJqdXB5dGVyL2Jhc2UiLAp3ViA9ICIwMDAwMDIwMjIuKnpmaW5hbC0uMDAw
SLURM_CONF=/etc/slurm/slurm.conf
LOADEDMODULES=jupyter/base/2022-03-30
JUPYTERHUB_OAUTH_SCOPES=["access:servers!server=uloak/", "access:servers!user=uloak"]
SCRATCH=/scratch
JUPYTERHUB_SERVICE_PREFIX=/jhub/user/uloak/
SLURM_NODE_ALIASES=(null)
SLURM_JOB_QOS=normal
LMOD_ROOT=/opt/lmod
SLURM_TOPOLOGY_ADDR_PATTERN=switch.switch.switch.node
MAIL=/var/spool/mail/uloak
JUPYTERHUB_SERVER_NAME=
ZE_AFFINITY_MASK=0
SLURM_CPUS_ON_NODE=10
SLURM_JOB_NUM_NODES=1
SLURM_MEM_PER_NODE=92160
SHELL=/bin/bash
LMOD_SITE_NAME=KIT
_ModuleTable_Sz_=2
SLURM_JOB_UID=365250
SLURM_JOB_PARTITION=gpu_4
JUPYTERHUB_DEFAULT_URL=/lab
SLURM_SCRIPT_CONTEXT=prologue_task
LSDF=/lsdf
TMOUT=36000
KIT_FAMILY_JUPYTER_VERSION=2022-03-30
SLURM_JOB_USER=uloak
CUDA_VISIBLE_DEVICES=0
JUPYTERHUB_API_URL=http://10.0.3.226:8081/jhub/hub/api
LMOD_FAMILY_JUPYTER=jupyter/base
SHLVL=2
SLURM_SUBMIT_HOST=uc2n994.localdomain
PYTHONPATH=/home/kit/stud/uloak/.local/lib/python3.8/site-packages:/opt/bwhpc/common/jupyter/base/2022-03-30/lib/python3.8/site-packages
SLURM_JOB_ACCOUNT=kit
JUPYTERHUB_HOST=
MANPATH=/opt/lmod/lmod/share/man:
SLURM_EXPORT_ENV=PATH,VIRTUAL_ENV,LANG,JUPYTERHUB_API_TOKEN,JPY_API_TOKEN,JUPYTERHUB_CLIENT_ID,JUPYTERHUB_HOST,JUPYTERHUB_OAUTH_CALLBACK_URL,JUPYTERHUB_OAUTH_SCOPES,JUPYTERHUB_OAUTH_ACCESS_SCOPES,JUPYTERHUB_OAUTH_CLIENT_ALLOWED_SCOPES,JUPYTERHUB_USER,JUPYTERHUB_SERVER_NAME,JUPYTERHUB_API_URL,JUPYTERHUB_ACTIVITY_URL,JUPYTERHUB_BASE_URL,JUPYTERHUB_SERVICE_PREFIX,JUPYTERHUB_SERVICE_URL,JUPYTERHUB_DEFAULT_URL,USER,HOME,SHELL
JPY_API_TOKEN=067372d099b546149e800ba14cf695da
TEMP=/scratch
MODULEPATH=/opt/bwhpc/kit/modulefiles:/opt/bwhpc/common/modulefiles/Core
SLURM_GTIDS=0
LOGNAME=uloak
DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/365250/bus
CLUSTER=uc2
JUPYTERHUB_OAUTH_CALLBACK_URL=/jhub/user/uloak/oauth_callback
XDG_RUNTIME_DIR=/run/user/365250
MODULEPATH_modshare=/usr/share/modulefiles:1:/usr/share/Modules/modulefiles:1:/etc/modulefiles:1
MODULEPATH_ROOT=/opt/modulefiles
LMOD_PACKAGE_PATH=/etc/lmod/?.lua;;
JUPYTERHUB_API_TOKEN=067372d099b546149e800ba14cf695da
KIT_FAMILY_JUPYTER=jupyter/base
PATH=/opt/bwhpc/common/jupyter/base/2022-03-30/bin:/sbin:/bin:/usr/sbin:/usr/bin
SLURM_JOB_ID=21895975
_LMFILES_=/opt/bwhpc/common/modulefiles/Core/jupyter/base/2022-03-30.lua
MODULESHOME=/opt/lmod/lmod
LMOD_SETTARG_FULL_SUPPORT=no
HISTSIZE=1000
LMOD_PKG=/opt/lmod/lmod
LMOD_CMD=/opt/lmod/lmod/libexec/lmod
SLURM_LOCALID=0
GPU_DEVICE_ORDINAL=0
LESSOPEN=||/usr/bin/lesspipe.sh %s
OMP_NUM_THREADS=1
LMOD_DIR=/opt/lmod/lmod/libexec
```
``

