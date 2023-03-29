## Login to BwUniCluster2.0
Preferred login node: `uc2.scc.kit.edu`

```bash
ssh uloak@uc2.scc.kit.edu
```

See [docs.](https://wiki.bwhpc.de/e/BwUniCluster2.0/Login)

## Jupyter 🪐
See [here.](https://uc2-jupyter.scc.kit.edu/)

## Docker 🐳
- https://wiki.bwhpc.de/e/BwUniCluster2.0/Containers
https://github.com/runpod/containers
- https://www.runpod.io/blog/how-to-achieve-true-ssh-on-runpod
- https://wiki.bwhpc.de/e/BwUniCluster2.0/Containers
- Images that should work: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

```shell
docker image build -t first-try . 
docker container run --name first-try
docker run --env-file .env first-try
```


## Add data from Onedrive to GCP
In Google Cloud Platform Console:

f the file is bigger than 4.6 GB you can still do it but you need to mount the bucket in your Cloud Shell using gcsfuse:

Create a directory in your Cloud Shell user home

```bash
mkdir ~/thesis-bucket-option-trade-classification
```

Now mount your bucket in that directory using gcsfuse:

```bash
gcsfuse thesis-bucket-option-trade-classification ~/thesis-bucket-option-trade-classification
```

Change the current directory to mount point directory:

```bash
 cd thesis-bucket-option-trade-classification
```

(if you want to have some fun run "df -h ." to see how much space you got in that mount point)

Now use wget to get the file directly into your bucket (sample using 10GB file off the web):

```bash
wget "https://public.am.files.1drv.com/y4mknFRQq6J1y2ZpJATEx-XFs19E8AsgP43fWyzKkNGOJ_KUIJ-XeVcjArOudVMCMnH_17pw714KTATmf4ZoflJqP8plzoIW79SpLZqZs6ZCeIdXoEVL4-2j47KH7uNDl8cneVZHqcPTQzzR5rMGwfJHYOZEdlnuG2V97xJq7ljKRRA-qsHsMDR9yJbyFzKm5FoifReQ0OvmiNSVedrkudb_FVpp0hpRVCyllKbHDg-vTg" -O data/raw/matched_ise_quotes.csv
```

## Hashes of files

```
2907e9a03f91a202b17e3a5779e90be9d11f8515  matched_ise_quotes.csv
f52c6ba9617ea1516b23b863b46078c6  matched_cboe_quotes.csv
afee3427993806bfb27fdfad9a54521d  livevol_ise_full_sample.csv
```

## Modules
```
compilerfintel/19.1
devel/python/3.8.6 inteL19.1
jupyter/base/2023-03-23
```