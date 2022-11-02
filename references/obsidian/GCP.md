

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


## Hashes
```
2907e9a03f91a202b17e3a5779e90be9d11f8515  matched_ise_quotes.csv
```