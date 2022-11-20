#!/bin/bash

# Create a netrc file
echo "preparing auth tokens..."

# adapted from: https://jwenz723.medium.com/fetching-private-go-modules-during-docker-build-5b76aa690280
echo -e "\
\napi.wandb.ai\n\
login user\n\
password $WANDB_TOKEN\n\
\n" >> $HOME/.netrc && \
chmod 600 $HOME/.netrc

# Create gcloud credentials file
mkdir -p $HOME/.config/gcloud/ &&\
# https://stackoverflow.com/a/48470187/5755604
JSON_FMT='{"client_id":"%s","client_secret":"%s","refresh_token":"%s","type":"authorized_user"}\n'
printf "$JSON_FMT" "$CLIENT_ID" "$CLIENT_SECRET" "$REFRESH_TOKEN" >> $HOME/.config/gcloud/application_default_credentials.json
chmod 600 $HOME/.config/gcloud/application_default_credentials.json

# try to set up git and clone repo
echo "preparing git..."

git config --global user.name $NAME && \
git config --global user.email $EMAIL &&\
git clone https://$GITHUB_TOKEN@github.com/$REPOSITORY --depth=1&&\
cd thesis
pip install .

echo "pod started..."

if [[ $PUBLIC_KEY ]]
then
    mkdir -p ~/.ssh
    chmod 700 ~/.ssh
    cd ~/.ssh
    echo $PUBLIC_KEY >> authorized_keys
    chmod 700 -R ~/.ssh
    cd /
    service ssh start
fi

if [[ $JUPYTER_PASSWORD ]]
then
    cd /
    jupyter lab --allow-root --no-browser --port=8888 --ip=* --ServerApp.terminado_settings='{"shell_command":["/bin/bash"]}' --ServerApp.token=$JUPYTER_PASSWORD --ServerApp.allow_origin=* --ServerApp.preferred_dir=/workspace --ServerApp.iopub_data_rate_limit=1.0e10
else
    sleep infinity
fi
