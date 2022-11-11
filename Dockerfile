ARG BASE_IMAGE=runpod/stack:20.04

FROM ${BASE_IMAGE} as dev-base

SHELL ["/bin/bash", "-o", "pipefail", "-c"]
ENV DEBIAN_FRONTEND noninteractive\
    SHELL=/bin/bash\
    WANDB_TOKEN\
    GITHUB_TOKEN\
    CLIENT_ID\
    CLIENT_SECRET\
    REFRESH_TOKEN\
    NAME\
    EMAIL\
    WORKDIR $HOME

RUN apt-get update --yes && \
    # - apt-get upgrade is run to patch known vulnerabilities in apt-get packages as
    #   the ubuntu base image is rebuilt too seldom sometimes (less than once a month)
    apt-get upgrade --yes && \
    apt install --yes --no-install-recommends\
    wget\
    bash\
    curl\
    git\
    openssh-server\
    unzip &&\
    apt-get clean && rm -rf /var/lib/apt/lists/* && \
    echo "en_US.UTF-8 UTF-8" > /etc/locale.gen

RUN pip install\
    black \
    mypy \
    jupyterlab \
    ipywidgets \
    jupyter-archive

# Install vscode extension
# https://github.com/cdr/code-server/issues/171
RUN mkdir -p $HOME/.vscode/extensions/ && \
    # Naming convention https://${publisher}.gallery.vsassets.io/_apis/public/gallery/publisher/${publisher}/extension/${extensionname}/${version}/assetbyname/Microsoft.VisualStudio.Services.VSIXPackage
    # Install vscode jupyter - required by python extension
    wget --retry-on-http-error=429 --waitretry 15 --tries 5 --no-verbose https://ms-toolsai.gallery.vsassets.io/_apis/public/gallery/publisher/ms-toolsai/extension/jupyter/latest/assetbyname/Microsoft.VisualStudio.Services.VSIXPackage -O ms-toolsai.jupyter-latest.zip && \
    unzip -o ms-toolsai.jupyter-latest.zip extension/* && \
    rm ms-toolsai.jupyter-latest.zip && \
    mv extension $HOME/.vscode/extensions/ms-toolsai.jupyter-latest && \
    # Install vscode python
    wget --no-verbose https://ms-python.gallery.vsassets.io/_apis/public/gallery/publisher/ms-python/extension/python/latest/assetbyname/Microsoft.VisualStudio.Services.VSIXPackage -O ms-python.python-latest.zip && \
    unzip -o ms-python.python-latest.zip extension/* && \
    rm ms-python.python-latest.zip && \
    mv extension $HOME/.vscode/extensions/ms-python.python-latest&& \
    # Install auto docstring
    wget --no-verbose https://njpwerner.gallery.vsassets.io/_apis/public/gallery/publisher/njpwerner/extension/autodocstring/latest/assetbyname/Microsoft.VisualStudio.Services.VSIXPackage -O njpwerner.autodocstring-latest.zip && \
    unzip - njpwerner.autodocstring-latest.zip extension/* && \
    rm njpwerner.autodocstring-latest.zip && \
    mv extension $HOME/.vscode/extensions/njpwerner.autodocstring-latest

ADD start.sh .

RUN chmod +x start.sh

CMD [ "./start.sh" ]

