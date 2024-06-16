FROM continuumio/miniconda3

WORKDIR /Hack_final

COPY . .
RUN conda env create -f environment.yml

SHELL ["conda", "run", "-n", "myenv", "/bin/bash", "-c"]

ENTRYPOINT ["conda", "run", "--no-capture-output", "-n", "hack_env", "python", "app.py"]