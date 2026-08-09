# Select base image (can be ubuntu, python, shiny etc)
FROM python:3.11-slim

# Create user name and home directory variables.
# The variables are later used as $USER and $HOME.
ENV USER=username
ENV HOME=/home/$USER

# Add user to system. The app directory is created owned by that user so nothing
# below needs a recursive chown: `chown -R` after copying the models would make
# overlayfs copy every one of them into a second layer, doubling their weight in
# the image.
RUN useradd -m -u 1000 $USER \
    && mkdir -p $HOME/app \
    && chown $USER:$USER $HOME/app

# Set working directory (this is where the code should go)
WORKDIR $HOME/app

# Dependencies first. This is by far the most expensive layer — torch alone is
# most of the image — so it sits above the source and the model weights, and
# editing either no longer reinstalls it. No compiler is installed because every
# dependency ships a prebuilt wheel; verify with
#     pip install --dry-run --only-binary=:all: -r requirements.txt
# before adding one back.
COPY --chown=$USER:$USER requirements.txt $HOME/app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Application code, then the model weights last: they are the largest thing in
# the image and the least often edited.
COPY --chown=$USER:$USER pyproject.toml $HOME/app/pyproject.toml
COPY --chown=$USER:$USER main.py $HOME/app/main.py
COPY --chown=$USER:$USER model.py $HOME/app/model.py
COPY --chown=$USER:$USER session.py $HOME/app/session.py
COPY --chown=$USER:$USER viewer.py $HOME/app/viewer.py
COPY --chown=$USER:$USER preprocessing.py $HOME/app/preprocessing.py
COPY --chown=$USER:$USER architectures.py $HOME/app/architectures.py
COPY --chown=$USER:$USER checkpoints.py $HOME/app/checkpoints.py
COPY --chown=$USER:$USER data/ $HOME/app/data

USER $USER

EXPOSE 7860
ENV GRADIO_SERVER_NAME="0.0.0.0"
ENV GRADIO_ANALYTICS_ENABLED=false
ENV OMP_NUM_THREADS=4

CMD ["python", "main.py"]
