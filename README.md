## Training Dense Object Nets on the Entropy cluster

Configure the virtualenv e.g. `mkvirtualenv gdon` and `pip install -r requirements.txt`.

The data is already available on the Entropy server under `/scidatasm/dense_object_nets`.
Alternatively, you can download new data as below:
```bash
python config/download_pdc_data.py config/dense_correspondence/dataset/composite/caterpillar_upright.yaml <full_path_to_data_location>
```

Build the docker image on your local machine:
```bash
cd pytorch-dense-correspondence
git submodule update --init --recursive
cd docker
./docker_build.py
```

Sing up for Docker Hub: https://hub.docker.com/signup.
This will be needed to download docker images on the Entropy cluster.
Login `docker login` on your machine using the created credentials

Then tag and push your docker to Docker Hub:
```
docker tag <docker-id-eg-f69d0749ca1e> <hub-user>/<repo-name>:<tag>
docker push <hub-user>/<repo-name>:<tag>
```

Login into the Entropy cluster `<user_name>@entropy.mimuw.edu.pl`.

Download the docker image using Singularity:
```singularity pull /results/$USER/gdon_latest.sif docker://<hub-user>/<repo-name>```
Comment: Pulling creates a .sif file, which compresses all docker layers to a single SIF file.
This is a heavy file (~4GB). However, we also want to access the SIF file on worker nodes.
Therefore, we need to save the SIF file to a directory that is synced with the worker nodes.
These are `/results` (5GB limit for students) and `/scidatasm` (sync every 10min).
Fow now, the scripts expect the SIF file under `/results/$USER/gdon_latest.sif`.

Upload the code to the Entropy cluster under `/results/$USER/`.
You can do this by setting up SSH Agent Forwarding and cloning this repository from
GitHub, see [this](https://developer.github.com/v3/guides/using-ssh-agent-forwarding/) for more information.
We store the code under `/results` dir because the code also needs to be available for the worker nodes.
Comment: A convenient way is to develop on your local machine (with IDE and git access etc.)
and deploy small incremental changes to the Entropy server.
When using PyCharm, it is very convenient to configure automatic deployment of your changes to the Entropy server.
You can do this under `Tools -> Deployment -> Configureation`.
Select `SFTP` and `OpenSSH config and authentication agent`.

After cloning the repo to the server run on the server:
```bash
cd general-dense-object-nets
git submodule update --init --recursive
```

You then need to add the `.env` file in config with the Neptun setup. See the Logging section below.

Now you are ready to submit your job using
`bash run_batch.sh` from the code directory.
You can see the status of your jobs using `squeue` and the logs under
`/results/$USER/train_gdon_log.txt`.
Alternatively, you can run an interactive job using the `run.sh` script.

You can access the Jupyter and Tensorboard running on the Entropy cluster by setting the tunnel e.g.:
```bash
ssh -N -L 8888:localhost:8888 <user>@entropy.mimuw.edu.pl
```

For additional information refer to the document with the project description:
https://docs.google.com/document/d/1Cq5LK8KdpZXHa9k9BCUp3NHovZnRnwo60e0jzbM_y18/edit?usp=sharing

## AP Loss
In training config file add some metadata for AP Loss and it's sampling strategy. Example:
```yaml
loss_function:
  name: 'aploss'
  nq: 25
  num_samples: 150
  sampler:
    name: 'random' # choice: {'don', 'ring', 'random'}
```

Some sampling strategies require additional params (ex `ring` startegy)
```yaml
loss_function:
  name: 'aploss'
  nq: 25
  num_samples: 150
  sampler:
    name: 'ring' # choice: {'don', 'ring', 'random'}
    inner_radius: 20
    outter_radius: 30
```

or

## Logging
Currently we support logging to Neptune. Sign up here https://neptune.ai/.
Create `config/.env` with the following and paste your key there:
```bash
export NEPTUNE_API_TOKEN="YOUR KEY"
```

In training config file add some metadata for logging. Example:
```yaml
logging:
  backend: 'neptune'
  username: 'tgasior'
  project: 'general-dense-object-nets'
  experiment: 'shoes'
  description: 'This is example description'
  tags: # list as many tags you want. They intend to help you search/filter experiemnts
    - 'general-dense-object-nets'
    - 'tomek'
    - 'aploss'
```

## Original README
### Updates

- September 4, 2018: Tutorial and data now available!  [We have a tutorial now available here](./doc/tutorial_getting_started.md), which walks through step-by-step of getting this repo running.
- June 26, 2019: We have updated the repo to pytorch 1.1 and CUDA 10. For code used for the experiments in the paper see [here](https://github.com/RobotLocomotion/pytorch-dense-correspondence/releases/tag/pytorch-0.3).


## Dense Correspondence Learning in PyTorch

In this project we learn Dense Object Nets, i.e. dense descriptor networks for previously unseen, potentially deformable objects, and potentially classes of objects:

![](./doc/caterpillar_trim.gif)  |  ![](./doc/shoes_trim.gif) | ![](./doc/hats_trim.gif)
:-------------------------:|:-------------------------:|:-------------------------:

We also demonstrate using Dense Object Nets for robotic manipulation tasks:

![](./doc/caterpillar_grasps.gif)  |  ![](./doc/shoe_tongue_grasps.gif)
:-------------------------:|:-------------------------:

### Dense Object Nets: Learning Dense Visual Descriptors by and for Robotic Manipulation

This is the reference implementation for our paper:

[PDF](https://arxiv.org/pdf/1806.08756.pdf) | [Video](https://www.youtube.com/watch?v=L5UW1VapKNE)

[Pete Florence*](http://www.peteflorence.com/), [Lucas Manuelli*](http://lucasmanuelli.com/), [Russ Tedrake](https://groups.csail.mit.edu/locomotion/russt.html)

<em><b>Abstract:</b></em> What is the right object representation for manipulation? We would like robots to visually perceive scenes and learn an understanding of the objects in them that (i) is task-agnostic and can be used as a building block for a variety of manipulation tasks, (ii) is generally applicable to both rigid and non-rigid objects, (iii) takes advantage of the strong priors provided by 3D vision, and (iv) is entirely learned from self-supervision.  This is hard to achieve with previous methods: much recent work in grasping does not extend to grasping specific objects or other tasks, whereas task-specific learning may require many trials to generalize well across object configurations or other tasks.  In this paper we present Dense Object Nets, which build on recent developments in self-supervised dense descriptor learning, as a consistent object representation for visual understanding and manipulation. We demonstrate they can be trained quickly (approximately 20 minutes) for a wide variety of previously unseen and potentially non-rigid objects.  We additionally present novel contributions to enable multi-object descriptor learning, and show that by modifying our training procedure, we can either acquire descriptors which generalize across classes of objects, or descriptors that are distinct for each object instance. Finally, we demonstrate the novel application of learned dense descriptors to robotic manipulation. We demonstrate grasping of specific points on an object across potentially deformed object configurations, and demonstrate using class general descriptors to transfer specific grasps across objects in a class.

#### Citing

If you find this code useful in your work, please consider citing:

```
@article{florencemanuelli2018dense,
  title={Dense Object Nets: Learning Dense Visual Object Descriptors By and For Robotic Manipulation},
  author={Florence, Peter and Manuelli, Lucas and Tedrake, Russ},
  journal={Conference on Robot Learning},
  year={2018}
}
```

### Tutorial

- [getting started with pytorch-dense-correspondence](./doc/tutorial_getting_started.md)

### Code Setup

- [setting up docker image](doc/docker_build_instructions.md)
- [recommended docker workflow ](doc/recommended_workflow.md)

### Dataset

- [data organization](doc/data_organization.md)
- [data pre-processing for a single scene](doc/data_processing_single_scene.md)

### Training and Evaluation
- [training a network](doc/training.md)
- [evaluating a trained network](doc/dcn_evaluation.md)
- [pre-trained models](doc/model_zoo.md)

### Miscellaneous
- [coordinate conventions](doc/coordinate_conventions.md)
- [testing](doc/testing.md)

### Git management

To prevent the repo from growing in size, recommend always "restart and clear outputs" before committing any Jupyter notebooks.  If you'd like to save what your notebook looks like, you can always "download as .html", which is a great way to snapshot the state of that notebook and share.
