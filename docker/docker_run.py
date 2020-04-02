#!/usr/bin/env python
from __future__ import print_function

import argparse
import os
import socket
import getpass
import yaml

if __name__=="__main__":
    user_name = getpass.getuser()
    default_image_name = user_name + '-pytorch-dense-correspondence'
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--image", type=str, help="(required) name of the image that this container is derived from", default=default_image_name)

    parser.add_argument("-c", "--container", type=str, default="pytorch-container", help="(optional) name of the container")\

    parser.add_argument("-d", "--dry_run", action='store_true', help="(optional) perform a dry_run, print the command that would have been executed but don't execute it.")

    parser.add_argument("-e", "--entrypoint", type=str, default="", help="(optional) thing to run in container")

    parser.add_argument("-p", "--passthrough", type=str, default="", help="(optional) extra string that will be tacked onto the docker run command, allows you to pass extra options. Make sure to put this in quotes and leave a space before the first character")

    args = parser.parse_args()
    print(f"running docker container derived from image {args.image}")
    source_dir = os.path.join(os.getcwd(), "../")
    config_file = os.path.join(source_dir, 'config', 'docker_run_config.yaml')

    print(source_dir)

    image_name = args.image
    home_directory = f'/home/{user_name}'
    dense_correspondence_source_dir = os.path.join(home_directory, 'code')

    cmd = "xhost +local:root \n"
    cmd += "docker run --gpus 1"
    if args.container:
        cmd += f" --name {args.container} "

    # enable graphics
    cmd += " -e DISPLAY -e QT_X11_NO_MITSHM=1 -v /tmp/.X11-unix:/tmp/.X11-unix:rw "

    # mount source
    cmd += f" -v {source_dir}:{home_directory}/code "

    # mount ssh keys
    cmd += f" -v ~/.ssh:{home_directory}/.ssh "

    # mount media
    cmd += " -v /media:/media "

    # mount torch folder (where pytorch standard models (i.e. resnet34) are stored)
    cmd += f" -v ~/.torch:{home_directory}/.torch "

    # login as current user
    cmd += f" --user {user_name} "

    # uncomment below to mount your data volume
    config_yaml = yaml.load(open(config_file))
    host_name = socket.gethostname()

    data_directory = config_yaml[host_name][user_name]['path_to_data_directory']
    cmd += f" -v {data_directory}:{home_directory}/data "

    # expose UDP ports
    cmd += " -p 8888:8888 "
    cmd += " --ipc=host "

    # share host machine network
    cmd += " --network=host "

    cmd += " " + args.passthrough + " "

    cmd += " --privileged -v /dev/bus/usb:/dev/bus/usb " # allow usb access

    cmd += " --rm " # remove the image when you exit


    if args.entrypoint and args.entrypoint != "":
        cmd += f'--entrypoint="{args.entrypoint}" '
    else:
        cmd += "-it "
    cmd += args.image
    cmd_endxhost = "xhost -local:root"

    print("command = \n \n", cmd, "\n", cmd_endxhost)
    print("")

    # build the docker image
    if not args.dry_run:
        print("executing shell command")
        code = os.system(cmd)
        print("Executed with code ", code)
        os.system(cmd_endxhost)
        # Squash return code to 0/1, as
        # Docker's very large return codes
        # were tricking Jenkins' failure
        # detection
        exit(code != 0)
    else:
        print("dry run, not executing command")
        exit(0)
