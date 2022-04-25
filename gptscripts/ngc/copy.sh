#!/bin/bash


# +------------------------+-----------------------+-----------------------+--------------+-----------------------+--------+---------------------+-------+-----------+
# | Id                     | Name                  | Description           | ACE          | Creator Username      | Shared | Created Date        | Owned | Size      |
# | ZydrZx5GQmSSIYDaJmulLw | gpt2-zh               |                       | nv-us-west-2 | Ray Zhang             | No     | 2022-04-22 07:51:06 | Yes   | 0 B       |

# +-------------------+------------+-------------------+-------------------+--------------+--------+------------+-----------+--------------+-------+---------+
# | Id                | Integer Id | Name              | Description       | ACE          | Shared | Size       | Status    | Created Date | Owned | Pre-pop |
# | PT7e0qA9R1icfYUty | 99309      | oscar_zh          |                   | nv-us-west-2 | No     | 78.97 GB   | COMPLETED | 2022-04-21   | Yes   | No      |



IMAGE=nvidia/pytorch:22.03-py3
NAME=oscar-gpt2-pretrain
INSTANCE=cpu.x86.tiny

DATASETID=96766
DATASET=/mount/oscar_zh
WORKSPACE=/mount/gpt2-zh
WORKDIR=${WORKSPACE}/Megatron-LM
DATADIR=${WORKDIR}/data/bert/new2016
EXE=${WORKDIR}/gptscripts/convert_oscar.py

COMMAND="pwd; ls ${DATASET}; cp ${DATASET}/* ${DATADIR}/"

ngc batch run --name ${NAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result /results \
    --preempt RUNONCE \
    --total-runtime 600s \
    --ace nv-us-west-2 --org nvidian --team sae \
    --datasetid ${DATASETID}:${DATASET} \
    --workspace qG7q67EMSma_OzVg98v7-A:${WORKSPACE}:RW



#!/bin/bash
DATE=`date +%m-%d-%yT%H-%M`
# Define my NGC user constants
NGC_CLI_API_KEY="*******************************************"
NGC_CLI_ORG="my_org"
NGC_CLI_ACE="our_fastest_ace"

OUTPUT="--format_type json"
OPTS="${OUTPUT}"

# Define my NGC job constants
NAME="--name our_job-${DATE}"
DATASET="--datasetid 1234:/mnt/huge_dataset"
IMAGE="--image nvidia/tensorflow-test"
INSTANCE="--instance biggest_one"
PREEMPT="--preempt RUNONCE"
RESULT="--result /result"
COMMAND="--commandline bash /mnt/dataset/run_my_job.sh"
JOB_OPTS="${NAME} ${DATASET} ${IMAGE} ${INSTANCE} ${PREEMPT} ${RESULT} ${COMMAND}"

# Pre-declare our commands
declare -a commands=("ngc user who" "ngc batch run ${JOB_OPTS}")

for command in "${commands[@]}"; do
    ${command} ${OPTS}
done