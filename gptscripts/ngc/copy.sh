#!/bin/bash


# +------------------------+-----------------------+-----------------------+--------------+-----------------------+--------+---------------------+-------+-----------+
# | Id                     | Name                  | Description           | ACE          | Creator Username      | Shared | Created Date        | Owned | Size      |
# | ZydrZx5GQmSSIYDaJmulLw | gpt2-zh               |                       | nv-us-west-2 | Ray Zhang             | No     | 2022-04-22 07:51:06 | Yes   | 0 B       |

# +-------------------+------------+-------------------+-------------------+--------------+--------+------------+-----------+--------------+-------+---------+
# | Id                | Integer Id | Name              | Description       | ACE          | Shared | Size       | Status    | Created Date | Owned | Pre-pop |
# | PT7e0qA9R1icfYUty | 99309      | oscar_zh          |                   | nv-us-west-2 | No     | 78.97 GB   | COMPLETED | 2022-04-21   | Yes   | No      |



IMAGE=nvidia/pytorch:22.03-py3
# NAME=ml-model.notamodel-oscar-gpt2-zh.exempt-tc-gpu 
NAME=ml-model.notamodel-convert-oscar.exempt-tc-gpu 
INSTANCE=cpu.x86.tiny

# DATASETID=99309
# DATASET=/mnt/oscar_zh
WORKSPACE=/mnt/gpt2-zh
WORKDIR=${WORKSPACE}/Megatron-LM
DATADIR=${WORKDIR}/data/oscar
# EXE=${WORKDIR}/gptscripts/convert_oscar.py

# COMMAND="pwd; ls ${DATASET};rm -rf ${DATADIR}; mkdir ${DATADIR}; cp -rf ${DATASET}/* ${DATADIR}/ ; sleep 3600"
# COMMAND="sleep 2400"
# --datasetid ${DATASETID}:${DATASET} \
COMMAND="cd ${WORKDIR}; python ./gptscripts/convert_oscar.py --input_path ./data/oscar/ --output_path ./data/; sleep 3600"
ngc batch run --name ${NAME} --image ${IMAGE} --instance ${INSTANCE} --commandline "${COMMAND}" --result /results \
    --preempt RUNONCE \
    --total-runtime 21600s \
    --ace nv-us-west-2 --org nvidian --team sae \
    --workspace ZydrZx5GQmSSIYDaJmulLw:${WORKSPACE}:RW \
