#!/bin/bash
# FILE: wrapper
#
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -q all.q
#$ -N Mortezajob
#$ -M mv19977@essex.ac.uk
#$ -m be
#$ -o $HOME/output_GIA.txt


python36 ./GIA.py