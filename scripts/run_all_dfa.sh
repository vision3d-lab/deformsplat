#!/bin/bash
SECONDS=0
set -e        # exit when error
set -o xtrace # print command

GPU=0

#                           GPU     object                      idx_from idx_to cam_idx 
bash scripts/deform_dfa.sh ${GPU}   "beagle_dog(s1)"            520      525    16      &&
bash scripts/deform_dfa.sh ${GPU}   "beagle_dog(s1_24fps)"      190      215    32      &&
bash scripts/deform_dfa.sh ${GPU}   "wolf(Howling)"             10       60     24      &&
bash scripts/deform_dfa.sh ${GPU}   "bear(walk)"                110      140    16      &&
bash scripts/deform_dfa.sh ${GPU}   "cat(run)"                  25       30     32      &&
bash scripts/deform_dfa.sh ${GPU}   "cat(walk_final)"           10       20     32      &&
bash scripts/deform_dfa.sh ${GPU}   "wolf(Run)"                 20       25     16      &&
bash scripts/deform_dfa.sh ${GPU}   "cat(walkprogressive_noz)"  25       30     32      &&
bash scripts/deform_dfa.sh ${GPU}   "duck(eat_grass)"           5        15     32      &&
bash scripts/deform_dfa.sh ${GPU}   "duck(swim)"                145      160    16      &&
bash scripts/deform_dfa.sh ${GPU}   "whiteTiger(roaringwalk)"   15       25     32      &&
bash scripts/deform_dfa.sh ${GPU}   "fox(attitude)"             95       145    24      &&
bash scripts/deform_dfa.sh ${GPU}   "wolf(Walk)"                85       95     16      &&
bash scripts/deform_dfa.sh ${GPU}   "fox(walk)"                 70       75     24      &&
bash scripts/deform_dfa.sh ${GPU}   "panda(walk)"               15       25     32      &&
bash scripts/deform_dfa.sh ${GPU}   "lion(Walk)"                30       35     32      &&
bash scripts/deform_dfa.sh ${GPU}   "panda(acting)"             95       100    32      &&
bash scripts/deform_dfa.sh ${GPU}   "panda(run)"                5        10     32      &&
bash scripts/deform_dfa.sh ${GPU}   "lion(Run)"                 50       55     24      &&
bash scripts/deform_dfa.sh ${GPU}   "duck(walk)"                200      230    16      &&
bash scripts/deform_dfa.sh ${GPU}   "whiteTiger(run)"           25       70     32      &&
bash scripts/deform_dfa.sh ${GPU}   "wolf(Damage)"              0        110    32      &&
bash scripts/deform_dfa.sh ${GPU}   "cat(walksniff)"            70       150    32      &&
bash scripts/deform_dfa.sh ${GPU}   "bear(run)"                 0        2      16      &&
bash scripts/deform_dfa.sh ${GPU}   "fox(run)"                  25       30     32
