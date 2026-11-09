#!/bin/bash
# Tree identity of the image and of each arm (no timing): every tracked file of the
# expected tree, as the container sees it, and the change markers each arm imports.
B=$HOME/spark-lab/agents/candidate-51-own-marks; OM=$B/bin/om.sh
$OM common verify-tree-img img -- python3 /h/verify_tree.py /h/c50-d5778ab5.txt
$OM common verify-tree-base base -- python3 /h/verify_tree.py /h/base-14c5a2dc.txt
$OM common verify-tree-c51 c51 -- python3 /h/verify_tree.py /h/c51-987ba058.txt
$OM common verify-overlay-base base -- python3 /h/verify_overlay.py
$OM common verify-overlay-c51 c51 -- python3 /h/verify_overlay.py
