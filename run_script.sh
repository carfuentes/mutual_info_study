#Submission options:
#$-S /bin/bash
#$-N sv4
#$-cwd
#$-t 1-250
#$-cwd
export PYTHONPATH=$HOME/.local/lib/python3/site-packages/:$PYTHONPATH

/share/apps/bin/python3 run_grid_search_with_sklearn.py $SGE_TASK_ID
