#!/bin/bash  -x

read -r -d "" help <<-EOF
Usage: $0 [options] confname

Configure and run the workflow notebooks.

The configuration files are in YAML format in (configuration-root)/(confname), and the parametrized
notebooks are written to (parametrized-root)/(confname).

Options:
 -h, --help                 display this help
 -e, --environment          activate this environment before running papermill
     --conda-path           path to the conda executable
 -w, --workflow-root        the root of the workflow notebooks
 -c, --configuration-root   the root of the configuration files
 -p, --parametrized-root    the root of the parametrized notebooks
     --executed-root        the root of the executed directories
     --walltime             the walltime to request using qsub
     --memory               the memory to request using qsub
     --queue                the queue to submit the jobs to
EOF

if ! normalized=$(getopt -o hw:c:p:e: --long help,conda-path:,environment:,workflow-root:,configuration-root:,parametrized-root:,executed-root:,walltime:,memory:,queue: -n "run-workflow" -- "$@"); then
    echo "failed to parse arguments" >&2
    exit 1
fi

eval set -- "$normalized"

workflow_root="$(pwd)/notebooks/workflow"
configuration_root="$workflow_root/configuration"
parametrized_root="$workflow_root/parametrized"
#executed_root="$workflow_root/executed"
executed_root="$workflow_root/executed"
walltime="04:00:00"
memory="120GB"
queue="mpi_1"

while true; do
    case "$1" in
        -h|--help)
            echo "$help"
            exit 0
            ;;

        -w|--workflow-root)
            workflow_root="$2"
            shift 2
            ;;

        -c|--configuration-root)
            configuration_root="$2"
            shift 2
            ;;

        -p|--parametrized-root)
            parametrized_root="$2"
            shift 2
            ;;

        --executed-root)
            executed_root="$2"
            shift 2
            ;;

        -e|--environment)
            environment="$2"
            shift 2
            ;;

        --conda-path)
            conda_path="$2"
            shift 2
            ;;

        --queue)
            queue="$2"
            shift 2
            ;;

        --memory)
            memory="$2"
            shift 2
            ;;

        --walltime)
            walltime="$2"
            shift 2
            ;;

        --)
            shift
            break
            ;;

        *)
            echo "invalid option: $1"
            exit 1
            ;;
    esac
done

if [ "$#" != 1 ]; then
    echo "invalid number of arguments: $#"
    exit 1
fi

conf_id="$1"
if [ ! -d "$configuration_root/$conf_id" ]; then
    echo "configuration $configuration_root/$conf_id does not exist"
    exit 2
fi

# conda
if [[ "$environment" != "" && "$conda_path" == "" ]]; then
    echo "need to provide the path to conda when activating a environment";
    exit 3
elif [[ "$environment" != "" && "$conda_path" != "" ]]; then
    conda_root="$(dirname "$(dirname "$conda_path")")"
    # shellcheck source=/dev/null
    source "$conda_root/etc/profile.d/conda.sh"

    conda activate "$environment"
fi

# parametrize the notebooks
mkdir -p "$parametrized_root"
mkdir -p "$parametrized_root/$conf_id"

find "$workflow_root" -maxdepth 1 -type f -name "*.ipynb" | sort -h | while read -r notebook; do
    configuration_path="$configuration_root/$conf_id/$(basename "$notebook" .ipynb).yaml"
    if [ ! -f "$configuration_path" ]; then
        continue
    fi
    papermill --prepare-only \
              --kernel python3 \
              "$notebook" \
              "$parametrized_root/$conf_id/$(basename "$notebook")" \
              -f "$configuration_path"
done

script_dir="$(dirname "$(readlink -f -- "${BASH_SOURCE[0]}")")"


# execute the notebooks
mkdir -p "$executed_root/$conf_id"
find "$parametrized_root/$conf_id" -maxdepth 1 -type f -name "*.ipynb" | sort -h | while read -r notebook; do
    executed_path="$executed_root/$conf_id/$(basename "$notebook")"
    html_path="$(basename "$executed_path" .ipynb).html"

    if which qsub >/dev/null; then
        # automatically use qsub if available
        echo 'do qsub'
        output=$(
            qsub -N "$conf_id" \
                 -W "depend=afterany:$after" \
                 -l "mem=$memory,walltime=$walltime" \
                 -q "$queue" \
                 -- \
                 "$script_dir/execute-notebook.sh" \
                 --conda-path "$conda_path" \
                 --environment "$environment" \
                 "$notebook" \
                 "$executed_path" \
                 "$html_path"
              )
        after=$(echo "$output" | awk '{print $1}')
    else
        "$script_dir/execute-notebook.sh" \
            --conda-path "$conda_path" \
            --environment "$environment" \
            "$notebook" \
            "$executed_path" \
            "$html_path"
    fi
done
