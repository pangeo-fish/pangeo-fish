#!/usr/bin/env bash

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
EOF

if ! normalized=$(getopt -o hw:c:p:e: --long help,conda-path:,environment:,workflow-root:,configuration-root:,parametrized-root:,executed-root: -n "run-workflow" -- "$@"); then
    echo "failed to parse arguments" >&2
    exit 1
fi

eval set -- "$normalized"

workflow_root="$(pwd)/notebooks/workflow"
configuration_root="$workflow_root/configuration"
parametrized_root="$workflow_root/parametrized"
executed_root="$workflow_root/executed"

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

find "$workflow_root" -maxdepth 1 -type f -name "0[1-3]_*.ipynb" | sort -h | while read -r notebook; do
    papermill --prepare-only \
              --kernel python3 \
              "$notebook" \
              "$parametrized_root/$conf_id/$(basename "$notebook")" \
              -f "$configuration_root/$conf_id/$(basename "$notebook" .ipynb).yaml"
done

# execute the notebooks
mkdir -p "$executed_root/$conf_id"
find "$parametrized_root/$conf_id" -maxdepth 1 -type f -name "*.ipynb" | sort -h | while read -r notebook; do
    executed_path="$executed_root/$conf_id/$(basename "$notebook")"
    html_path="$(basename "$executed_path" .ipynb).html"
    /usr/bin/time -v jupyter nbconvert --execute --allow-errors \
            "$notebook" \
            --to notebook \
            --output "$executed_path"

    /usr/bin/time -v jupyter nbconvert --allow-errors \
            "$executed_path" \
            --to html \
            --output "$html_path"
done
