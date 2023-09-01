#!/bin/bash -x
#bash
#source .bashrc

if ! normalized=$(getopt -o he: --long help,environment:,conda-path: -n "execute-notebook" -- "$@"); then
    echo "failed to parse arguments" >&2
    exit 1
fi
eval set -- "$normalized"

environment=""

read -r -d "" help <<-EOF
Usage: $0 [options] input_path executed_path html_path

Run a notebook and convert it to html.

Options:
 -h, --help             display this help
     --conda-path       path to the conda executable
 -e, --environment      run the notebook in this environment
EOF

while true; do
    case "$1" in
        -h|--help)
            echo "$help"
            exit 0
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

input_notebook="$1"
output_notebook="$2"
output_html="$3"

if [[ "$environment" != "" && "$conda_path" == "" ]]; then
    echo "need the conda path when activating a environment" >&2
    exit 3
elif [[ "$environment" != "" && "$conda_path" != "" ]]; then
    conda_root="$(dirname "$(dirname "$conda_path")")"
    # shellcheck source=/dev/null
    source "$conda_root/etc/profile.d/conda.sh"
    conda activate "$environment"
fi

/usr/bin/time -v jupyter nbconvert --execute --allow-errors \
              "$input_notebook" \
              --to notebook \
              --output "$output_notebook"

/usr/bin/time -v jupyter nbconvert --allow-errors \
              "$output_notebook" \
              --to html \
              --output "$output_html"
