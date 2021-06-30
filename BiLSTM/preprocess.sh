# https://www.cyberciti.biz/faq/find-command-exclude-ignore-files/
find ./data -maxdepth 1 ! -name 'SafeSend-u6etyJgw6CwkFGXt' ! -name 'SafeSend-u6etyJgw6CwkFGXt.zip'  ! -path './data' -exec rm -r '{}' +

python build_onto_dataset.py
