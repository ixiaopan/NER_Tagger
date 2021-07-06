# https://www.cyberciti.biz/faq/find-command-exclude-ignore-files/
find ./data -maxdepth 1 ! -name 'glove.6B' ! -name 'glove.6B.zip' ! -name 'toy' ! -name 'SafeSend-u6etyJgw6CwkFGXt' ! -name 'SafeSend-u6etyJgw6CwkFGXt.zip'  ! -path './data' -exec rm -r '{}' +

python clean_onto_dataset.py
