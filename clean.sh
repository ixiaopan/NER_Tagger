# https://www.cyberciti.biz/faq/find-command-exclude-ignore-files/
find ./data -maxdepth 1 ! -name 'glove.6B' ! -name 'glove.6B.zip' ! -name 'toy' ! -name 'SafeSend-u6etyJgw6CwkFGXt' ! -name 'SafeSend-u6etyJgw6CwkFGXt.zip'  ! -path './data' -exec rm -r '{}' +

python clean_onto_dataset.py

python split_onto_dataset.py

python build_onto_profile.py --data_dir='./data/bc'
python build_onto_profile.py --data_dir='./data/bn'
python build_onto_profile.py --data_dir='./data/mz'
python build_onto_profile.py --data_dir='./data/nw'
python build_onto_profile.py --data_dir='./data/tc'
python build_onto_profile.py --data_dir='./data/wb'
python build_onto_profile.py --data_dir='./data/pool'
