# https://www.cyberciti.biz/faq/find-command-exclude-ignore-files/
find ./data -maxdepth 1 ! -name 'splits' ! -name 'glove.6B' ! -name 'glove.6B.zip' ! -name 'toy' ! -name 'SafeSend-u6etyJgw6CwkFGXt' ! -name 'SafeSend-u6etyJgw6CwkFGXt.zip'  ! -path './data' -exec rm -r '{}' +

# python clean_onto_dataset.py

# split train, valid, test for each domain
# python split_onto_dataset.py

python clean_split_standard.py

# create data profile
# inDomain
python build_onto_profile.py --data_dir='./data/bc'
python build_onto_profile.py --data_dir='./data/bn'
python build_onto_profile.py --data_dir='./data/mz'
python build_onto_profile.py --data_dir='./data/nw'
python build_onto_profile.py --data_dir='./data/tc'
python build_onto_profile.py --data_dir='./data/wb'

# mulDomain shared crf
python build_onto_profile.py --data_dir='./data/pool'

# leave one domain out
python split_pool_loov.py

python build_onto_profile.py --data_dir='./data/pool_bc' --dataset='train,valid'
python build_onto_profile.py --data_dir='./data/pool_bn' --dataset='train,valid'
python build_onto_profile.py --data_dir='./data/pool_mz' --dataset='train,valid'
python build_onto_profile.py --data_dir='./data/pool_nw' --dataset='train,valid'
python build_onto_profile.py --data_dir='./data/pool_tc' --dataset='train,valid'
python build_onto_profile.py --data_dir='./data/pool_wb' --dataset='train,valid'
