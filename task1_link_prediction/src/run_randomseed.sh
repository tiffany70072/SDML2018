pip3 install --upgrade gensim
mkdir results
python3 get_data.py
wget -O data/emb_word2vec_300_randomseed.npy "https://www.dropbox.com/s/35ul1yx9yahhlbs/emb_word2vec_300_randomseed.npy?dl=1"
time python3 main_randomseed.py word2vec regcl 2-1-0 30
python3 data/pred-txt-to-csv.py results/word2vec_regcl_2-1-0.txt