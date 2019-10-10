pip3 install --upgrade gensim
mkdir results
python3 get_data.py
time python3 main.py word2vec regcl 2-1-0 30
python3 data/pred-txt-to-csv.py results/word2vec_regcl_2-1-0.txt

# cd ../../ML1_2
# python3 test_random.py
# cd ../task2/src
