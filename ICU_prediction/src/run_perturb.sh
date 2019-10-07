idxs="6 7 8 9 10"
for idx in $idxs;do
	python3 data_augmentation_perturb.py 1
	python3 data_augmentation_perturb.py 2 $idx
done