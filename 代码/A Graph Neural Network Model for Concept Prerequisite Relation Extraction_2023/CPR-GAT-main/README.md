
## Anonymous code repository for CPR-GAT

1. Clone the repository.
2. Create the environment from the `environment.yml` file using `conda env create -f environment.yml` command.
3. Download GloVe using `wget http://nlp.stanford.edu/data/glove.6B.zip` command.
4. Unzip the glove6B.zip using `unzip glove*.zip` command.
5. Use this file (`glove.6B.300d.txt`) or file path in the glove_path variable in feature.py file.
6. Change your current directory to the CPR-GAT folder using `cd CPR-GAT` command and all the code from this folder.
7. To create feature vector for node and edges, run `python3 MOOC-DSA/feature/feature.py` file.
8. For training and inference, run `python3 MOOC-DSA/model/gat-model.py` file.
