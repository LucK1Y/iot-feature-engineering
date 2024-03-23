# Integration of Malware Dataset into Fedstellar
To integrate the malware dataset into Fedstellar, follow these steps:

- Copy the malwares folder into the directory **fedstellar/learning/pytorch/malwares**.
- Modify the file **fedstellar/node_start.py** as follows:
    - Add the necessary import statements.
    - Include the malware option.


    ````python
    from fedstellar.learning.pytorch.malwares.malwares import MalwaresDataset
    from fedstellar.learning.pytorch.malwares.models.mlp import MalwaresModelMLP

    ...

        elif dataset == "MALWARE":
            dataset = MalwaresDataset(
                sub_id=idx, number_sub=n_nodes, iid=iid
            )
            if model_name == "MLP":
                model = MalwaresModelMLP(
                    input_size=dataset.train_set.__getitem__(0)[0].shape[0]
                )
            else:
                raise ValueError(f"Model {model} not supported")
    ````

- Organize the data files from **training/data/cleaned** into the following structure:
    ````shell
    fedstellar/data
    ├── user
    │   ├── 0.csv
    │   ├── 1.csv
    │   ├── 2.csv
    │   ├── 3.csv
    │   ├── 4.csv
    │   ├── 5.csv
    │   ├── 6.csv
    │   └── 7.csv
    └── merged_df.csv
    ````

- Add the option ``<option>MALWARE</option>`` to the frontend HTML file located at **fedstellar/frontend/templates/deployment.html**: 
    ````html
                <select class="form-control" id="datasetSelect" name="dataset" style="display: inline; width: 50%">
                    <option selected>MNIST</option>
                    <option>FashionMNIST</option>
                    <option>CIFAR10</option>
                    <option>SYSCALL</option>
                    <option>SYSCALL</option>
                    <option>MALWARE</option>
                </select>
    ````

- Optional steps:
    - To test the dataset and run the model locally without Docker, perform the following: 
        - Add the test_malwarefile to **fedstellar/test_malware.py**
        - Edit the path to a valid configuration file and run the test.
    - To generate the documentation, place the **start_doc.sh** script in the root directory.
