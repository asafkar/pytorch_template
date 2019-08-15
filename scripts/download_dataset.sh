mkdir -p data/datasets
touch data/datasets/.gitkeep  # prevents git from deleting these directories

cd data/datasets

# Download dataset
for dataset in "<data_set_name>.tgz" ; do
    echo "Downloading $dataset"
    wget <addr_of_dataset>/$dataset
    tar -xvf $dataset
    rm $dataset
done

cd ../../